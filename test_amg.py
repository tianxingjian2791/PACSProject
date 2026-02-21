# manual construction of a two-level AMG hierarchy
from scipy.sparse.linalg import cg
import scipy as sp
import numpy as np
from pyamg.gallery import poisson, stencil_grid
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.aggregation import smoothed_aggregation_solver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.multilevel import MultilevelSolver
from pyamg.strength import classical_strength_of_connection
from pyamg.classical.interpolate import direct_interpolation
from pyamg.classical.split import RS

import torch
import torch.nn as nn
from model.cnn_model import CNNModel

def safe_matvec(A, x):
    """Try to adapt to different types of A: sparse matrix/ndarray/LinearOperator"""
    try:
        # First try the most common case: A is a sparse matrix(e.g. csr_matrix)
        return A @ x
    except Exception:
        try:
            return A.dot(x)
        except Exception:
            # If A is a LinearOperator, it should have a .matvec method
            return A.matvec(x)

def cg_with_history(A, b, ml = None, x0=None, tol=1e-8, maxiter=None):
    hist = {'iter': 0, 'resnorm': [], 'x': []}
    # hist = {'iter': 0, 'x': []}

    def cb(xk):
        hist['iter'] += 1
        hist['x'].append(xk.copy())
        r = b - safe_matvec(A, xk)
        hist['resnorm'].append(np.linalg.norm(r))
        
    M = ml.aspreconditioner(cycle='V')             # preconditioner    
    x, info = cg(A, b, x0=x0, rtol=tol, maxiter=maxiter, callback=cb, M=M)
    return x, info, hist

# compute necessary operators
# A = poisson((100, 100), format='csr')  # 2D Poisson problem
# 2D diffusion problem with variable coefficients
nx = ny = 100
step_size = 1.0/(nx+1)
epsilon_list = np.linspace(0.1, 10, 100)  # 100 varying diffusion coefficients
sten_dict = {epsilon: diffusion_stencil_2d(epsilon=epsilon, theta=0, type='FE') for epsilon in epsilon_list}
A_list = [stencil_grid(sten_dict[epsilon], (nx, ny), format='csr') for epsilon in epsilon_list]

# Load the trained CNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "./weights/D_stage1_cnn/D_stage1_cnn/best_model.pt"
model = CNNModel(in_channels=1, hidden_channels=64)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model = model.to(device)

n1 = 0  # count the number of times choosing the best theta is better than choosing 0.25
n2 = 0  # count the number of times choosing 0.25 is better than choosing the best theta
n3 = 0  # count the number of times choosing the best theta is the same as choosing 0.25
np.random.seed(42)  # for reproducibility
for i, A in enumerate(A_list):
    print(f"Matrix {i}: shape={A.shape}, nnz={A.nnz}, epsilon={epsilon_list[i]:.2f}")
    b = np.random.rand(A.shape[0])  # random RHS
    data = torch.zeros((50, 2502))  # store the data as the model input
    for j, theta in enumerate(np.linspace(0.02, 1.0, 50)):
        C = classical_strength_of_connection(A, theta=theta)
        splitting = RS(C)
        P = direct_interpolation(A, C, splitting)
        R = P.T
        A_second_level = R @ A @ P  # coarse-level matrix
        
        # make levels[1].A become a pooled tensor with shape (50, 50) from the original shape (10000, 5000)
        pool = nn.AvgPool2d(kernel_size=(A_second_level.shape[0]//50, A_second_level.shape[1]//50))
        data[j, 0] = theta
        data[j, 1] = -np.log2(step_size)
        data[j, 2:] = pool(torch.tensor(A_second_level.toarray(), dtype=torch.float32).reshape(1,1,A_second_level.shape[0],A_second_level.shape[1])).flatten()
    
    data = data.to(device)
    output = model(data)  # convergence factor prediction
    idx = torch.argmin(output)  # find the index of the best convergence factor
    best_theta = np.linspace(0.02, 1.0, 50)[idx.item()]

    C1 = classical_strength_of_connection(A, theta=best_theta)
    C2 = classical_strength_of_connection(A, theta=0.25)  # a common choice for theta in classical AMG, used for comparison
    splitting1 = RS(C1)
    splitting2 = RS(C2)
    P1 = direct_interpolation(A, C1, splitting1)
    P2 = direct_interpolation(A, C2, splitting2)
    R1 = P1.T
    R2 = P2.T
    # store first level data
    levels1 = []
    levels2 = []
    levels1.append(MultilevelSolver.Level())
    levels2.append(MultilevelSolver.Level())
    levels1.append(MultilevelSolver.Level())
    levels2.append(MultilevelSolver.Level())
    levels1[0].A = A
    levels2[0].A = A
    levels1[0].C = C1
    levels2[0].C = C2
    levels1[0].splitting = splitting1
    levels2[0].splitting = splitting2
    levels1[0].P = P1
    levels2[0].P = P2
    levels1[0].R = R1
    levels2[0].R = R2
    # store second level data
    levels1[1].A = R1 @ A @ P1                      # coarse-level matrix    
    levels2[1].A = R2 @ A @ P2                      # coarse-level matrix    

    # create MultilevelSolver
    ml1 = MultilevelSolver(levels1, coarse_solver='splu')
    ml2 = MultilevelSolver(levels2, coarse_solver='splu')
    change_smoothers(ml1, presmoother=('gauss_seidel', {'iterations': 1, 'sweep': 'symmetric'}),
                    postsmoother=('gauss_seidel', {'iterations': 1, 'sweep': 'symmetric'}))
    change_smoothers(ml2, presmoother=('gauss_seidel', {'iterations': 1, 'sweep': 'symmetric'}),
                    postsmoother=('gauss_seidel', {'iterations': 1, 'sweep': 'symmetric'}))
    # print(ml)

    x1, info1, hist1 = cg_with_history(A, b, ml1, tol=1e-8, maxiter=30) # solve with CG
    rho1 = hist1['resnorm'][-1]**(1/hist1['iter']) if hist1['iter'] > 0 else float('inf')
    print("\nThe number of iterations:", hist1['iter'], "The last residual norm:", hist1['resnorm'][-1])
    print("\nexit code:", info1)

    x2, info2, hist2 = cg_with_history(A, b, ml2, tol=1e-8, maxiter=30) # solve with CG
    rho2 = hist2['resnorm'][-1]**(1/hist2['iter']) if hist2['iter'] > 0 else float('inf')
    print("\nThe number of iterations:", hist2['iter'], "The last residual norm:", hist2['resnorm'][-1])
    print("\nexit code:", info2)

    if rho1 < rho2:
        n1 += 1
    elif rho1 > rho2:
        n2 += 1
    else:
        n3 += 1


print(f"\nBest theta is better than 0.25 in {n1} cases, while 0.25 is better in {n2} cases.")

"""
# example of classical AMG usage
from pyamg.gallery import poisson
from pyamg import ruge_stuben_solver
A = poisson((10,),format='csr')
ml = ruge_stuben_solver(A,max_coarse=3)
print(ml)


# example of coarse grid solver usage
import numpy as np
from pyamg.gallery import poisson
from pyamg import coarse_grid_solver
A = poisson((10, 10), format='csr')
b = A @ np.ones(A.shape[0])
cgs = coarse_grid_solver('lu')
x = cgs(A, b)
print("Solution:", x)
"""
