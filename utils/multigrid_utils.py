"""
Multigrid utilities for AMG operations

Includes:
    - Coarse grid operator computation
    - Relaxation (smoothing) matrices
    - Two-grid convergence analysis
    - Loss functions for P-value prediction
"""

import torch
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import inv as sparse_inv, eigsh
from typing import Tuple, Optional


def compute_coarse_grid_operator(A: np.ndarray, P: np.ndarray, R: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute coarse grid operator A_c = R * A * P

    Parameters:
        A: fine grid operator [n, n]
        P: prolongation matrix [n, n_c]
        R: restriction matrix [n_c, n] (default: R = P^T)

    Returns:
        A_c: coarse grid operator [n_c, n_c]
    """
    if R is None:
        R = P.T

    A_c = R @ A @ P
    return A_c


def jacobi_relaxation_matrix(A: np.ndarray, omega: float = 0.8) -> np.ndarray:
    """
    Compute Jacobi relaxation matrix S = I - omega * D^{-1} * A

    Parameters:
        A: system matrix [n, n]
        omega: damping parameter

    Returns:
        S: Jacobi iteration matrix [n, n]
    """
    n = A.shape[0]
    I = np.eye(n)

    # Extract diagonal
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)

    S = I - omega * D_inv @ A
    return S


def gauss_seidel_relaxation_matrix(A: np.ndarray) -> np.ndarray:
    """
    Compute Gauss-Seidel relaxation matrix S = -(D+L)^{-1} * U

    Parameters:
        A: system matrix [n, n]

    Returns:
        S: Gauss-Seidel iteration matrix [n, n]
    """
    # Split A into L, D, U
    L = np.tril(A, k=-1)  # Lower triangular
    D = np.diag(np.diag(A))  # Diagonal
    U = np.triu(A, k=1)  # Upper triangular

    # Compute -(D+L)^{-1} * U
    DL = D + L
    DL_inv = np.linalg.inv(DL)
    S = -DL_inv @ U

    return S


def two_grid_error_matrix(A: np.ndarray, P: np.ndarray, S: np.ndarray,
                          R: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute two-grid error propagation matrix M = S * C * S

    where C = I - P * A_c^{-1} * R * A is the coarse correction operator

    Parameters:
        A: fine grid operator [n, n]
        P: prolongation matrix [n, n_c]
        S: smoother iteration matrix [n, n]
        R: restriction matrix [n_c, n] (default: R = P^T)

    Returns:
        M: two-grid error matrix [n, n]
    """
    n = A.shape[0]
    I = np.eye(n)

    if R is None:
        R = P.T

    # Compute coarse grid operator
    A_c = compute_coarse_grid_operator(A, P, R)

    # Invert coarse grid operator
    A_c_inv = np.linalg.inv(A_c)

    # Compute coarse correction operator C = I - P * A_c^{-1} * R * A
    C = I - P @ A_c_inv @ R @ A

    # Two-grid error matrix: M = S * C * S
    M = S @ C @ S

    return M


def convergence_factor_frobenius(M: np.ndarray) -> float:
    """
    Compute Frobenius norm of error matrix ||M||_F

    This is a differentiable approximation to the spectral radius

    Parameters:
        M: error propagation matrix [n, n]

    Returns:
        Frobenius norm
    """
    return np.linalg.norm(M, ord='fro')


def convergence_factor_spectral(M: np.ndarray) -> float:
    """
    Compute spectral radius rho(M) = max|lambda_i(M)|

    This is the actual convergence factor but is not differentiable

    Parameters:
        M: error propagation matrix [n, n]

    Returns:
        Spectral radius
    """
    eigenvalues = np.linalg.eigvals(M)
    return np.max(np.abs(eigenvalues))


class TwoGridLoss(torch.nn.Module):
    """
    Two-grid convergence loss for training P-value predictor

    Computes loss based on the two-grid error matrix:
        M = S * C * S
    where C = I - P * A_c^{-1} * R * A

    Loss can be:
        - Frobenius norm: ||M||_F (differentiable, for training)
        - Spectral radius: rho(M) (for evaluation)
    """

    def __init__(self, loss_type: str = 'frobenius'):
        """
        Initialize loss

        Parameters:
            loss_type: 'frobenius' or 'spectral'
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(self, A_list, P_list, S_list):
        """
        Compute loss for a batch

        Parameters:
            A_list: list of system matrices
            P_list: list of predicted prolongation matrices
            S_list: list of smoother matrices

        Returns:
            Average loss over batch
        """
        batch_loss = 0.0

        for A, P, S in zip(A_list, P_list, S_list):
            # Convert to numpy if needed
            if isinstance(A, torch.Tensor):
                A = A.detach().cpu().numpy()
            if isinstance(P, torch.Tensor):
                P = P.detach().cpu().numpy()
            if isinstance(S, torch.Tensor):
                S = S.detach().cpu().numpy()

            # Compute two-grid error matrix
            M = two_grid_error_matrix(A, P, S)

            # Compute loss
            if self.loss_type == 'frobenius':
                loss = convergence_factor_frobenius(M)
            else:  # spectral
                loss = convergence_factor_spectral(M)

            batch_loss += loss

        return batch_loss / len(A_list)


def compute_two_grid_convergence(A: np.ndarray, P: np.ndarray, S: np.ndarray,
                                 num_iterations: int = 10) -> Tuple[float, np.ndarray]:
    """
    Compute two-grid convergence factor by power iteration

    Parameters:
        A: system matrix
        P: prolongation matrix
        S: smoother matrix
        num_iterations: number of power iterations

    Returns:
        Tuple of (convergence_factor, error_history)
    """
    n = A.shape[0]

    # Random initial error
    e = np.random.randn(n)
    e = e / np.linalg.norm(e)

    # Compute two-grid error matrix
    M = two_grid_error_matrix(A, P, S)

    # Power iteration
    error_norms = []
    for _ in range(num_iterations):
        e = M @ e
        error_norm = np.linalg.norm(e)
        error_norms.append(error_norm)

    # Estimate convergence factor from error reduction
    error_norms = np.array(error_norms)
    convergence_factor = np.exp(np.mean(np.log(error_norms[1:] / error_norms[:-1])))

    return convergence_factor, error_norms


def sparse_two_grid_error_matrix(A_sparse, P_sparse, S_sparse,
                                 R_sparse=None) -> csr_matrix:
    """
    Compute two-grid error matrix for sparse matrices

    Parameters:
        A_sparse: sparse system matrix
        P_sparse: sparse prolongation matrix
        S_sparse: sparse smoother matrix
        R_sparse: sparse restriction matrix (default: P^T)

    Returns:
        M: sparse two-grid error matrix
    """
    from scipy.sparse import eye, csr_matrix
    from scipy.sparse.linalg import spsolve

    n = A_sparse.shape[0]
    I = eye(n, format='csr')

    if R_sparse is None:
        R_sparse = P_sparse.T

    # Compute coarse grid operator A_c = R * A * P
    A_c = R_sparse @ A_sparse @ P_sparse

    # For sparse inversion, we compute P * (A_c \ (R * A))
    # which is more efficient than computing A_c^{-1} explicitly
    RA = R_sparse @ A_sparse

    # Solve A_c * X = RA for X
    # X will have shape [n_c, n]
    X = np.zeros((A_c.shape[0], n))
    for i in range(n):
        X[:, i] = spsolve(A_c, RA[:, i].toarray().flatten())

    # Coarse correction: C = I - P * X
    C = I - P_sparse @ csr_matrix(X)

    # Two-grid error: M = S * C * S
    M = S_sparse @ C @ S_sparse

    return M
