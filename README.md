# PACSProject

This is a project for the PACS course.  
Its research topic is accelerating AMG by deep learning method.

## Project structure

```
|--PACSProject
    |--include
        |--*.hpp
    |--src
        |--*.cpp
    |--model
        |--model.py
    |--data
        |--data_processing.py
    |--main.py
```


## Differential Problems

### **Advection Equation with Discontinuous Galerkin Method**
**Problem Statement:**  
Solve the scalar linear advection equation:  
$$
\frac{\partial u}{\partial t} + \boldsymbol{\beta} \cdot \nabla u = 0
$$  
in a domain \(\Omega \subset \mathbb{R}^2\) over time \(t \in (0, T]\), with initial condition \(u(\mathbf{x}, 0) = u_0(\mathbf{x})\) and inflow boundary conditions \(u = g\) on \(\partial\Omega_{\text{in}} = \{\mathbf{x} \in \partial\Omega : \boldsymbol{\beta} \cdot \mathbf{n} < 0\}\).

**Weak Formulation (Discontinuous Galerkin):**  
Multiply by test function \(v\), integrate over cell \(K\), and apply integration by parts:  
$$
\int_K \frac{\partial u}{\partial t} v \, d\mathbf{x} - \int_K u \boldsymbol{\beta} \cdot \nabla v \, d\mathbf{x} + \int_{\partial K} (\boldsymbol{\beta} \cdot \mathbf{n}) u v \, ds = 0.
$$  
Replace boundary term with upwind flux \(\hat{u}\):  
$$
\hat{u} = 
\begin{cases} 
u^+ & \text{if } \boldsymbol{\beta} \cdot \mathbf{n}_K < 0, \\
u^- & \text{otherwise},
\end{cases}
$$  
where \(u^-\) and \(u^+\) denote solutions inside/outside \(K\). The semi-discrete form is:  
$$
\int_K \frac{\partial u_h}{\partial t} v_h \, d\mathbf{x} - \int_K u_h \boldsymbol{\beta} \cdot \nabla v_h \, d\mathbf{x} + \int_{\partial K} (\boldsymbol{\beta} \cdot \mathbf{n}) \hat{u} v_h \, ds = 0.
$$

**Time Discretization:**  
Use a 3rd-order explicit Runge-Kutta (RK) scheme:  
1. \(\mathbf{k}_1 = \mathbf{M}^{-1} \mathbf{F}(t_n, \mathbf{u}_n)\),  
2. \(\mathbf{k}_2 = \mathbf{M}^{-1} \mathbf{F}(t_n + \Delta t, \mathbf{u}_n + \Delta t \mathbf{k}_1)\),  
3. \(\mathbf{u}_{n+1} = \mathbf{u}_n + \frac{\Delta t}{6} (\mathbf{k}_1 + 4\mathbf{k}_2 + \mathbf{k}_3)\),  
where \(\mathbf{M}\) is the mass matrix, and \(\mathbf{F}\) encodes spatial operators.

---

### **Laplace Equation with Boundary Element Method (BEM)**
**Problem Statement:**  
Solve the Laplace equation in an exterior domain \(\Omega = \mathbb{R}^2 \setminus \Gamma\) (where \(\Gamma\) is a bounded hole):  
$$
-\Delta u = 0 \quad \text{in} \quad \Omega,
$$  
with Dirichlet condition \(u = g\) on \(\partial\Omega\), and \(u \to 0\) as \(|\mathbf{x}| \to \infty\).

**Boundary Integral Formulation:**  
Represent the solution via the **double layer potential**:  
$$
u(\mathbf{x}) = \int_{\Gamma} \mu(\mathbf{y}) \frac{\partial G(\mathbf{x}, \mathbf{y})}{\partial \mathbf{n_y}} \, dS(\mathbf{y}),  
$$  
where \(G(\mathbf{x}, \mathbf{y}) = -\frac{1}{2\pi} \ln|\mathbf{x} - \mathbf{y}|\) is the Laplace Green's function. The density \(\mu\) solves the boundary integral equation:  
$$
-\mu(\mathbf{x}) + \int_{\Gamma} \mu(\mathbf{y}) \frac{\partial G(\mathbf{x}, \mathbf{y})}{\partial \mathbf{n_y}} \, dS(\mathbf{y}) = g(\mathbf{x}), \quad \mathbf{x} \in \Gamma.
$$

**Weak Formulation (Galerkin BEM):**  
Discretize \(\Gamma\) into boundary elements. Seek \(\mu_h \in V_h\) (continuous FE space) such that:  
$$
\forall v_h \in V_h: \quad 
\left\langle - \mu_h, v_h \right\rangle_\Gamma + 
\left\langle \int_{\Gamma} \mu_h(\mathbf{y}) \frac{\partial G(\cdot, \mathbf{y})}{\partial \mathbf{n_y}} dS(\mathbf{y}), v_h \right\rangle_\Gamma = \langle g, v_h \rangle_\Gamma.
$$  
This yields the linear system:  
$$
(\mathbf{A} + \mathbf{B}) \boldsymbol{\mu} = \mathbf{b},
$$  
where:  
- \(A_{ij} = -\int_{\Gamma} \varphi_j \varphi_i \, dS\),  
- \(B_{ij} = \int_{\Gamma} \int_{\Gamma} \varphi_j(\mathbf{y}) \frac{\partial G(\mathbf{x}, \mathbf{y})}{\partial \mathbf{n_y}} \varphi_i(\mathbf{x}) \, dS_{\mathbf{y}} dS_{\mathbf{x}}\),  
- \(b_i = \int_{\Gamma} g \varphi_i \, dS\).

**Key Challenges:**  
- **Singular Integrals:** Handle weak singularities in \(\frac{\partial G}{\partial \mathbf{n_y}}\) via coordinate transformations.  
- **Mesh Refinement:** Resolve sharp corners on \(\Gamma\) to ensure solution accuracy.

---
