"""
AMG utilities for C/F splitting and prolongation construction

Implements classical AMG operations:
    - Classical Ruge-Stüben C/F splitting
    - Baseline prolongation matrix construction
    - Matrix conversion utilities
"""

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from typing import Tuple, List, Set


def classical_cf_splitting(A: csr_matrix, theta: float = 0.25) -> np.ndarray:
    """
    Classical Ruge-Stüben C/F splitting

    Parameters:
        A: sparse system matrix [n, n]
        theta: strong threshold parameter

    Returns:
        cf_markers: array where 1=coarse, 0=fine [n]
    """
    n = A.shape[0]

    # Compute strength of connection matrix
    S = compute_strength_matrix(A, theta)

    # Initialize all as undecided
    cf_markers = np.zeros(n, dtype=int)  # 0=undecided, 1=coarse, -1=fine

    # Compute measures (number of strong connections)
    lambda_vals = np.array(S.sum(axis=1)).flatten()

    # Create coarse and fine sets
    coarse_nodes = set()
    fine_nodes = set()
    undecided = set(range(n))

    while undecided:
        # Find node with maximum measure
        max_lambda = -1
        max_node = -1
        for i in undecided:
            if lambda_vals[i] > max_lambda:
                max_lambda = lambda_vals[i]
                max_node = i

        # Make it coarse
        coarse_nodes.add(max_node)
        cf_markers[max_node] = 1
        undecided.remove(max_node)

        # Find strongly connected neighbors
        neighbors = S[max_node].nonzero()[1]

        for j in neighbors:
            if j in undecided:
                # Make it fine
                fine_nodes.add(j)
                cf_markers[j] = -1
                undecided.remove(j)

                # Update measures of its neighbors
                j_neighbors = S[j].nonzero()[1]
                for k in j_neighbors:
                    if k in undecided:
                        lambda_vals[k] += 1

    # Return binary array (1=coarse, 0=fine)
    return (cf_markers == 1).astype(int)


def compute_strength_matrix(A: csr_matrix, theta: float) -> csr_matrix:
    """
    Compute strength of connection matrix

    Node i is strongly connected to j if:
        -a_ij >= theta * max_k(-a_ik)

    Parameters:
        A: system matrix
        theta: strong threshold

    Returns:
        S: strength matrix (binary, 1 = strong connection)
    """
    n = A.shape[0]
    S = lil_matrix((n, n), dtype=float)

    A_coo = A.tocoo()

    # Compute max off-diagonal for each row
    max_offdiag = np.zeros(n)
    for i, j, val in zip(A_coo.row, A_coo.col, A_coo.data):
        if i != j:  # Off-diagonal
            max_offdiag[i] = max(max_offdiag[i], -val)

    # Determine strong connections
    for i, j, val in zip(A_coo.row, A_coo.col, A_coo.data):
        if i != j:  # Off-diagonal only
            if -val >= theta * max_offdiag[i]:
                S[i, j] = 1.0

    return S.tocsr()


def compute_baseline_prolongation(A: csr_matrix, cf_markers: np.ndarray) -> csr_matrix:
    """
    Compute baseline prolongation matrix using direct interpolation

    For coarse nodes: P[c, c] = 1
    For fine nodes: P[f, c] = interpolation weights

    Parameters:
        A: system matrix [n, n]
        cf_markers: C/F markers (1=coarse, 0=fine) [n]

    Returns:
        P: prolongation matrix [n, n_c]
    """
    n = A.shape[0]
    coarse_nodes = np.where(cf_markers == 1)[0]
    fine_nodes = np.where(cf_markers == 0)[0]
    num_coarse = len(coarse_nodes)

    # Map coarse node index to column in P
    coarse_to_col = {c: i for i, c in enumerate(coarse_nodes)}

    # Build prolongation matrix
    P = lil_matrix((n, num_coarse))

    # Identity for coarse nodes
    for c in coarse_nodes:
        P[c, coarse_to_col[c]] = 1.0

    # Interpolation for fine nodes
    A_coo = A.tocoo()

    for f in fine_nodes:
        # Find strongly connected coarse neighbors
        row_start = A.indptr[f]
        row_end = A.indptr[f + 1]

        coarse_neighbors = []
        coarse_weights = []

        diagonal_val = 0.0

        for idx in range(row_start, row_end):
            j = A.indices[idx]
            val = A.data[idx]

            if f == j:
                diagonal_val = val
            elif cf_markers[j] == 1:  # Coarse neighbor
                coarse_neighbors.append(j)
                coarse_weights.append(-val)  # Off-diagonal is negative

        if len(coarse_neighbors) == 0:
            # No coarse neighbors, assign to nearest coarse node
            # (simplified approach)
            nearest_c = coarse_nodes[0]
            P[f, coarse_to_col[nearest_c]] = 1.0
        else:
            # Direct interpolation: normalize weights
            total_weight = sum(coarse_weights)
            if abs(total_weight) > 1e-12:
                for c, w in zip(coarse_neighbors, coarse_weights):
                    P[f, coarse_to_col[c]] = w / total_weight

    return P.tocsr()


def extract_coarse_nodes(cf_markers: np.ndarray) -> np.ndarray:
    """
    Extract coarse node indices from C/F markers

    Parameters:
        cf_markers: C/F markers (1=coarse, 0=fine)

    Returns:
        coarse_nodes: indices of coarse nodes
    """
    return np.where(cf_markers == 1)[0]


def extract_fine_nodes(cf_markers: np.ndarray) -> np.ndarray:
    """
    Extract fine node indices from C/F markers

    Parameters:
        cf_markers: C/F markers (1=coarse, 0=fine)

    Returns:
        fine_nodes: indices of fine nodes
    """
    return np.where(cf_markers == 0)[0]


def compute_interpolation_sparsity_pattern(A: csr_matrix, cf_markers: np.ndarray) -> Tuple[List, List]:
    """
    Compute sparsity pattern for prolongation matrix

    Returns which coarse nodes each fine node should interpolate from

    Parameters:
        A: system matrix
        cf_markers: C/F markers

    Returns:
        Tuple of (fine_nodes, coarse_neighbors_per_fine)
    """
    fine_nodes = extract_fine_nodes(cf_markers)
    coarse_nodes = extract_coarse_nodes(cf_markers)

    coarse_set = set(coarse_nodes)
    coarse_neighbors = []

    for f in fine_nodes:
        # Find strongly connected coarse neighbors
        row_start = A.indptr[f]
        row_end = A.indptr[f + 1]

        f_coarse_neighbors = []
        for idx in range(row_start, row_end):
            j = A.indices[idx]
            if j in coarse_set:
                f_coarse_neighbors.append(j)

        coarse_neighbors.append(f_coarse_neighbors)

    return fine_nodes, coarse_neighbors


def visualize_cf_splitting(cf_markers: np.ndarray, grid_size: int = None):
    """
    Visualize C/F splitting (for 2D grid problems)

    Parameters:
        cf_markers: C/F markers
        grid_size: size of 2D grid (if structured grid)
    """
    if grid_size is None:
        grid_size = int(np.sqrt(len(cf_markers)))

    n = len(cf_markers)
    if grid_size * grid_size != n:
        print("Warning: not a square grid, cannot visualize")
        return

    grid = cf_markers.reshape((grid_size, grid_size))

    print("\nC/F Splitting Visualization:")
    print("C = Coarse (1), F = Fine (0)")
    print(grid)
    print(f"\nCoarse nodes: {np.sum(cf_markers == 1)}")
    print(f"Fine nodes: {np.sum(cf_markers == 0)}")
    print(f"Coarsening ratio: {np.sum(cf_markers == 1) / n:.2%}")
