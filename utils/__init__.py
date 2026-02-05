"""
Utility modules for AMG learning

This package contains utility functions for:
    - Multigrid operations (coarse grid, relaxation, two-grid analysis)
    - AMG operations (C/F splitting, prolongation)
    - Training utilities (checkpointing, logging, metrics)
"""

from .multigrid_utils import (
    compute_coarse_grid_operator,
    jacobi_relaxation_matrix,
    gauss_seidel_relaxation_matrix,
    two_grid_error_matrix,
    convergence_factor_frobenius,
    convergence_factor_spectral,
    TwoGridLoss,
    compute_two_grid_convergence,
    sparse_two_grid_error_matrix
)

from .amg_utils import (
    classical_cf_splitting,
    compute_strength_matrix,
    compute_baseline_prolongation,
    extract_coarse_nodes,
    extract_fine_nodes,
    compute_interpolation_sparsity_pattern,
    visualize_cf_splitting
)

from .training_utils import (
    Checkpointer,
    MetricsLogger,
    compute_accuracy,
    compute_relative_error,
    EarlyStopping,
    set_random_seed,
    count_parameters,
    get_device
)

__all__ = [
    # Multigrid utilities
    'compute_coarse_grid_operator',
    'jacobi_relaxation_matrix',
    'gauss_seidel_relaxation_matrix',
    'two_grid_error_matrix',
    'convergence_factor_frobenius',
    'convergence_factor_spectral',
    'TwoGridLoss',
    'compute_two_grid_convergence',
    'sparse_two_grid_error_matrix',

    # AMG utilities
    'classical_cf_splitting',
    'compute_strength_matrix',
    'compute_baseline_prolongation',
    'extract_coarse_nodes',
    'extract_fine_nodes',
    'compute_interpolation_sparsity_pattern',
    'visualize_cf_splitting',

    # Training utilities
    'Checkpointer',
    'MetricsLogger',
    'compute_accuracy',
    'compute_relative_error',
    'EarlyStopping',
    'set_random_seed',
    'count_parameters',
    'get_device'
]
