"""
Data processing modules for AMG learning

This package contains dataset and dataloader classes for:
    - CNN data (pooled matrix images)
    - GNN data (sparse graph representations)
    - Unified data (both theta and P-value prediction)
"""

from .cnn_data_processing import (
    CSVDataset,
    create_dataloaders
)

from .gnn_data_processing import (
    GNNThetaDataset,
    GNNPValueDataset,
    create_theta_data_loaders,
    create_p_value_data_loaders
)

from .unified_data_processing import (
    UnifiedAMGDataset,
    UnifiedAMGDataLoader,
    create_unified_data_loaders,
    split_unified_csv
)

__all__ = [
    # CNN data
    'CSVDataset',
    'create_dataloaders',

    # GNN data
    'GNNThetaDataset',
    'GNNPValueDataset',
    'create_theta_data_loaders',
    'create_p_value_data_loaders',

    # Unified data
    'UnifiedAMGDataset',
    'UnifiedAMGDataLoader',
    'create_unified_data_loaders',
    'split_unified_csv'
]
