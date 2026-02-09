"""
Unified data processing for two-stage AMG learning pipeline

This module provides unified dataset and loader classes that support:
    - Stage 1: Theta prediction (CNN or GNN)
    - Stage 2: P-value prediction (GNN)
    - Full pipeline: Both stages together
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import csv
from typing import Tuple, Optional, Literal
from scipy.sparse import csr_matrix

# Import existing data processing
from .cnn_data_processing import CSVDataset
from .gnn_data_processing import GNNThetaDataset, GNNPValueDataset


class UnifiedAMGDataset(Dataset):
    """
    Unified dataset supporting both Stage 1 and Stage 2 data

    This dataset can provide:
        - Stage 1 CNN data: pooled matrix images
        - Stage 1 GNN data: sparse graph representation
        - Stage 2 GNN data: graph with C/F splitting and baseline P
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal['train', 'test'] = 'train',
        stage1_type: Literal['CNN', 'GNN'] = 'CNN',
        csv_file: str = 'unified.csv'
    ):
        """
        Initialize unified dataset
        """
        self.root_dir = root_dir
        self.split = split
        self.stage1_type = stage1_type
        self.csv_file = csv_file

        # Paths for different data formats
        self.theta_cnn_path = os.path.join(root_dir, split, 'theta_cnn')
        self.theta_gnn_path = os.path.join(root_dir, split, 'theta_gnn')
        self.p_value_path = os.path.join(root_dir, split, 'p_value')

        # Load appropriate Stage 1 dataset
        if stage1_type == 'CNN':
            # Load CNN data (pooled matrices)
            csv_path = os.path.join(self.theta_cnn_path, 'raw', csv_file)
            if os.path.exists(csv_path):
                self.stage1_dataset = CSVDataset(csv_path)
            else:
                print(f"Warning: {csv_path} not found. Stage 1 CNN data unavailable.")
                self.stage1_dataset = None
        else:  # GNN
            # Load GNN data (sparse graphs)
            if os.path.exists(self.theta_gnn_path):
                self.stage1_dataset = GNNThetaDataset(
                    root=self.theta_gnn_path,
                    csv_file=csv_file
                )
            else:
                print(f"Warning: {self.theta_gnn_path} not found. Stage 1 GNN data unavailable.")
                self.stage1_dataset = None

        # Load Stage 2 dataset (always GNN)
        if os.path.exists(self.p_value_path):
            self.stage2_dataset = GNNPValueDataset(
                root=self.p_value_path,
                csv_file=csv_file,
                node_indicators=True,
                edge_indicators=True
            )
        else:
            print(f"Warning: {self.p_value_path} not found. Stage 2 data unavailable.")
            self.stage2_dataset = None

    def __len__(self):
        """Return length based on available datasets"""
        if self.stage1_dataset is not None:
            return len(self.stage1_dataset)
        elif self.stage2_dataset is not None:
            return len(self.stage2_dataset)
        else:
            return 0

    def get_stage1_sample(self, idx: int):
        """Get Stage 1 sample (CNN or GNN format)"""
        if self.stage1_dataset is None:
            raise ValueError("Stage 1 dataset not loaded")
        return self.stage1_dataset[idx]

    def get_stage2_sample(self, idx: int):
        """Get Stage 2 sample (GNN P-value format)"""
        if self.stage2_dataset is None:
            raise ValueError("Stage 2 dataset not loaded")
        return self.stage2_dataset[idx]

    def __getitem__(self, idx: int):
        """
        Get unified sample containing both Stage 1 and Stage 2 data

        Returns:
            Tuple of (stage1_data, stage2_data)
        """
        stage1_data = self.get_stage1_sample(idx) if self.stage1_dataset else None
        stage2_data = self.get_stage2_sample(idx) if self.stage2_dataset else None
        return stage1_data, stage2_data


class UnifiedAMGDataLoader:
    """
    Unified data loader for two-stage pipeline

    Provides separate loaders for Stage 1 and Stage 2,
    or combined loading for joint training
    """

    def __init__(
        self,
        dataset: UnifiedAMGDataset,
        batch_size_stage1: int = 32,
        batch_size_stage2: int = 16,
        shuffle: bool = True,
        num_workers: int = 4
    ):
        """
        Initialize unified data loader
        """
        self.dataset = dataset
        self.batch_size_stage1 = batch_size_stage1
        self.batch_size_stage2 = batch_size_stage2
        self.shuffle = shuffle
        self.num_workers = num_workers

    def get_stage1_loader(self):
        """Get DataLoader for Stage 1 only"""
        if self.dataset.stage1_dataset is None:
            raise ValueError("Stage 1 dataset not available")

        if self.dataset.stage1_type == 'CNN':
            # CNN uses regular PyTorch DataLoader
            return torch.utils.data.DataLoader(
                self.dataset.stage1_dataset,
                batch_size=self.batch_size_stage1,
                shuffle=self.shuffle,
                num_workers=self.num_workers
            )
        else:  # GNN
            # GNN uses PyG DataLoader
            return DataLoader(
                self.dataset.stage1_dataset,
                batch_size=self.batch_size_stage1,
                shuffle=self.shuffle,
                num_workers=self.num_workers
            )

    def get_stage2_loader(self):
        """Get DataLoader for Stage 2 only"""
        if self.dataset.stage2_dataset is None:
            raise ValueError("Stage 2 dataset not available")

        return DataLoader(
            self.dataset.stage2_dataset,
            batch_size=self.batch_size_stage2,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )


def create_unified_data_loaders(
    data_dir: str,
    stage1_type: Literal['CNN', 'GNN'] = 'CNN',
    csv_file: str = 'unified.csv',
    batch_size_stage1: int = 32,
    batch_size_stage2: int = 16,
    num_workers: int = 4
) -> Tuple:
    """
    Factory function to create unified train and test loaders

    Returns:
        Tuple of (train_loader, test_loader) - UnifiedAMGDataLoader instances
    """
    train_dataset = UnifiedAMGDataset(
        root_dir=data_dir,
        split='train',
        stage1_type=stage1_type,
        csv_file=csv_file
    )

    test_dataset = UnifiedAMGDataset(
        root_dir=data_dir,
        split='test',
        stage1_type=stage1_type,
        csv_file=csv_file
    )

    train_loader = UnifiedAMGDataLoader(
        dataset=train_dataset,
        batch_size_stage1=batch_size_stage1,
        batch_size_stage2=batch_size_stage2,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = UnifiedAMGDataLoader(
        dataset=test_dataset,
        batch_size_stage1=batch_size_stage1,
        batch_size_stage2=batch_size_stage2,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"Unified training samples: {len(train_dataset)}")
    print(f"Unified test samples: {len(test_dataset)}")

    return train_loader, test_loader


def split_unified_csv(
    unified_csv_path: str,
    output_dir: str,
    train_ratio: float = 0.8
):
    """
    Split a unified CSV file into train and test sets
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read all rows
    with open(unified_csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Shuffle and split
    np.random.shuffle(rows)
    split_idx = int(len(rows) * train_ratio)

    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]

    # Write train file
    train_path = os.path.join(output_dir, 'train', 'raw', 'unified.csv')
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(train_rows)

    # Write test file
    test_path = os.path.join(output_dir, 'test', 'raw', 'unified.csv')
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    with open(test_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(test_rows)

    print(f"Split {len(rows)} samples into:")
    print(f"  Train: {len(train_rows)} samples -> {train_path}")
    print(f"  Test: {len(test_rows)} samples -> {test_path}")
