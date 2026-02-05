"""
Enhanced GNN data processing supporting both theta prediction and P-value prediction

This module extends the existing GAT data processing to support:
    1. Theta prediction (Stage 1): Like existing, load sparse matrix + theta/rho
    2. P-value prediction (Stage 2): Load sparse matrix + C/F splitting + baseline P
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from scipy.sparse import csr_matrix, coo_matrix
import csv
import os
import glob
from typing import List, Tuple, Optional, Literal


class GNNThetaDataset(InMemoryDataset):
    """
    Dataset for GNN theta prediction (Stage 1)

    CSV Format:
        num_rows, num_cols, theta, rho, h, nnz, [values], [row_ptrs], [col_indices]
    """

    def __init__(self, root, csv_file, transform=None, pre_transform=None):
        """
        Initialize the dataset

        Parameters:
            root: the root of dataset
            csv_file: CSV file name
            transform: optional
            pre_transform: optional
        """
        self.csv_file = csv_file
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [self.csv_file]

    @property
    def processed_file_names(self):
        return [self.csv_file[:-4] + '_gnn_theta.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        raw_path = os.path.join(self.raw_dir, self.csv_file)

        print(f"Processing {raw_path} for GNN theta prediction...")

        with open(raw_path, newline='') as f:
            reader = csv.reader(f)

            row_cnt = 0
            for row in reader:
                row_cnt += 1
                if row_cnt % 100 == 0:
                    print(f"Processed {row_cnt} samples")

                # Parse CSV row
                num_rows = int(row[0])
                num_cols = int(row[1])
                theta_v = float(row[2])
                rho = float(row[3])
                h_v = float(row[4])
                nnz = int(row[5])

                # Parse sparse matrix in CSR format
                val_start = 6
                val_end = val_start + nnz
                values = list(map(float, row[val_start:val_end]))

                ptr_len = num_rows + 1
                ptr_start = val_end
                ptr_end = ptr_start + ptr_len
                row_ptrs = list(map(int, row[ptr_start:ptr_end]))

                col_start = ptr_end
                col_end = col_start + nnz
                col_indices = list(map(int, row[col_start:col_end]))

                # Convert to tensors
                row_ptrs = torch.tensor(row_ptrs, dtype=torch.long)
                col_indices = torch.tensor(col_indices, dtype=torch.long)
                values = torch.tensor(values, dtype=torch.float)

                # Construct edge_index
                edge_index = []
                for i in range(num_rows):
                    for j in range(row_ptrs[i], row_ptrs[i+1]):
                        edge_index.append([i, col_indices[j]])
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

                # Node feature: degree
                degrees = torch.zeros(num_rows, dtype=torch.float)
                for i in range(num_rows):
                    degrees[i] = row_ptrs[i+1] - row_ptrs[i]
                x = degrees.view(-1, 1)

                # Edge feature: matrix values
                edge_attr = values.view(-1, 1)

                # Scalar features and labels
                y = torch.tensor([rho], dtype=torch.float)
                theta = torch.tensor([theta_v], dtype=torch.float)
                h = torch.tensor([h_v], dtype=torch.float)
                log_h = -torch.log2(h)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    theta=theta,
                    log_h=log_h
                )
                data_list.append(data)

        # Optional filtering and transformation
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GNNPValueDataset(Dataset):
    """
    Dataset for GNN P-value prediction (Stage 2)

    This dataset loads:
        - Sparse matrix A
        - C/F splitting (coarse nodes)
        - Baseline prolongation matrix P
        - Smoother matrix S (for loss computation)

    CSV Format (extended):
        num_rows, num_cols, theta, rho, h, nnz,
        [values], [row_ptrs], [col_indices],
        num_coarse, [coarse_node_indices],
        nnz_P, [P_values], [P_row_ptrs], [P_col_indices],
        nnz_S, [S_values], [S_row_ptrs], [S_col_indices]
    """

    def __init__(self, root, csv_file,
                 node_indicators=True, edge_indicators=True,
                 transform=None, pre_transform=None):
        """
        Initialize P-value dataset

        Parameters:
            root: dataset root directory
            csv_file: CSV filename
            node_indicators: include coarse/fine node indicators
            edge_indicators: include baseline/non-baseline edge indicators
            transform: optional transform
            pre_transform: optional pre-transform
        """
        self.csv_file = csv_file
        self.node_indicators = node_indicators
        self.edge_indicators = edge_indicators
        super().__init__(root, transform, pre_transform)

        self.processed_subdir = os.path.join(self.processed_dir, self.csv_file[:-4])

        # Load processed files
        self.processed_files = sorted(glob.glob(os.path.join(self.processed_subdir, 'data_*.pt')))
        if not self.processed_files:
            self.process()
            self.processed_files = sorted(glob.glob(os.path.join(self.processed_subdir, 'data_*.pt')))

    @property
    def raw_file_names(self):
        return [self.csv_file]

    @property
    def processed_file_names(self):
        processed_subdir = os.path.join(self.processed_dir, self.csv_file[:-4])
        os.makedirs(processed_subdir, exist_ok=True)
        return glob.glob(os.path.join(processed_subdir, 'data_*.pt'))

    def download(self):
        pass

    def process(self):
        """Process CSV and save individual samples as .pt files"""
        if not self.processed_file_names:
            raw_path = os.path.join(self.raw_dir, self.csv_file)
            processed_subdir = os.path.join(self.processed_dir, self.csv_file[:-4])
            os.makedirs(processed_subdir, exist_ok=True)

            print(f"Processing {raw_path} for GNN P-value prediction...")

            with open(raw_path, newline='') as f:
                reader = csv.reader(f)

                for idx, row in enumerate(reader):
                    if idx % 50 == 0:
                        print(f"Processed {idx} samples")

                    # Parse basic info
                    num_rows = int(row[0])
                    num_cols = int(row[1])
                    theta_v = float(row[2])
                    rho = float(row[3])
                    h_v = float(row[4])
                    nnz = int(row[5])

                    # Parse matrix A (CSR format)
                    offset = 6
                    values_A = np.array(list(map(float, row[offset:offset+nnz])))
                    offset += nnz

                    row_ptrs_A = np.array(list(map(int, row[offset:offset+num_rows+1])))
                    offset += num_rows + 1

                    col_indices_A = np.array(list(map(int, row[offset:offset+nnz])))
                    offset += nnz

                    # Create sparse matrix A
                    A = csr_matrix((values_A, col_indices_A, row_ptrs_A),
                                   shape=(num_rows, num_cols))

                    # Parse coarse nodes
                    num_coarse = int(row[offset])
                    offset += 1
                    coarse_nodes = np.array(list(map(int, row[offset:offset+num_coarse])))
                    offset += num_coarse

                    # Parse baseline prolongation matrix P
                    nnz_P = int(row[offset])
                    offset += 1
                    values_P = np.array(list(map(float, row[offset:offset+nnz_P])))
                    offset += nnz_P

                    row_ptrs_P = np.array(list(map(int, row[offset:offset+num_rows+1])))
                    offset += num_rows + 1

                    col_indices_P = np.array(list(map(int, row[offset:offset+nnz_P])))
                    offset += nnz_P

                    baseline_P = csr_matrix((values_P, col_indices_P, row_ptrs_P),
                                           shape=(num_rows, num_coarse))

                    # Parse smoother matrix S (for loss computation)
                    nnz_S = int(row[offset])
                    offset += 1
                    values_S = np.array(list(map(float, row[offset:offset+nnz_S])))
                    offset += nnz_S

                    row_ptrs_S = np.array(list(map(int, row[offset:offset+num_rows+1])))
                    offset += num_rows + 1

                    col_indices_S = np.array(list(map(int, row[offset:offset+nnz_S])))

                    S = csr_matrix((values_S, col_indices_S, row_ptrs_S),
                                  shape=(num_rows, num_rows))

                    # Convert to PyG Data
                    data = self._create_pyg_data(
                        A, coarse_nodes, baseline_P, S,
                        theta_v, rho, h_v
                    )

                    # Apply transforms
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    # Save individual sample
                    torch.save(data, os.path.join(processed_subdir, f'data_{idx}.pt'))

    def _create_pyg_data(self, A, coarse_nodes, baseline_P, S, theta, rho, h):
        """Create PyTorch Geometric Data from sparse matrices"""
        # Convert A to COO for edge_index
        A_coo = A.tocoo()
        edge_index = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long)

        # Create edge features
        if self.edge_indicators:
            baseline_P_coo = baseline_P.tocoo()
            baseline_edges = set(zip(baseline_P_coo.row, baseline_P_coo.col))

            edge_in_baseline = []
            edge_not_in_baseline = []

            for i in range(len(A_coo.row)):
                # Check if this edge is in baseline prolongation
                # Note: baseline_P edges go from fine nodes to coarse nodes
                is_baseline = (A_coo.row[i], A_coo.col[i]) in baseline_edges
                edge_in_baseline.append(1.0 if is_baseline else 0.0)
                edge_not_in_baseline.append(0.0 if is_baseline else 1.0)

            edge_attr = torch.tensor(
                np.column_stack([A_coo.data, edge_in_baseline, edge_not_in_baseline]),
                dtype=torch.float32
            )
        else:
            edge_attr = torch.tensor(A_coo.data.reshape(-1, 1), dtype=torch.float32)

        # Create node features
        if self.node_indicators:
            num_nodes = A.shape[0]
            coarse_indicator = np.zeros(num_nodes)
            coarse_indicator[coarse_nodes] = 1.0
            fine_indicator = 1.0 - coarse_indicator

            x = torch.tensor(
                np.column_stack([coarse_indicator, fine_indicator]),
                dtype=torch.float32
            )
        else:
            # Use degree as node feature
            degrees = np.array(A.sum(axis=1)).flatten()
            x = torch.tensor(degrees.reshape(-1, 1), dtype=torch.float32)

        # Global features (placeholder)
        u = torch.zeros(1, 128, dtype=torch.float32)

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=u,
            num_nodes=A.shape[0]
        )

        # Store additional information (not for batching, accessed separately)
        data.coarse_nodes = torch.tensor(coarse_nodes, dtype=torch.long)
        data.theta = torch.tensor([theta], dtype=torch.float)
        data.rho = torch.tensor([rho], dtype=torch.float)
        data.log_h = torch.tensor([-np.log2(h)], dtype=torch.float)

        # Store matrices as numpy arrays (will be accessed separately during training)
        # We don't include them in the batching process
        data.A_sparse = A  # Keep as scipy sparse
        data.baseline_P_sparse = baseline_P
        data.S_sparse = S

        return data

    def len(self):
        return len(self.processed_files)

    def get(self, idx):
        """Load individual sample"""
        data = torch.load(self.processed_files[idx], weights_only=False)
        return data


def create_theta_data_loaders(data_dir, train_file, test_file,
                              batch_size=32, num_workers=4):
    """
    Create data loaders for theta prediction (Stage 1)

    Parameters:
        data_dir: dataset directory
        train_file: training CSV filename
        test_file: test CSV filename
        batch_size: batch size
        num_workers: number of dataloader workers

    Returns:
        train_loader, test_loader
    """
    train_dataset = GNNThetaDataset(
        root=os.path.join(data_dir, 'train'),
        csv_file=train_file
    )

    test_dataset = GNNThetaDataset(
        root=os.path.join(data_dir, 'test'),
        csv_file=test_file
    )

    print(f"Theta training samples: {len(train_dataset)}")
    print(f"Theta test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def create_p_value_data_loaders(data_dir, train_file, test_file,
                                batch_size=16, num_workers=4,
                                node_indicators=True, edge_indicators=True):
    """
    Create data loaders for P-value prediction (Stage 2)

    Parameters:
        data_dir: dataset directory
        train_file: training CSV filename
        test_file: test CSV filename
        batch_size: batch size (typically smaller for Stage 2)
        num_workers: number of dataloader workers
        node_indicators: include coarse/fine indicators
        edge_indicators: include baseline/non-baseline indicators

    Returns:
        train_loader, test_loader
    """
    train_dataset = GNNPValueDataset(
        root=os.path.join(data_dir, 'train'),
        csv_file=train_file,
        node_indicators=node_indicators,
        edge_indicators=edge_indicators
    )

    test_dataset = GNNPValueDataset(
        root=os.path.join(data_dir, 'test'),
        csv_file=test_file,
        node_indicators=node_indicators,
        edge_indicators=edge_indicators
    )

    print(f"P-value training samples: {len(train_dataset)}")
    print(f"P-value test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader
