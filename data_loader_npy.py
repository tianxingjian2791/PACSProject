"""
Optimized data loaders supporting both CSV and NPY/NPZ formats
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
import glob

class GNNThetaDatasetNPY(Dataset):
    """
    PyTorch Geometric Dataset for loading theta_gnn data from NPZ files

    Directory structure:
        root/
            sample_00000.npz
            sample_00001.npz
            ...

    Each npz file contains:
        - edge_index: (2, num_edges)
        - edge_attr: (num_edges,)
        - y: scalar theta value
        - metadata: [n, rho, h, epsilon]
    """

    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        # Find all npz files
        self.sample_files = sorted(glob.glob(os.path.join(root, "sample_*.npz")))

        if len(self.sample_files) == 0:
            raise ValueError(f"No NPZ files found in {root}")

        print(f"Found {len(self.sample_files)} NPZ samples in {root}")
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [os.path.basename(f) for f in self.sample_files]

    @property
    def processed_file_names(self):
        return []  # No processing needed, load directly

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.sample_files)

    def get(self, idx):
        # Load NPZ file
        npz_file = self.sample_files[idx]
        data = np.load(npz_file)

        # Extract arrays
        edge_index = torch.from_numpy(data['edge_index']).long()
        edge_attr = torch.from_numpy(data['edge_attr']).float().view(-1, 1)  # Shape: (num_edges, 1)
        theta = torch.from_numpy(data['theta']).float()
        y = torch.from_numpy(data['y']).float()  # rho value
        metadata = data['metadata']

        # Get number of nodes from metadata
        num_nodes = int(metadata[0])
        h = metadata[2]

        # Calculate node features (degree)
        degrees = torch.zeros(num_nodes, dtype=torch.float)
        for i in range(num_nodes):
            degrees[i] = (edge_index[0] == i).sum().item()
        x = degrees.view(-1, 1)

        # Create PyG Data object matching the original GNNThetaDataset
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            theta=theta,
            log_h=-torch.log2(torch.tensor([h], dtype=torch.float)),
            num_nodes=num_nodes
        )


class PValueDatasetNPY(Dataset):
    """
    PyTorch Geometric Dataset for loading p_value data from NPZ files

    Returns PyG Data objects for proper batching support.

    Directory structure:
        root/
            sample_00000.npz
            sample_00001.npz
            ...

    Each npz file contains:
        - A_values, A_row_ptr, A_col_idx: CSR matrix A
        - coarse_nodes: coarse node indices
        - P_values, P_row_ptr, P_col_idx: CSR matrix P
        - S_values, S_row_ptr, S_col_idx: CSR matrix S
        - metadata: [n, theta, rho, h]
    """

    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        self.sample_files = sorted(glob.glob(os.path.join(root, "sample_*.npz")))

        if len(self.sample_files) == 0:
            raise ValueError(f"No NPZ files found in {root}")

        print(f"Found {len(self.sample_files)} NPZ samples in {root}")
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [os.path.basename(f) for f in self.sample_files]

    @property
    def processed_file_names(self):
        return []  # No processing needed

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.sample_files)

    def get(self, idx):
        """Load individual sample and convert to PyG Data object"""
        from torch_geometric.data import Data
        from scipy.sparse import csr_matrix

        # Load NPZ file
        npz_file = self.sample_files[idx]
        data = np.load(npz_file)

        # Parse metadata
        metadata = data['metadata']
        n = int(metadata[0])
        theta = float(metadata[1])
        rho = float(metadata[2])
        h = float(metadata[3])

        # Reconstruct CSR matrix A
        A = csr_matrix(
            (data['A_values'], data['A_col_idx'], data['A_row_ptr']),
            shape=(n, n)
        )

        # Convert A to COO for edge_index
        A_coo = A.tocoo()
        edge_index = torch.tensor(
            np.vstack([A_coo.row, A_coo.col]),
            dtype=torch.long
        )
        edge_attr = torch.tensor(A_coo.data, dtype=torch.float32).view(-1, 1)

        # Coarse nodes
        coarse_nodes = torch.from_numpy(data['coarse_nodes']).long()

        # Create node features (coarse/fine indicators)
        coarse_indicator = torch.zeros(n, dtype=torch.float32)
        coarse_indicator[coarse_nodes] = 1.0
        fine_indicator = 1.0 - coarse_indicator
        x = torch.stack([coarse_indicator, fine_indicator], dim=1)

        # Reconstruct baseline P matrix for edge indicators
        P = csr_matrix(
            (data['P_values'], data['P_col_idx'], data['P_row_ptr']),
            shape=(n, len(coarse_nodes))
        )
        P_coo = P.tocoo()
        baseline_edges = set(zip(P_coo.row, P_coo.col))

        # Add edge indicators (in baseline / not in baseline)
        edge_in_baseline = []
        edge_not_in_baseline = []
        for i in range(len(A_coo.row)):
            # Note: baseline P edges are different from A edges
            # For now, just use zeros as placeholders
            edge_in_baseline.append(0.0)
            edge_not_in_baseline.append(1.0)

        edge_attr_full = torch.tensor(
            np.column_stack([A_coo.data, edge_in_baseline, edge_not_in_baseline]),
            dtype=torch.float32
        )

        # Global features (placeholder)
        u = torch.zeros(1, 128, dtype=torch.float32)

        # Create PyG Data object
        pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr_full,
            u=u,
            num_nodes=n
        )

        # Store additional info (not batched)
        pyg_data.coarse_nodes = coarse_nodes
        pyg_data.theta = torch.tensor([theta], dtype=torch.float)
        pyg_data.rho = torch.tensor([rho], dtype=torch.float)
        pyg_data.log_h = torch.tensor([-np.log2(h)], dtype=torch.float)

        # Store matrices as scipy sparse (accessed separately during training)
        pyg_data.A_sparse = A
        pyg_data.baseline_P_sparse = P

        # Reconstruct S matrix
        S = csr_matrix(
            (data['S_values'], data['S_col_idx'], data['S_row_ptr']),
            shape=(n, n)
        )
        pyg_data.S_sparse = S

        return pyg_data


def create_theta_data_loaders_npy(dataset_root, train_problem='train_D', test_problem='test_D',
                                  batch_size=32, num_workers=4):
    """
    Create data loaders for NPY format theta_gnn dataset

    Args:
        dataset_root: Path to dataset root (e.g., 'datasets/unified')
        train_problem: Training problem directory name (e.g., 'train_D', 'train_GL')
        test_problem: Test problem directory name (e.g., 'test_D', 'test_GL')
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        train_loader, test_loader
    """
    from torch_geometric.loader import DataLoader

    # Paths to NPY directories
    train_path = os.path.join(dataset_root, 'train', 'raw', 'theta_gnn_npy', train_problem)
    test_path = os.path.join(dataset_root, 'test', 'raw', 'theta_gnn_npy', test_problem)

    # Check if NPY data exists
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"NPY data not found at {train_path}\n"
            "Please generate NPY data using: ./build/generate_amg_data -p <type> -s train -f theta-gnn -c <scale>"
        )

    print(f"Loading NPY datasets from:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    train_dataset = GNNThetaDatasetNPY(train_path)
    test_dataset = GNNThetaDatasetNPY(test_path)

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


def create_pvalue_data_loaders_npy(dataset_root, train_problem='train_D', test_problem='test_D',
                                   batch_size=32, num_workers=4):
    """
    Create data loaders for NPY format p_value dataset

    Args:
        dataset_root: Path to dataset root (e.g., 'datasets/unified')
        train_problem: Training problem directory name (e.g., 'train_D', 'train_GL')
        test_problem: Test problem directory name (e.g., 'test_D', 'test_GL')
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        train_loader, test_loader (PyG DataLoaders)
    """
    from torch_geometric.loader import DataLoader

    # Paths to NPY directories
    train_path = os.path.join(dataset_root, 'train', 'raw', 'p_value_npy', train_problem)
    test_path = os.path.join(dataset_root, 'test', 'raw', 'p_value_npy', test_problem)

    # Check if NPY data exists
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"NPY data not found at {train_path}\n"
            "Please generate NPY data using: ./build/generate_amg_data -p <type> -s train -f p-value -c <scale>"
        )

    print(f"Loading NPY datasets from:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    train_dataset = PValueDatasetNPY(train_path)
    test_dataset = PValueDatasetNPY(test_path)

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


if __name__ == "__main__":
    # Test loading
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader_npy.py <dataset_root>")
        print("Example: python data_loader_npy.py datasets/unified")
        sys.exit(1)

    dataset_root = sys.argv[1]

    print("Testing NPY data loaders...")
    print()

    try:
        print("=" * 50)
        print("Testing theta_gnn NPY loader...")
        print("=" * 50)
        train_loader, test_loader = create_theta_data_loaders_npy(dataset_root, batch_size=4)
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Test loading first batch
        batch = next(iter(train_loader))
        print(f"Batch shape: {batch}")
        print(f"  edge_index: {batch.edge_index.shape}")
        print(f"  edge_attr: {batch.edge_attr.shape}")
        print(f"  y: {batch.y.shape}")
        print("✅ theta_gnn NPY loader works!")

    except FileNotFoundError as e:
        print(f"⚠️  theta_gnn NPY data not found: {e}")

    print()

    try:
        print("=" * 50)
        print("Testing p_value NPY loader...")
        print("=" * 50)
        train_loader, test_loader = create_pvalue_data_loaders_npy(dataset_root, batch_size=4)
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Test loading first batch
        batch = next(iter(train_loader))
        print(f"Sample keys: {list(batch[0].keys())}")
        print("✅ p_value NPY loader works!")

    except FileNotFoundError as e:
        print(f"⚠️  p_value NPY data not found: {e}")
