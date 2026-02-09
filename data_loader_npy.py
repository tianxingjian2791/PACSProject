"""
Optimized data loaders supporting both CSV and NPY/NPZ formats
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import glob

class GNNThetaDatasetNPY(Dataset):
    """
    PyTorch Geometric Dataset for loading theta_gnn data from NPZ files

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


class CNNThetaDatasetNPY(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading theta_cnn data from NPZ files

    Each npz file contains:
        - pooled_matrix: (50, 50) pooled matrix
        - y: rho (convergence factor) - CORRECTED after C++ fix
        - metadata: [n, rho, h, theta]

    Returns:
        X: Concatenated tensor [theta, log_h, flattened_matrix] of shape (2502,)
           - theta: strong threshold parameter (input feature)
           - log_h: -log2(mesh_size) (input feature)
           - flattened_matrix: 2500 pooled matrix values
        y: rho (convergence factor) - the prediction target
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Find all npz files
        self.sample_files = sorted(glob.glob(os.path.join(root, "sample_*.npz")))

        if len(self.sample_files) == 0:
            raise ValueError(f"No NPZ files found in {root}")

        print(f"Found {len(self.sample_files)} NPZ samples in {root}")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        # Load NPZ file
        npz_file = self.sample_files[idx]
        data = np.load(npz_file)

        # Extract pooled matrix (50x50)
        pooled_matrix = data['pooled_matrix']  # Shape: (50, 50)

        # Extract y (rho - convergence factor, the target to predict)
        # After C++ fix, y now correctly contains rho instead of theta
        y_data = data['y']
        rho = float(y_data[0]) if y_data.ndim > 0 else float(y_data)

        # Extract metadata: [n, rho, h, theta]
        # Note: metadata[1] also contains rho, but we use data['y'] for consistency
        metadata = data['metadata']
        h = metadata[2]      # Mesh size
        theta = metadata[3]  # Strong threshold (input feature)

        # Compute log_h (input feature)
        log_h = -np.log2(h)

        # Flatten pooled matrix
        matrix_flat = pooled_matrix.flatten()  # Shape: (2500,)

        # CNN model expects input format: [theta, log_h, flattened_matrix]
        # Total shape: (2502,) = [1 + 1 + 2500]
        X = np.concatenate([[theta], [log_h], matrix_flat])

        # Target is rho (convergence factor)
        y = rho

        # Convert to tensors
        X = torch.from_numpy(X).float()
        y = torch.tensor(y, dtype=torch.float32)

        # Return as tuple (input, target) for standard PyTorch DataLoader
        if self.transform:
            X = self.transform(X)

        return X, y


class PValueDatasetNPY(Dataset):
    """
    PyTorch Geometric Dataset for loading p_value data from NPZ files

    Returns PyG Data objects for proper batching support.

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

        # Coarse nodes
        coarse_nodes = torch.from_numpy(data['coarse_nodes']).long()
        num_coarse = len(coarse_nodes)

        # Create node features (coarse/fine indicators)
        coarse_indicator = torch.zeros(n, dtype=torch.float32)
        coarse_indicator[coarse_nodes] = 1.0
        fine_indicator = 1.0 - coarse_indicator
        x = torch.stack([coarse_indicator, fine_indicator], dim=1)

        # Reconstruct baseline P matrix
        P = csr_matrix(
            (data['P_values'], data['P_col_idx'], data['P_row_ptr']),
            shape=(n, num_coarse)
        )
        P_coo = P.tocoo()

        # Create mapping from (row, coarse_node_idx) to P value
        # P[i, j] means prolongation from node i to coarse node j
        # But coarse_nodes[j] is the actual node index
        P_value_map = {}
        for i in range(len(P_coo.row)):
            row_idx = P_coo.row[i]
            coarse_idx = P_coo.col[i]  # Index into coarse_nodes array
            coarse_node = coarse_nodes[coarse_idx].item()
            P_value_map[(row_idx, coarse_node)] = P_coo.data[i]

        # For each edge in A, check if it's in the prolongation sparsity pattern
        # and create target P-values
        edge_in_baseline = []
        edge_not_in_baseline = []
        y_edge = []

        for i in range(len(A_coo.row)):
            row = A_coo.row[i]
            col = A_coo.col[i]

            # Check if this edge (row, col) is in the prolongation pattern
            # An edge is in P if it goes from a node to a coarse node
            if (row, col) in P_value_map:
                edge_in_baseline.append(1.0)
                edge_not_in_baseline.append(0.0)
                y_edge.append(P_value_map[(row, col)])
            else:
                edge_in_baseline.append(0.0)
                edge_not_in_baseline.append(1.0)
                y_edge.append(0.0)  # No prolongation value for this edge

        # Create edge features: [A_value, in_baseline, not_in_baseline]
        edge_attr_full = torch.tensor(
            np.column_stack([A_coo.data, edge_in_baseline, edge_not_in_baseline]),
            dtype=torch.float32
        )

        # Create target tensor for P-values
        y_edge_tensor = torch.tensor(y_edge, dtype=torch.float32).view(-1, 1)

        # Global features (placeholder)
        u = torch.zeros(1, 128, dtype=torch.float32)

        # Create PyG Data object
        pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr_full,
            y_edge=y_edge_tensor,  # Target P-values for each edge
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


def create_theta_cnn_data_loaders_npy(dataset_root, train_problem='train_D', test_problem='test_D',
                                       batch_size=32, num_workers=4):
    """
    Create data loaders for NPY format theta_cnn dataset

    Returns:
        train_loader, test_loader
    """
    from torch.utils.data import DataLoader

    # Paths to NPY directories
    train_path = os.path.join(dataset_root, 'train', 'raw', 'theta_cnn_npy', train_problem)
    test_path = os.path.join(dataset_root, 'test', 'raw', 'theta_cnn_npy', test_problem)

    # Check if NPY data exists
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"NPY data not found at {train_path}\n"
        )

    print(f"Loading CNN NPY datasets from:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    train_dataset = CNNThetaDatasetNPY(train_path)
    test_dataset = CNNThetaDatasetNPY(test_path)

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

