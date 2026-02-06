"""
Benchmark NPY vs CSV data loading speed
"""

import time
import torch
from torch_geometric.loader import DataLoader

# Import both data loaders
from data_loader_npy import GNNThetaDatasetNPY
from data import GNNThetaDataset

def benchmark_loading(dataset, name, num_iterations=5):
    """Benchmark data loading speed"""
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    times = []
    for i in range(num_iterations):
        start = time.time()
        for batch in loader:
            # Just load, don't process
            pass
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    rate = len(dataset) / avg_time

    print(f"{name}:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Avg time: {avg_time*1000:.2f}ms")
    print(f"  Rate: {rate:.1f} samples/s")
    print()

    return avg_time, rate

if __name__ == "__main__":
    print("="*60)
    print("NPY vs CSV Loading Speed Benchmark")
    print("="*60)
    print()

    # Benchmark NPY format
    print("Loading NPY dataset...")
    npy_dataset = GNNThetaDatasetNPY('datasets/unified/train/raw/theta_gnn/theta_gnn_npy/train_D_small')
    npy_time, npy_rate = benchmark_loading(npy_dataset, "NPY Format")

    # Benchmark CSV format
    print("Loading CSV dataset...")
    csv_dataset = GNNThetaDataset(
        root='datasets/unified/train',
        csv_file='theta_gnn/train_D_small.csv'
    )
    csv_time, csv_rate = benchmark_loading(csv_dataset, "CSV Format")

    # Compare
    print("="*60)
    print("Comparison:")
    print(f"  NPY is {csv_time/npy_time:.1f}x faster than CSV")
    print(f"  NPY rate: {npy_rate:.1f} samples/s")
    print(f"  CSV rate: {csv_rate:.1f} samples/s")
    print("="*60)
