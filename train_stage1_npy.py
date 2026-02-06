"""
Training script with NPY format support

Usage:
    # Train with NPY format
    python train_stage1_npy.py --dataset datasets/unified --format npy --epochs 3

    # Train with CSV format (original)
    python train_stage1_npy.py --dataset datasets/unified --format csv --epochs 3
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

# Import original data loaders
from train_stage1 import GNNModel, gnn_train, gnn_test, create_theta_data_loaders

# Import NPY data loaders
from data_loader_npy import create_theta_data_loaders_npy

def main():
    parser = argparse.ArgumentParser(description='Train GNN for theta prediction')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--format', type=str, default='npy', choices=['csv', 'npy'],
                       help='Data format: csv or npy')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden-channels', type=int, default=64, help='Hidden channels')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark loading speed')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Data format: {args.format}')
    print()

    # Create data loaders based on format
    print("Loading datasets...")
    load_start = time.time()

    if args.format == 'npy':
        # For testing, we use the small NPY datasets
        # In production, use full datasets
        import os
        from data_loader_npy import GNNThetaDatasetNPY

        # Try small dataset first
        train_path = os.path.join(args.dataset, 'train', 'raw', 'theta_gnn', 'theta_gnn_npy', 'train_D_small')
        test_path = os.path.join(args.dataset, 'test', 'raw', 'theta_gnn', 'theta_gnn_npy', 'test_D_small')

        if not os.path.exists(train_path):
            # Fall back to full dataset
            train_path = os.path.join(args.dataset, 'train', 'raw', 'theta_gnn_npy', 'train_D')
            test_path = os.path.join(args.dataset, 'test', 'raw', 'theta_gnn_npy', 'test_D')

        print(f"Loading NPY from: {train_path}")
        train_dataset = GNNThetaDatasetNPY(train_path)
        test_dataset = GNNThetaDatasetNPY(test_path)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    else:  # CSV format
        # Use small CSV for testing
        import os
        csv_file = os.path.join(args.dataset, 'train', 'raw', 'theta_gnn', 'train_D_small.csv')
        if not os.path.exists(csv_file):
            # Fall back to regular dataset
            train_loader, test_loader = create_theta_data_loaders(
                args.dataset, batch_size=args.batch_size
            )
        else:
            # Use small dataset
            from data_loader import GNNThetaDataset
            train_dataset = GNNThetaDataset(
                root=os.path.join(args.dataset, 'train'),
                csv_file='theta_gnn/train_D_small.csv'
            )
            test_dataset = GNNThetaDataset(
                root=os.path.join(args.dataset, 'test'),
                csv_file='theta_gnn/test_D_small.csv'
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    load_time = time.time() - load_start
    print(f'Dataset loading time: {load_time:.2f}s')
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    print()

    # Benchmark loading if requested
    if args.benchmark:
        print("=" * 60)
        print("Benchmarking data loading speed...")
        print("=" * 60)

        # Benchmark: iterate through all batches
        start = time.time()
        total_samples = 0
        for batch in train_loader:
            total_samples += batch.num_graphs
        elapsed = time.time() - start

        print(f'Format: {args.format}')
        print(f'Total samples loaded: {total_samples}')
        print(f'Loading time: {elapsed:.3f}s')
        print(f'Loading rate: {total_samples/elapsed:.1f} samples/s')
        print("=" * 60)
        print()

    # Create model
    model = GNNModel(hidden_channels=args.hidden_channels, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    print()

    # Training loop
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = gnn_train(model, train_loader, optimizer, device)
        test_loss = gnn_test(model, test_loader, device)

        epoch_time = time.time() - epoch_start

        print(f'Epoch {epoch:02d}/{args.epochs:02d} | '
              f'Train Loss: {train_loss:.6f} | '
              f'Test Loss: {test_loss:.6f} | '
              f'Time: {epoch_time:.2f}s')

    print()
    print("✅ Training complete!")

if __name__ == '__main__':
    main()
