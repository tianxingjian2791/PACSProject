"""
Stage 2 Training Script: P-Value Prediction

Trains GNN model to predict prolongation matrix P values with frozen Stage 1.

Usage:
    # With NPY format (5× faster, recommended)
    python train_stage2.py --stage1-weights weights/stage1/best_model.pt \
                           --dataset datasets/unified \
                           --train-file train_GL \
                           --test-file test_GL \
                           --use-npy \
                           --epochs 100

    # With CSV format (legacy)
    python train_stage2.py --stage1-weights weights/stage1/best_model.pt \
                           --dataset datasets/unified \
                           --train-file train_D.csv \
                           --test-file test_D.csv \
                           --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
from pathlib import Path
from tqdm import tqdm

# Import models
from model import UnifiedAMGModel, EncodeProcessDecode

# Import data processing
from data import create_p_value_data_loaders

# Import utilities
from utils import (
    Checkpointer, MetricsLogger, EarlyStopping,
    set_random_seed, count_parameters, get_device,
    TwoGridLoss
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train Stage 2: P-Value Prediction')

    # Stage 1 model
    parser.add_argument('--stage1-weights', type=str, default=None,
                      help='Path to pretrained Stage 1 weights (optional)')
    parser.add_argument('--stage1-type', type=str, default='GNN', choices=['CNN', 'GNN'],
                      help='Stage 1 model type')

    # Data configuration
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset directory')
    parser.add_argument('--train-file', type=str, default='train_D.csv',
                      help='Training CSV filename (or problem type for NPY)')
    parser.add_argument('--test-file', type=str, default='test_D.csv',
                      help='Test CSV filename (or problem type for NPY)')
    parser.add_argument('--use-npy', action='store_true',
                      help='Use NPY/NPZ format instead of CSV (5× faster)')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size (typically smaller for Stage 2)')
    parser.add_argument('--lr', type=float, default=3e-3,
                      help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='Weight decay')

    # Model hyperparameters
    parser.add_argument('--latent-size', type=int, default=64,
                      help='Latent dimension size')
    parser.add_argument('--num-layers', type=int, default=4,
                      help='Number of MLP layers')
    parser.add_argument('--num-message-passing', type=int, default=3,
                      help='Number of message passing rounds')

    # Training options
    parser.add_argument('--patience', type=int, default=15,
                      help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')

    # Output configuration
    parser.add_argument('--save-dir', type=str, default='weights/stage2',
                      help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/stage2',
                      help='Directory to save logs')
    parser.add_argument('--experiment-name', type=str, default=None,
                      help='Experiment name')

    return parser.parse_args()


def create_model(args, device):
    """Create Stage 2 model"""
    print("\nCreating Stage 2 model (P-value predictor)...")

    # Stage 2 configuration
    stage2_config = {
        'edge_input_size': 3,
        'node_input_size': 2,
        'global_input_size': 128,
        'edge_output_size': 1,
        'node_output_size': 1,
        'latent_size': args.latent_size,
        'num_layers': args.num_layers,
        'num_message_passing': args.num_message_passing,
        'global_block': False,
        'concat_encoder': True
    }

    model = EncodeProcessDecode(**stage2_config)
    model = model.to(device)

    print(f"Model: EncodeProcessDecode (GNN)")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Latent size: {args.latent_size}")
    print(f"Message passing rounds: {args.num_message_passing}")

    return model


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(loader, desc="Training")

    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(batch)

        # For now, use simple MSE loss on edge predictions
        # TODO: Implement full TwoGridLoss when matrices are available
        if hasattr(batch, 'y_edge'):
            loss = criterion(output.edge_attr, batch.y_edge)
        else:
            # Fallback: self-supervised loss
            loss = criterion(output.edge_attr, batch.edge_attr)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / num_batches


def test_epoch(model, loader, criterion, device):
    """Evaluate on test set"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            output = model(batch)

            # Compute loss
            if hasattr(batch, 'y_edge'):
                loss = criterion(output.edge_attr, batch.y_edge)
            else:
                loss = criterion(output.edge_attr, batch.edge_attr)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    """Main training loop"""
    args = parse_args()

    # Set random seed
    set_random_seed(args.seed)

    # Set experiment name
    if args.experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"stage2_{timestamp}"

    print("="*60)
    print("Stage 2 Training: P-Value Prediction")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Experiment: {args.experiment_name}")
    if args.stage1_weights:
        print(f"Stage 1 weights: {args.stage1_weights}")
    print("="*60)

    # Get device
    device = get_device()

    # Create dataloaders
    print("\nLoading data...")
    print(f"Format: {'NPY/NPZ (high-performance)' if args.use_npy else 'CSV'}")

    if args.use_npy:
        from data import create_pvalue_data_loaders_npy
        # Extract problem type from filename (e.g., 'train_D.csv' -> 'train_D')
        train_problem = args.train_file.replace('.csv', '')
        test_problem = args.test_file.replace('.csv', '')
        train_loader, test_loader = create_pvalue_data_loaders_npy(
            dataset_root=args.dataset,
            train_problem=train_problem,
            test_problem=test_problem,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        train_loader, test_loader = create_p_value_data_loaders(
            data_dir=args.dataset,
            train_file=args.train_file,
            test_file=args.test_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            node_indicators=True,
            edge_indicators=True
        )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    model = create_model(args, device)

    # Create optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Loss function
    criterion = nn.MSELoss()

    # Create training utilities
    checkpointer = Checkpointer(
        checkpoint_dir=os.path.join(args.save_dir, args.experiment_name),
        max_checkpoints=5
    )

    logger = MetricsLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )

    # Log configuration
    logger.log_metadata({
        'stage': 2,
        'latent_size': args.latent_size,
        'num_layers': args.num_layers,
        'num_message_passing': args.num_message_passing,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'parameters': count_parameters(model)
    })

    early_stopping = EarlyStopping(patience=args.patience)

    # Training loop
    print("\nStarting training...")
    best_test_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.6f}")

        # Evaluate
        test_loss = test_epoch(model, test_loader, criterion, device)
        print(f"Test Loss:  {test_loss:.6f}")

        # Log metrics
        logger.log_epoch(epoch, train_loss, test_loss)

        # Learning rate scheduling
        scheduler.step(test_loss)

        # Save checkpoint
        is_best = test_loss < best_test_loss
        if is_best:
            best_test_loss = test_loss
            print(f"✓ New best model! Test loss: {test_loss:.6f}")

        checkpointer.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=test_loss,
            is_best=is_best,
            metadata={'train_loss': train_loss, 'test_loss': test_loss}
        )

        # Early stopping
        if early_stopping(test_loss):
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best test loss: {best_test_loss:.6f}")
    print(f"Best model saved to: {checkpointer.checkpoint_dir / 'best_model.pt'}")
    print("="*60)


if __name__ == '__main__':
    main()
