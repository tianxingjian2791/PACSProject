"""
Unified Training Script: Full Two-Stage Pipeline

Trains the complete unified model:
  1. Train Stage 1 (theta prediction)
  2. Freeze Stage 1
  3. Train Stage 2 (P-value prediction)

Usage:
    python train_unified.py --stage1-model GNN --dataset datasets/unified --epochs-stage1 50 --epochs-stage2 100
"""

import torch
import argparse
import os
from pathlib import Path

# Import models
from model import UnifiedAMGModel, create_unified_model

# Import utilities
from utils import set_random_seed, get_device


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train Unified Model: Two-Stage Pipeline')

    # Model configuration
    parser.add_argument('--stage1-model', type=str, default='GNN', choices=['CNN', 'GNN'],
                      help='Stage 1 model type')

    # Data configuration
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset directory')
    parser.add_argument('--train-file', type=str, default='train_D.csv',
                      help='Training CSV filename')
    parser.add_argument('--test-file', type=str, default='test_D.csv',
                      help='Test CSV filename')

    # Stage 1 training
    parser.add_argument('--epochs-stage1', type=int, default=50,
                      help='Epochs for Stage 1')
    parser.add_argument('--batch-size-stage1', type=int, default=32,
                      help='Batch size for Stage 1')
    parser.add_argument('--lr-stage1', type=float, default=0.001,
                      help='Learning rate for Stage 1')

    # Stage 2 training
    parser.add_argument('--epochs-stage2', type=int, default=100,
                      help='Epochs for Stage 2')
    parser.add_argument('--batch-size-stage2', type=int, default=16,
                      help='Batch size for Stage 2')
    parser.add_argument('--lr-stage2', type=float, default=3e-3,
                      help='Learning rate for Stage 2')

    # Model hyperparameters
    parser.add_argument('--hidden-channels', type=int, default=64,
                      help='Hidden channels for Stage 1')
    parser.add_argument('--latent-size', type=int, default=64,
                      help='Latent size for Stage 2')

    # Training options
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Dataloader workers')

    # Output configuration
    parser.add_argument('--save-dir', type=str, default='weights/unified',
                      help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/unified',
                      help='Directory to save logs')
    parser.add_argument('--experiment-name', type=str, default=None,
                      help='Experiment name')

    # Pipeline options
    parser.add_argument('--skip-stage1', action='store_true',
                      help='Skip Stage 1 training (use existing weights)')
    parser.add_argument('--stage1-weights', type=str, default=None,
                      help='Path to pretrained Stage 1 weights')

    return parser.parse_args()


def main():
    """Main unified training pipeline"""
    args = parse_args()

    # Set random seed
    set_random_seed(args.seed)

    # Set experiment name
    if args.experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"unified_{args.stage1_model}_{timestamp}"

    print("\n" + "="*70)
    print(" "*20 + "UNIFIED AMG TRAINING PIPELINE")
    print("="*70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Stage 1 Model: {args.stage1_model}")
    print("="*70)

    # Get device
    device = get_device()

    # =================================================================
    # PHASE 1: Train Stage 1 (Theta Prediction)
    # =================================================================
    if not args.skip_stage1:
        print("\n" + "="*70)
        print(" "*25 + "PHASE 1: STAGE 1 TRAINING")
        print(" "*22 + "(Theta Prediction)")
        print("="*70)

        # Create Stage 1 training arguments
        stage1_args = argparse.Namespace(
            model=args.stage1_model,
            dataset=args.dataset,
            train_file=args.train_file,
            test_file=args.test_file,
            epochs=args.epochs_stage1,
            batch_size=args.batch_size_stage1,
            lr=args.lr_stage1,
            weight_decay=1e-5,
            hidden_channels=args.hidden_channels,
            dropout=0.25,
            patience=10,
            num_workers=args.num_workers,
            seed=args.seed,
            save_dir=os.path.join(args.save_dir, 'stage1'),
            log_dir=os.path.join(args.log_dir, 'stage1'),
            experiment_name=args.experiment_name + '_stage1'
        )

        # Train Stage 1
        print("\nTraining Stage 1...")
        train_stage1_with_args(stage1_args)

        # Set Stage 1 weights path
        args.stage1_weights = os.path.join(
            stage1_args.save_dir,
            stage1_args.experiment_name,
            'best_model.pt'
        )

        print(f"\n✓ Stage 1 training complete!")
        print(f"  Weights saved to: {args.stage1_weights}")

    else:
        print("\n⊳ Skipping Stage 1 training (using existing weights)")
        if not args.stage1_weights:
            raise ValueError("Must provide --stage1-weights when using --skip-stage1")

    # =================================================================
    # PHASE 2: Train Stage 2 (P-Value Prediction)
    # =================================================================
    print("\n" + "="*70)
    print(" "*25 + "PHASE 2: STAGE 2 TRAINING")
    print(" "*20 + "(P-Value Prediction)")
    print("="*70)

    # Create Stage 2 training arguments
    stage2_args = argparse.Namespace(
        stage1_weights=args.stage1_weights,
        stage1_type=args.stage1_model,
        dataset=args.dataset,
        train_file=args.train_file,
        test_file=args.test_file,
        epochs=args.epochs_stage2,
        batch_size=args.batch_size_stage2,
        lr=args.lr_stage2,
        weight_decay=1e-5,
        latent_size=args.latent_size,
        num_layers=4,
        num_message_passing=3,
        patience=15,
        num_workers=args.num_workers,
        seed=args.seed,
        save_dir=os.path.join(args.save_dir, 'stage2'),
        log_dir=os.path.join(args.log_dir, 'stage2'),
        experiment_name=args.experiment_name + '_stage2'
    )

    # Train Stage 2
    print("\nTraining Stage 2...")
    train_stage2_with_args(stage2_args)

    print(f"\n✓ Stage 2 training complete!")

    # =================================================================
    # PHASE 3: Save Unified Model
    # =================================================================
    print("\n" + "="*70)
    print(" "*22 + "PHASE 3: SAVING UNIFIED MODEL")
    print("="*70)

    # Create unified model with both stages
    unified_model = create_unified_model(
        stage1_type=args.stage1_model,
        stage1_weights=args.stage1_weights,
        stage2_weights=os.path.join(
            stage2_args.save_dir,
            stage2_args.experiment_name,
            'best_model.pt'
        ),
        device=device
    )

    # Save complete unified model
    unified_save_path = os.path.join(args.save_dir, args.experiment_name, 'unified_model.pt')
    os.makedirs(os.path.dirname(unified_save_path), exist_ok=True)
    torch.save(unified_model.state_dict(), unified_save_path)

    print(f"\n✓ Unified model saved to: {unified_save_path}")

    # =================================================================
    # COMPLETE
    # =================================================================
    print("\n" + "="*70)
    print(" "*25 + "TRAINING COMPLETE!")
    print("="*70)
    print("\nModel Components:")
    print(f"  Stage 1: {args.stage1_weights}")
    print(f"  Stage 2: {os.path.join(stage2_args.save_dir, stage2_args.experiment_name, 'best_model.pt')}")
    print(f"  Unified: {unified_save_path}")
    print("\nNext steps:")
    print(f"  1. Evaluate: python evaluate.py --model {unified_save_path}")
    print(f"  2. Visualize: Check logs in {args.log_dir}")
    print("="*70 + "\n")


def train_stage1_with_args(args):
    """Train Stage 1 with given arguments"""
    from data import create_theta_data_loaders
    from model import CNNModel, GNNModel, cnn_train, cnn_test, gnn_train, gnn_test
    from utils import Checkpointer, MetricsLogger, EarlyStopping, count_parameters, get_device
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import os

    device = get_device()

    # Create dataloaders
    print(f"\nLoading data from {args.dataset}...")
    if args.model == 'CNN':
        from data.cnn_data_processing import CSVDataset
        from torch.utils.data import DataLoader

        train_path = os.path.join(args.dataset, 'train', 'raw', 'theta_cnn', args.train_file)
        test_path = os.path.join(args.dataset, 'test', 'raw', 'theta_cnn', args.test_file)

        train_dataset = CSVDataset(train_path)
        test_dataset = CSVDataset(test_path)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:  # GNN
        train_loader, test_loader = create_theta_data_loaders(
            data_dir=args.dataset,
            train_file=args.train_file,
            test_file=args.test_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    if args.model == 'CNN':
        model = CNNModel(in_channels=1, matrix_size=50, hidden_channels=args.hidden_channels, dropout=args.dropout)
    else:  # GNN
        model = GNNModel(hidden_channels=args.hidden_channels, dropout=args.dropout)

    model = model.to(device)
    print(f"Model: {args.model}, Parameters: {count_parameters(model):,}")

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Create training utilities
    checkpointer = Checkpointer(checkpoint_dir=os.path.join(args.save_dir, args.experiment_name), max_checkpoints=5)
    logger = MetricsLogger(log_dir=args.log_dir, experiment_name=args.experiment_name)
    logger.log_metadata({'model': args.model, 'hidden_channels': args.hidden_channels, 'parameters': count_parameters(model)})
    early_stopping = EarlyStopping(patience=args.patience)

    # Training loop
    best_test_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        if args.model == 'CNN':
            train_loss = cnn_train(model, train_loader, optimizer, device)
        else:
            train_loss = gnn_train(model, train_loader, optimizer, device)

        # Evaluate
        if args.model == 'CNN':
            test_loss = cnn_test(model, test_loader, device)
        else:
            test_loss = gnn_test(model, test_loader, device)

        print(f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

        # Log and save
        logger.log_epoch(epoch, train_loss, test_loss)
        scheduler.step(test_loss)

        is_best = test_loss < best_test_loss
        if is_best:
            best_test_loss = test_loss
            print(f"✓ New best model!")

        checkpointer.save(model=model, optimizer=optimizer, epoch=epoch, loss=test_loss, is_best=is_best,
                         metadata={'train_loss': train_loss, 'test_loss': test_loss})

        if early_stopping(test_loss):
            print(f"Early stopping triggered")
            break

    print(f"\nStage 1 training complete. Best test loss: {best_test_loss:.6f}")


def train_stage2_with_args(args):
    """Train Stage 2 with given arguments"""
    from data import create_p_value_data_loaders
    from model import EncodeProcessDecode
    from utils import Checkpointer, MetricsLogger, EarlyStopping, count_parameters, get_device
    import torch.optim as optim
    import torch.nn as nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import os

    device = get_device()

    # Load data
    print(f"\nLoading P-value data from {args.dataset}...")
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

    model = EncodeProcessDecode(**stage2_config).to(device)
    print(f"Model: EncodeProcessDecode, Parameters: {count_parameters(model):,}")

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    # Create training utilities
    checkpointer = Checkpointer(checkpoint_dir=os.path.join(args.save_dir, args.experiment_name), max_checkpoints=5)
    logger = MetricsLogger(log_dir=args.log_dir, experiment_name=args.experiment_name)
    logger.log_metadata({'latent_size': args.latent_size, 'parameters': count_parameters(model)})
    early_stopping = EarlyStopping(patience=args.patience)

    # Training loop
    best_test_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)

            if hasattr(batch, 'y_edge'):
                loss = criterion(output.edge_attr, batch.y_edge)
            else:
                loss = criterion(output.edge_attr, batch.edge_attr)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)

                if hasattr(batch, 'y_edge'):
                    loss = criterion(output.edge_attr, batch.y_edge)
                else:
                    loss = criterion(output.edge_attr, batch.edge_attr)

                total_loss += loss.item()

        test_loss = total_loss / len(test_loader)
        print(f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

        # Log and save
        logger.log_epoch(epoch, train_loss, test_loss)
        scheduler.step(test_loss)

        is_best = test_loss < best_test_loss
        if is_best:
            best_test_loss = test_loss
            print(f"✓ New best model!")

        checkpointer.save(model=model, optimizer=optimizer, epoch=epoch, loss=test_loss, is_best=is_best,
                         metadata={'train_loss': train_loss, 'test_loss': test_loss})

        if early_stopping(test_loss):
            print(f"Early stopping triggered")
            break

    print(f"\nStage 2 training complete. Best test loss: {best_test_loss:.6f}")


if __name__ == '__main__':
    main()
