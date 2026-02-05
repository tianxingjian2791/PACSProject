"""
Evaluation Script: Comprehensive Model Evaluation

Evaluates and compares different AMG approaches:
  1. Classical AMG (baseline)
  2. Learned Theta (Stage 1 only)
  3. Full Pipeline (Stage 1 + Stage 2)

Usage:
    python evaluate.py --model weights/unified/model.pt --dataset datasets/unified/test
"""

import torch
import numpy as np
import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Import models
from model import create_unified_model, GNNModel, CNNModel

# Import data
from data import create_theta_data_loaders

# Import utilities
from utils import get_device, set_random_seed


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate AMG Models')

    # Model configuration
    parser.add_argument('--model', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='unified',
                      choices=['unified', 'stage1_cnn', 'stage1_gnn'],
                      help='Model type')

    # Data configuration
    parser.add_argument('--dataset', type=str, required=True,
                      help='Test dataset directory')
    parser.add_argument('--test-file', type=str, default='test_D.csv',
                      help='Test CSV filename')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for evaluation')

    # Model hyperparameters
    parser.add_argument('--hidden-channels', type=int, default=64,
                      help='Hidden channels (for stage1 models)')
    parser.add_argument('--dropout', type=float, default=0.25,
                      help='Dropout rate')

    # Evaluation options
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')

    # Output configuration
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--save-predictions', action='store_true',
                      help='Save individual predictions to file')

    return parser.parse_args()


def load_model(args, device):
    """Load trained model"""
    print(f"\nLoading model from: {args.model}")

    if args.model_type == 'unified':
        # Load unified model
        checkpoint = torch.load(args.model, map_location=device)
        model = create_unified_model(
            stage1_type='GNN',  # Adjust based on checkpoint
            device=device
        )
        model.load_state_dict(checkpoint)

    elif args.model_type == 'stage1_gnn':
        # Load GNN theta predictor
        model = GNNModel(hidden_channels=args.hidden_channels, dropout=args.dropout)
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model = model.to(device)

    elif args.model_type == 'stage1_cnn':
        # Load CNN theta predictor
        model = CNNModel(in_channels=1, hidden_channels=args.hidden_channels, dropout=args.dropout)
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model = model.to(device)

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model.eval()
    print(f"✓ Model loaded successfully")

    return model


def evaluate_stage1(model, loader, device):
    """Evaluate Stage 1 (theta prediction)"""
    print("\nEvaluating Stage 1 (Theta Prediction)...")

    predictions = []
    targets = []
    metrics = {
        'mse': 0.0,
        'mae': 0.0,
        'relative_error': 0.0
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)

            # Forward pass
            output = model(batch)

            # Store predictions and targets
            predictions.extend(output.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    relative_error = np.mean(np.abs(predictions - targets) / (np.abs(targets) + 1e-8))

    metrics['mse'] = float(mse)
    metrics['mae'] = float(mae)
    metrics['relative_error'] = float(relative_error)

    print(f"\nStage 1 Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Relative Error: {relative_error:.4%}")

    return metrics, predictions, targets


def compute_statistics(predictions, targets):
    """Compute additional statistics"""
    stats = {}

    # Basic statistics
    stats['mean_prediction'] = float(np.mean(predictions))
    stats['std_prediction'] = float(np.std(predictions))
    stats['mean_target'] = float(np.mean(targets))
    stats['std_target'] = float(np.std(targets))

    # Error statistics
    errors = predictions - targets
    stats['mean_error'] = float(np.mean(errors))
    stats['std_error'] = float(np.std(errors))
    stats['max_error'] = float(np.max(np.abs(errors)))
    stats['min_error'] = float(np.min(np.abs(errors)))

    # Correlation
    if len(predictions) > 1:
        correlation = np.corrcoef(predictions, targets)[0, 1]
        stats['correlation'] = float(correlation)

    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    stats['r2_score'] = float(r2)

    return stats


def save_results(args, metrics, stats, predictions, targets):
    """Save evaluation results"""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"evaluation_{timestamp}.json"

    # Prepare results
    results = {
        'model': args.model,
        'model_type': args.model_type,
        'dataset': args.dataset,
        'timestamp': timestamp,
        'metrics': metrics,
        'statistics': stats,
        'num_samples': len(predictions)
    }

    # Save JSON
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Save predictions if requested
    if args.save_predictions:
        predictions_file = output_dir / f"predictions_{timestamp}.npz"
        np.savez(predictions_file,
                predictions=predictions,
                targets=targets)
        print(f"✓ Predictions saved to: {predictions_file}")

    return results_file


def print_summary(metrics, stats):
    """Print evaluation summary"""
    print("\n" + "="*60)
    print(" "*20 + "EVALUATION SUMMARY")
    print("="*60)

    print("\nPerformance Metrics:")
    print(f"  MSE:              {metrics['mse']:.6f}")
    print(f"  MAE:              {metrics['mae']:.6f}")
    print(f"  Relative Error:   {metrics['relative_error']:.4%}")

    print("\nStatistical Analysis:")
    print(f"  R² Score:         {stats.get('r2_score', 0):.4f}")
    print(f"  Correlation:      {stats.get('correlation', 0):.4f}")
    print(f"  Mean Error:       {stats['mean_error']:.6f}")
    print(f"  Std Error:        {stats['std_error']:.6f}")
    print(f"  Max Error:        {stats['max_error']:.6f}")

    print("\n" + "="*60)


def main():
    """Main evaluation function"""
    args = parse_args()

    # Set random seed
    set_random_seed(args.seed)

    print("="*60)
    print(" "*18 + "MODEL EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Type: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print("="*60)

    # Get device
    device = get_device()

    # Load model
    model = load_model(args, device)

    # Load test data
    print(f"\nLoading test data...")
    if args.model_type.startswith('stage1'):
        # Stage 1 evaluation
        _, test_loader = create_theta_data_loaders(
            data_dir=args.dataset,
            train_file='train_D.csv',  # Not used
            test_file=args.test_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        # Unified model evaluation
        print("Note: Unified model evaluation requires both Stage 1 and Stage 2 data")
        _, test_loader = create_theta_data_loaders(
            data_dir=args.dataset,
            train_file='train_D.csv',
            test_file=args.test_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    print(f"✓ Loaded {len(test_loader.dataset)} test samples")

    # Evaluate
    metrics, predictions, targets = evaluate_stage1(model, test_loader, device)

    # Compute statistics
    stats = compute_statistics(predictions, targets)

    # Print summary
    print_summary(metrics, stats)

    # Save results
    results_file = save_results(args, metrics, stats, predictions, targets)

    print("\n" + "="*60)
    print(" "*18 + "EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {results_file}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
