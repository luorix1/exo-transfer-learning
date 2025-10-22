#!/usr/bin/env python3
"""
Diagnose gradient issues in training.

This script analyzes potential causes of large gradients and provides
recommendations for fixing training instability.

Usage:
    python scripts/diagnose_gradient_issues.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Canonical_Camargo" \
        --subjects AB21 \
        --conditions treadmill
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataloader import DataHandler
from model.tcn import TCNModel
from loss import JointMomentLoss
from config.hyperparameters import DEFAULT_TCN_CONFIG


def analyze_data_scaling(dataset_root: str, subjects: list, conditions: list) -> dict:
    """
    Analyze the scaling of input and output data.
    
    Args:
        dataset_root: Path to dataset root
        subjects: List of subjects to analyze
        conditions: List of conditions to analyze
    
    Returns:
        Dictionary with data scaling analysis
    """
    print("📊 Analyzing data scaling...")
    
    # Create config
    config = DEFAULT_TCN_CONFIG.copy()
    config.update({
        "window_size": 50,
        "batch_size": 8,
        "dataset_proportion": 1.0,
        "validation_split": 0.2
    })
    
    # Load data
    data_handler = DataHandler(dataset_root, config)
    data_handler.load_data(
        train_data_partition=subjects,
        train_data_condition=conditions,
        test_data_partition=subjects
    )
    
    train_data = data_handler.train_data
    
    analysis = {
        "input_stats": {
            "mean": np.mean(train_data.input, axis=0),
            "std": np.std(train_data.input, axis=0),
            "min": np.min(train_data.input, axis=0),
            "max": np.max(train_data.input, axis=0),
            "range": np.max(train_data.input, axis=0) - np.min(train_data.input, axis=0)
        },
        "label_stats": {
            "mean": np.mean(train_data.label, axis=0),
            "std": np.std(train_data.label, axis=0),
            "min": np.min(train_data.label, axis=0),
            "max": np.max(train_data.label, axis=0),
            "range": np.max(train_data.label, axis=0) - np.min(train_data.label, axis=0)
        },
        "normalization_applied": {
            "input_mean": train_data.input_mean,
            "input_std": train_data.input_std,
            "label_mean": train_data.label_mean,
            "label_std": train_data.label_std
        }
    }
    
    return analysis


def test_gradient_flow(model, criterion, sample_batch, device) -> dict:
    """
    Test gradient flow through the model with a sample batch.
    
    Args:
        model: The model to test
        criterion: Loss function
        sample_batch: Sample input and target batch
        device: Device to run on
    
    Returns:
        Dictionary with gradient analysis
    """
    print("📊 Testing gradient flow...")
    
    model.train()
    
    # Forward pass
    input_batch, target_batch = sample_batch
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    # Forward pass
    output = model(input_batch)
    loss = criterion(output, target_batch)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Analyze gradients
    gradient_analysis = {
        "loss_value": loss.item(),
        "layer_gradients": {},
        "total_grad_norm": 0.0,
        "max_grad_norm": 0.0,
        "min_grad_norm": float('inf'),
        "zero_gradients": 0,
        "exploding_gradients": 0
    }
    
    total_norm = 0.0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_analysis["layer_gradients"][name] = grad_norm
            gradient_analysis["max_grad_norm"] = max(gradient_analysis["max_grad_norm"], grad_norm)
            gradient_analysis["min_grad_norm"] = min(gradient_analysis["min_grad_norm"], grad_norm)
            
            if grad_norm == 0:
                gradient_analysis["zero_gradients"] += 1
            elif grad_norm > 10.0:  # Threshold for exploding gradients
                gradient_analysis["exploding_gradients"] += 1
            
            total_norm += grad_norm ** 2
            param_count += 1
        else:
            gradient_analysis["zero_gradients"] += 1
    
    gradient_analysis["total_grad_norm"] = total_norm ** 0.5
    
    return gradient_analysis


def create_diagnostic_plots(analysis: dict, output_dir: Path) -> None:
    """
    Create diagnostic plots for gradient analysis.
    
    Args:
        analysis: Dictionary with analysis results
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Input data statistics
    ax1 = axes[0, 0]
    input_stats = analysis["data_scaling"]["input_stats"]
    channels = [f"Ch{i}" for i in range(len(input_stats["mean"]))]
    
    ax1.bar(channels, input_stats["std"], alpha=0.7, color='blue', label='Std Dev')
    ax1.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax1.set_title('Input Data Standard Deviations', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Label data statistics
    ax2 = axes[0, 1]
    label_stats = analysis["data_scaling"]["label_stats"]
    
    ax2.bar(['Label'], [label_stats["std"][0]], alpha=0.7, color='red', label='Std Dev')
    ax2.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax2.set_title('Label Data Standard Deviation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gradient norms by layer
    ax3 = axes[1, 0]
    if "gradient_flow" in analysis:
        layer_names = list(analysis["gradient_flow"]["layer_gradients"].keys())
        grad_norms = list(analysis["gradient_flow"]["layer_gradients"].values())
        
        ax3.bar(range(len(layer_names)), grad_norms, alpha=0.7, color='green')
        ax3.set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
        ax3.set_title('Gradient Norms by Layer', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(layer_names)))
        ax3.set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
DIAGNOSTIC SUMMARY

Data Scaling:
  Input range: {np.max(analysis["data_scaling"]["input_stats"]["range"]):.3f}
  Label range: {analysis["data_scaling"]["label_stats"]["range"][0]:.3f}
  Input std max: {np.max(analysis["data_scaling"]["input_stats"]["std"]):.3f}
  Label std: {analysis["data_scaling"]["label_stats"]["std"][0]:.3f}

Normalization:
  Input normalized: {analysis["data_scaling"]["normalization_applied"]["input_std"] is not None}
  Label normalized: {analysis["data_scaling"]["normalization_applied"]["label_std"] is not None}

Gradient Analysis:
"""
    
    if "gradient_flow" in analysis:
        gf = analysis["gradient_flow"]
        summary_text += f"""
  Loss value: {gf["loss_value"]:.6f}
  Total grad norm: {gf["total_grad_norm"]:.3f}
  Max layer norm: {gf["max_grad_norm"]:.3f}
  Zero gradients: {gf["zero_gradients"]}
  Exploding gradients: {gf["exploding_gradients"]}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Gradient Issue Diagnostics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'gradient_diagnostics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Diagnostic plot saved: {plot_path}")
    
    plt.show()


def provide_recommendations(analysis: dict) -> None:
    """
    Provide specific recommendations based on the analysis.
    
    Args:
        analysis: Dictionary with analysis results
    """
    print("\n" + "="*60)
    print("🔧 RECOMMENDATIONS")
    print("="*60)
    
    # Check data scaling
    input_std_max = np.max(analysis["data_scaling"]["input_stats"]["std"])
    label_std = analysis["data_scaling"]["label_stats"]["std"][0]
    
    if input_std_max > 2.0:
        print("⚠️  ISSUE: Input data has high variance")
        print("   SOLUTION: Ensure input normalization is working correctly")
        print("   Check: input_std values in dataloader")
    
    if label_std > 1.0:
        print("⚠️  ISSUE: Label data has high variance")
        print("   SOLUTION: Check if labels are properly normalized")
        print("   Consider: Label scaling or different loss function")
    
    # Check gradient flow
    if "gradient_flow" in analysis:
        gf = analysis["gradient_flow"]
        
        if gf["total_grad_norm"] > 10.0:
            print("⚠️  ISSUE: Large total gradient norm")
            print("   SOLUTION: Add gradient clipping")
            print("   Add: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)")
        
        if gf["max_grad_norm"] > 5.0:
            print("⚠️  ISSUE: Some layers have very large gradients")
            print("   SOLUTION: Check layer initialization and learning rate")
            print("   Consider: Lower learning rate or better initialization")
        
        if gf["exploding_gradients"] > 0:
            print("⚠️  ISSUE: Exploding gradients detected")
            print("   SOLUTION: Reduce learning rate significantly")
            print("   Try: learning_rate = 1e-6 or 1e-7")
        
        if gf["zero_gradients"] > 0:
            print("⚠️  ISSUE: Some parameters have zero gradients")
            print("   SOLUTION: Check for dead neurons or improper loss")
            print("   Consider: Different activation functions or loss scaling")
    
    # General recommendations
    print("\n💡 GENERAL RECOMMENDATIONS:")
    print("1. Add gradient clipping to your training loop")
    print("2. Monitor gradient norms during training")
    print("3. Consider using a smaller learning rate (1e-6 or 1e-7)")
    print("4. Ensure proper data normalization")
    print("5. Check model initialization (use proper weight initialization)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Diagnose gradient issues in training'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Path to dataset root directory'
    )
    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        default=['AB21'],
        help='List of subjects to analyze (default: AB21)'
    )
    parser.add_argument(
        '--conditions',
        type=str,
        nargs='+',
        default=['treadmill'],
        help='List of conditions to analyze (default: treadmill)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./gradient_diagnostics',
        help='Directory to save diagnostic plots'
    )
    parser.add_argument(
        '--test-gradient-flow',
        action='store_true',
        help='Test gradient flow through the model'
    )
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    
    if not dataset_root.exists():
        print(f"❌ Dataset root does not exist: {dataset_root}")
        return
    
    print(f"🔍 Diagnosing gradient issues for dataset: {dataset_root}")
    print(f"📊 Subjects: {args.subjects}")
    print(f"📊 Conditions: {args.conditions}")
    
    analysis = {}
    
    # Analyze data scaling
    print("\n📊 Step 1: Analyzing data scaling...")
    analysis["data_scaling"] = analyze_data_scaling(
        str(dataset_root),
        args.subjects,
        args.conditions
    )
    
    print("✅ Data scaling analysis complete!")
    
    # Test gradient flow if requested
    if args.test_gradient_flow:
        print("\n📊 Step 2: Testing gradient flow...")
        
        # Create model
        config = DEFAULT_TCN_CONFIG.copy()
        model = TCNModel(config)
        criterion = JointMomentLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Get sample batch
        data_handler = DataHandler(dataset_root, config)
        data_handler.load_data(
            train_data_partition=args.subjects,
            train_data_condition=args.conditions,
            test_data_partition=args.subjects
        )
        
        # Create a small sample batch
        sample_input = torch.randn(4, 6, 50).to(device)  # batch_size=4, channels=6, window=50
        sample_target = torch.randn(4, 1).to(device)     # batch_size=4, output=1
        sample_batch = (sample_input, sample_target)
        
        analysis["gradient_flow"] = test_gradient_flow(model, criterion, sample_batch, device)
        print("✅ Gradient flow analysis complete!")
    
    # Create diagnostic plots
    print("\n📊 Creating diagnostic plots...")
    create_diagnostic_plots(analysis, output_dir)
    
    # Provide recommendations
    provide_recommendations(analysis)
    
    print("\n✅ Gradient diagnostics complete!")


if __name__ == '__main__':
    main()
