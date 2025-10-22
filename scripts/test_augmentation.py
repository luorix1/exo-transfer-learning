#!/usr/bin/env python3
"""
Test script to verify that augmentation is working correctly in the dataloader.

This script loads the same data with and without augmentation to show the difference.

Usage:
    python scripts/test_augmentation.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Canonical_Camargo" \
        --subjects AB21 \
        --conditions treadmill
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src to path to import dataloader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataloader import DataHandler
from config.hyperparameters import DEFAULT_TCN_CONFIG


def test_augmentation_effect(
    dataset_root: str,
    subjects: list,
    conditions: list,
    imu_segments: list = ["pelvis", "femur"]
) -> dict:
    """
    Test the effect of augmentation by loading data with and without augmentation.
    
    Args:
        dataset_root: Path to dataset root
        subjects: List of subjects to test
        conditions: List of conditions to test
        imu_segments: IMU segments to use
    
    Returns:
        Dictionary with comparison results
    """
    # Create base hyperparameter config
    base_config = DEFAULT_TCN_CONFIG.copy()
    base_config.update({
        "window_size": 50,
        "batch_size": 8,
        "imu_segments": imu_segments,
        "dataset_proportion": 1.0,
        "validation_split": 0.2
    })
    
    results = {
        "without_augmentation": None,
        "with_augmentation": None,
        "success": False
    }
    
    try:
        # Test 1: Load data WITHOUT augmentation
        print("📊 Loading data WITHOUT augmentation...")
        config_no_aug = base_config.copy()
        config_no_aug["augment"] = False
        
        data_handler_no_aug = DataHandler(dataset_root, config_no_aug)
        data_handler_no_aug.load_data(
            train_data_partition=subjects,
            train_data_condition=conditions,
            test_data_partition=subjects
        )
        
        train_data_no_aug = data_handler_no_aug.train_data
        results["without_augmentation"] = {
            "input_data": train_data_no_aug.input.copy(),
            "label_data": train_data_no_aug.label.copy(),
            "data_length": len(train_data_no_aug.input)
        }
        
        print(f"  ✅ Loaded {results['without_augmentation']['data_length']} samples without augmentation")
        
        # Test 2: Load data WITH augmentation
        print("📊 Loading data WITH augmentation...")
        config_with_aug = base_config.copy()
        config_with_aug["augment"] = True
        
        data_handler_with_aug = DataHandler(dataset_root, config_with_aug)
        data_handler_with_aug.load_data(
            train_data_partition=subjects,
            train_data_condition=conditions,
            test_data_partition=subjects
        )
        
        train_data_with_aug = data_handler_with_aug.train_data
        results["with_augmentation"] = {
            "input_data": train_data_with_aug.input.copy(),
            "label_data": train_data_with_aug.label.copy(),
            "data_length": len(train_data_with_aug.input)
        }
        
        print(f"  ✅ Loaded {results['with_augmentation']['data_length']} samples with augmentation")
        
        results["success"] = True
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
    
    return results


def create_augmentation_comparison_plot(
    results: dict,
    output_dir: Path,
    dataset_name: str
) -> None:
    """
    Create a comparison plot showing the effect of augmentation.
    
    Args:
        results: Dictionary with augmentation test results
        output_dir: Directory to save the plot
        dataset_name: Name for the plot
    """
    if not results["success"]:
        print("❌ Cannot create plot - test failed")
        return
    
    no_aug_data = results["without_augmentation"]["input_data"]
    with_aug_data = results["with_augmentation"]["input_data"]
    
    # Take a sample of data for visualization (first 1000 samples)
    sample_size = min(1000, len(no_aug_data), len(with_aug_data))
    no_aug_sample = no_aug_data[:sample_size]
    with_aug_sample = with_aug_data[:sample_size]
    
    # Create time array
    time_data = np.arange(sample_size) / 100.0  # Assuming 100Hz
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot first gyro channel (pelvis x-axis)
    ax1 = axes[0, 0]
    ax1.plot(time_data, no_aug_sample[:, 0], color='blue', linewidth=1, alpha=0.8, label='No Augmentation')
    ax1.plot(time_data, with_aug_sample[:, 0], color='red', linewidth=1, alpha=0.8, label='With Augmentation')
    ax1.set_ylabel('Angular Velocity (rad/s)', fontsize=12, fontweight='bold')
    ax1.set_title('Pelvis X-Axis Gyro (First 1000 samples)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot second gyro channel (pelvis y-axis)
    ax2 = axes[0, 1]
    ax2.plot(time_data, no_aug_sample[:, 1], color='blue', linewidth=1, alpha=0.8, label='No Augmentation')
    ax2.plot(time_data, with_aug_sample[:, 1], color='red', linewidth=1, alpha=0.8, label='With Augmentation')
    ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=12, fontweight='bold')
    ax2.set_title('Pelvis Y-Axis Gyro (First 1000 samples)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot third gyro channel (pelvis z-axis)
    ax3 = axes[1, 0]
    ax3.plot(time_data, no_aug_sample[:, 2], color='blue', linewidth=1, alpha=0.8, label='No Augmentation')
    ax3.plot(time_data, with_aug_sample[:, 2], color='red', linewidth=1, alpha=0.8, label='With Augmentation')
    ax3.set_ylabel('Angular Velocity (rad/s)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax3.set_title('Pelvis Z-Axis Gyro (First 1000 samples)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot fourth gyro channel (femur x-axis)
    ax4 = axes[1, 1]
    ax4.plot(time_data, no_aug_sample[:, 3], color='blue', linewidth=1, alpha=0.8, label='No Augmentation')
    ax4.plot(time_data, with_aug_sample[:, 3], color='red', linewidth=1, alpha=0.8, label='With Augmentation')
    ax4.set_ylabel('Angular Velocity (rad/s)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax4.set_title('Femur X-Axis Gyro (First 1000 samples)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add statistics text
    stats_text = f"""
Dataset: {dataset_name}
Sample Size: {sample_size} samples

No Augmentation:
  Mean: {np.mean(no_aug_sample, axis=0)}
  Std:  {np.std(no_aug_sample, axis=0)}

With Augmentation:
  Mean: {np.mean(with_aug_sample, axis=0)}
  Std:  {np.std(with_aug_sample, axis=0)}

Difference (With - No):
  Mean: {np.mean(with_aug_sample, axis=0) - np.mean(no_aug_sample, axis=0)}
  Std:  {np.std(with_aug_sample, axis=0) - np.std(no_aug_sample, axis=0)}
"""
    
    # Add statistics as text box
    fig.text(0.02, 0.02, stats_text, fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'Augmentation Effect Comparison - {dataset_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = f'augmentation_comparison_{dataset_name.replace("/", "_")}.png'
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Augmentation comparison plot saved: {plot_path}")
    
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Test augmentation functionality in dataloader'
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
        help='List of subjects to test (default: AB21)'
    )
    parser.add_argument(
        '--conditions',
        type=str,
        nargs='+',
        default=['treadmill'],
        help='List of conditions to test (default: treadmill)'
    )
    parser.add_argument(
        '--imu-segments',
        type=str,
        nargs='+',
        default=['pelvis', 'femur'],
        help='IMU segments to use (default: pelvis femur)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./augmentation_test',
        help='Directory to save plots'
    )
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    
    if not dataset_root.exists():
        print(f"❌ Dataset root does not exist: {dataset_root}")
        return
    
    print(f"🔍 Testing augmentation for dataset: {dataset_root}")
    print(f"📊 Subjects: {args.subjects}")
    print(f"📊 Conditions: {args.conditions}")
    print(f"📊 IMU Segments: {args.imu_segments}")
    
    # Test augmentation effect
    print("\n🧪 Testing augmentation effect...")
    results = test_augmentation_effect(
        str(dataset_root),
        args.subjects,
        args.conditions,
        args.imu_segments
    )
    
    if not results["success"]:
        print(f"❌ Test failed: {results.get('error', 'Unknown error')}")
        if 'traceback' in results:
            print(f"\nTraceback:\n{results['traceback']}")
        return
    
    print("✅ Augmentation test completed successfully!")
    
    # Print comparison statistics
    no_aug = results["without_augmentation"]
    with_aug = results["with_augmentation"]
    
    print(f"\n📊 Results:")
    print(f"  Data length (no aug): {no_aug['data_length']}")
    print(f"  Data length (with aug): {with_aug['data_length']}")
    
    # Calculate differences
    no_aug_mean = np.mean(no_aug["input_data"], axis=0)
    with_aug_mean = np.mean(with_aug["input_data"], axis=0)
    no_aug_std = np.std(no_aug["input_data"], axis=0)
    with_aug_std = np.std(with_aug["input_data"], axis=0)
    
    print(f"\n📊 Mean values:")
    print(f"  No augmentation: {no_aug_mean}")
    print(f"  With augmentation: {with_aug_mean}")
    print(f"  Difference: {with_aug_mean - no_aug_mean}")
    
    print(f"\n📊 Standard deviations:")
    print(f"  No augmentation: {no_aug_std}")
    print(f"  With augmentation: {with_aug_std}")
    print(f"  Difference: {with_aug_std - no_aug_std}")
    
    # Create visualization
    print("\n📊 Creating comparison visualization...")
    dataset_name = f"{args.subjects[0]}_{args.conditions[0]}"
    create_augmentation_comparison_plot(results, output_dir, dataset_name)
    
    print("✅ Augmentation test complete!")


if __name__ == '__main__':
    main()
