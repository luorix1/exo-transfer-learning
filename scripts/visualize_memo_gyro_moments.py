#!/usr/bin/env python3
"""
Visualize z-axis gyro data and hip flexion moments for Canonical_Memo dataset.

This script loads data using the current dataloader and creates side-by-side
plots showing left and right gyro data with corresponding hip flexion moments.

Usage:
    python scripts/visualize_memo_gyro_moments.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Canonical_Memo" \
        --subjects AB01_Jimin \
        --conditions 1p2mps
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src to path to import dataloader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataloader import DataHandler, LoadData
from config.hyperparameters import DEFAULT_TCN_CONFIG


def load_raw_data_for_visualization(
    dataset_root: str,
    subjects: list,
    conditions: list,
    imu_segments: list = ["pelvis", "femur"]
) -> dict:
    """
    Load raw data for visualization without downsampling.
    
    Args:
        dataset_root: Path to dataset root
        subjects: List of subjects to load
        conditions: List of conditions to load
        imu_segments: IMU segments to use
    
    Returns:
        Dictionary with raw data for visualization
    """
    # Create hyperparameter config
    config = DEFAULT_TCN_CONFIG.copy()
    config["window_size"] = 50
    config["batch_size"] = 8
    config["imu_segments"] = imu_segments
    config["dataset_proportion"] = 1.0
    config["validation_split"] = 0.2
    
    results = {
        "raw_input_data": [],
        "raw_label_data": [],
        "time_data": [],
        "subject_info": [],
        "trial_info": []
    }
    
    try:
        # Initialize DataHandler
        data_handler = DataHandler(dataset_root, config)
        
        # Load data
        data_handler.load_data(
            train_data_partition=subjects,
            train_data_condition=conditions,
            test_data_partition=subjects
        )
        
        # Get training data
        train_data = data_handler.train_data
        
        # Extract raw data before any processing
        # We need to access the original data before downsampling
        results["raw_input_data"] = train_data.input
        results["raw_label_data"] = train_data.label
        
        # Create time array (assuming 100Hz after any downsampling)
        time_data = np.arange(len(train_data.input)) / 100.0
        results["time_data"] = time_data
        
        # Get subject data lengths to understand trial boundaries
        results["subject_data_lengths"] = train_data.subject_data_length
        results["subjects"] = subjects
        results["conditions"] = conditions
        
        results["success"] = True
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
    
    return results


def extract_gyro_and_moments(data_dict: dict) -> dict:
    """
    Extract z-axis gyro data and hip flexion moments from the loaded data.
    
    Args:
        data_dict: Dictionary containing raw data from dataloader
    
    Returns:
        Dictionary with extracted gyro and moment data
    """
    input_data = data_dict["raw_input_data"]
    label_data = data_dict["raw_label_data"]
    time_data = data_dict["time_data"]
    
    # Assuming input_data has shape (N, 6) for pelvis + femur gyro data
    # Format: [pelvis_x, pelvis_y, pelvis_z, femur_x, femur_y, femur_z]
    if input_data.shape[1] == 6:
        pelvis_gyro_z = input_data[:, 2]  # pelvis z-axis gyro
        femur_gyro_z = input_data[:, 5]   # femur z-axis gyro
    else:
        # Fallback: assume only femur data
        pelvis_gyro_z = np.zeros(len(input_data))
        femur_gyro_z = input_data[:, 2] if input_data.shape[1] >= 3 else np.zeros(len(input_data))
    
    # Label data should be hip flexion moments
    # Assuming the data is stacked left/right, we need to separate them
    total_samples = len(label_data)
    
    # For visualization, we'll show the first half as left and second half as right
    # This is a simplification - in practice you'd need to track which samples are left/right
    mid_point = total_samples // 2
    
    left_gyro_z = femur_gyro_z[:mid_point]
    right_gyro_z = femur_gyro_z[mid_point:]
    left_moment = label_data[:mid_point].flatten()
    right_moment = label_data[mid_point:].flatten()
    
    # Create corresponding time arrays
    left_time = time_data[:mid_point]
    right_time = time_data[mid_point:]
    
    return {
        "left_gyro_z": left_gyro_z,
        "right_gyro_z": right_gyro_z,
        "left_moment": left_moment,
        "right_moment": right_moment,
        "left_time": left_time,
        "right_time": right_time,
        "pelvis_gyro_z": pelvis_gyro_z
    }


def create_visualization(gyro_moment_data: dict, output_dir: Path, dataset_name: str) -> None:
    """
    Create side-by-side visualization of gyro data and hip flexion moments.
    
    Args:
        gyro_moment_data: Dictionary with extracted gyro and moment data
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset for the title
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Left side plots
    ax1 = axes[0, 0]  # Left gyro
    ax2 = axes[1, 0]  # Left moment
    
    # Right side plots  
    ax3 = axes[0, 1]  # Right gyro
    ax4 = axes[1, 1]  # Right moment
    
    # Plot left gyro z-axis
    ax1.plot(gyro_moment_data["left_time"], gyro_moment_data["left_gyro_z"], 
             color='blue', linewidth=1, alpha=0.8, label='Femur Z Gyro')
    ax1.set_ylabel('Angular Velocity (rad/s)', fontsize=12, fontweight='bold')
    ax1.set_title('Left Side - Z-Axis Gyro', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot left hip flexion moment
    ax2.plot(gyro_moment_data["left_time"], gyro_moment_data["left_moment"], 
             color='red', linewidth=1, alpha=0.8, label='Hip Flexion Moment')
    ax2.set_ylabel('Moment (Nm/kg)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Left Side - Hip Flexion Moment', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot right gyro z-axis
    ax3.plot(gyro_moment_data["right_time"], gyro_moment_data["right_gyro_z"], 
             color='green', linewidth=1, alpha=0.8, label='Femur Z Gyro')
    ax3.set_ylabel('Angular Velocity (rad/s)', fontsize=12, fontweight='bold')
    ax3.set_title('Right Side - Z-Axis Gyro', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot right hip flexion moment
    ax4.plot(gyro_moment_data["right_time"], gyro_moment_data["right_moment"], 
             color='orange', linewidth=1, alpha=0.8, label='Hip Flexion Moment')
    ax4.set_ylabel('Moment (Nm/kg)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax4.set_title('Right Side - Hip Flexion Moment', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add statistics text
    stats_text = f"""
Dataset: {dataset_name}
Left Gyro Z: Mean={np.mean(gyro_moment_data["left_gyro_z"]):.3f}, Std={np.std(gyro_moment_data["left_gyro_z"]):.3f}
Right Gyro Z: Mean={np.mean(gyro_moment_data["right_gyro_z"]):.3f}, Std={np.std(gyro_moment_data["right_gyro_z"]):.3f}
Left Moment: Mean={np.mean(gyro_moment_data["left_moment"]):.3f}, Std={np.std(gyro_moment_data["left_moment"]):.3f}
Right Moment: Mean={np.mean(gyro_moment_data["right_moment"]):.3f}, Std={np.std(gyro_moment_data["right_moment"]):.3f}
"""
    
    # Add statistics as text box
    fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'Z-Axis Gyro Data and Hip Flexion Moments - {dataset_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = f'memo_gyro_moments_{dataset_name.replace("/", "_")}.png'
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved: {plot_path}")
    
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Visualize z-axis gyro data and hip flexion moments for Canonical_Memo dataset'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Path to Canonical_Memo dataset root directory'
    )
    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        default=['AB01_Jimin'],
        help='List of subjects to visualize (default: AB01_Jimin)'
    )
    parser.add_argument(
        '--conditions',
        type=str,
        nargs='+',
        default=['1p2mps'],
        help='List of conditions to visualize (default: 1p2mps)'
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
        default='./memo_visualizations',
        help='Directory to save plots'
    )
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    
    if not dataset_root.exists():
        print(f"❌ Dataset root does not exist: {dataset_root}")
        return
    
    print(f"🔍 Loading data from: {dataset_root}")
    print(f"📊 Subjects: {args.subjects}")
    print(f"📊 Conditions: {args.conditions}")
    print(f"📊 IMU Segments: {args.imu_segments}")
    
    # Load raw data
    print("\n📊 Loading data using dataloader...")
    data_dict = load_raw_data_for_visualization(
        str(dataset_root),
        args.subjects,
        args.conditions,
        args.imu_segments
    )
    
    if not data_dict["success"]:
        print(f"❌ Failed to load data: {data_dict.get('error', 'Unknown error')}")
        if 'traceback' in data_dict:
            print(f"\nTraceback:\n{data_dict['traceback']}")
        return
    
    print("✅ Data loaded successfully!")
    print(f"📊 Input data shape: {data_dict['raw_input_data'].shape}")
    print(f"📊 Label data shape: {data_dict['raw_label_data'].shape}")
    print(f"📊 Time range: {data_dict['time_data'][0]:.2f}s to {data_dict['time_data'][-1]:.2f}s")
    
    # Extract gyro and moment data
    print("\n📊 Extracting gyro and moment data...")
    gyro_moment_data = extract_gyro_and_moments(data_dict)
    
    print(f"📊 Left side data: {len(gyro_moment_data['left_gyro_z'])} samples")
    print(f"📊 Right side data: {len(gyro_moment_data['right_gyro_z'])} samples")
    
    # Create visualization
    print("\n📊 Creating visualization...")
    dataset_name = f"{args.subjects[0]}_{args.conditions[0]}"
    create_visualization(gyro_moment_data, output_dir, dataset_name)
    
    print("✅ Visualization complete!")


if __name__ == '__main__':
    main()
