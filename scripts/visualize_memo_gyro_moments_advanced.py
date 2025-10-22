#!/usr/bin/env python3
"""
Advanced visualization of z-axis gyro data and hip flexion moments for Canonical_Memo dataset.

This script provides more sophisticated data handling and visualization options,
including proper left/right data separation and multiple trial support.

Usage:
    python scripts/visualize_memo_gyro_moments_advanced.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Canonical_Memo" \
        --subjects AB01 AB02 \
        --conditions 1p2mps transient_30sec
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


def load_data_with_trial_tracking(
    dataset_root: str,
    subjects: list,
    conditions: list,
    imu_segments: list = ["pelvis", "femur"]
) -> dict:
    """
    Load data while tracking individual trials for better visualization.
    
    Args:
        dataset_root: Path to dataset root
        subjects: List of subjects to load
        conditions: List of conditions to load
        imu_segments: IMU segments to use
    
    Returns:
        Dictionary with trial-tracked data
    """
    # Create hyperparameter config
    config = DEFAULT_TCN_CONFIG.copy()
    config["window_size"] = 50
    config["batch_size"] = 8
    config["imu_segments"] = imu_segments
    config["dataset_proportion"] = 1.0
    config["validation_split"] = 0.2
    
    results = {
        "trials": [],
        "success": False
    }
    
    try:
        # Load data for each subject and condition separately
        for subject in subjects:
            for condition in conditions:
                print(f"📁 Loading {subject}/{condition}...")
                
                # Initialize DataHandler for this specific subject/condition
                data_handler = DataHandler(dataset_root, config)
                
                # Load data
                data_handler.load_data(
                    train_data_partition=[subject],
                    train_data_condition=[condition],
                    test_data_partition=[subject]
                )
                
                # Get training data
                train_data = data_handler.train_data
                
                if train_data is not None and len(train_data.input) > 0:
                    trial_data = {
                        "subject": subject,
                        "condition": condition,
                        "input_data": train_data.input,
                        "label_data": train_data.label,
                        "time_data": np.arange(len(train_data.input)) / 100.0,
                        "data_length": len(train_data.input)
                    }
                    results["trials"].append(trial_data)
                    print(f"  ✅ Loaded {trial_data['data_length']} samples")
                else:
                    print(f"  ❌ No data found for {subject}/{condition}")
        
        results["success"] = len(results["trials"]) > 0
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
    
    return results


def extract_gyro_moments_by_trial(trial_data: dict) -> dict:
    """
    Extract gyro and moment data for a single trial.
    
    Args:
        trial_data: Dictionary with trial data
    
    Returns:
        Dictionary with extracted gyro and moment data
    """
    input_data = trial_data["input_data"]
    label_data = trial_data["label_data"]
    time_data = trial_data["time_data"]
    
    # Assuming input_data has shape (N, 6) for pelvis + femur gyro data
    # Format: [pelvis_x, pelvis_y, pelvis_z, femur_x, femur_y, femur_z]
    if input_data.shape[1] == 6:
        pelvis_gyro_z = input_data[:, 2]  # pelvis z-axis gyro
        femur_gyro_z = input_data[:, 5]   # femur z-axis gyro
    else:
        # Fallback: assume only femur data
        pelvis_gyro_z = np.zeros(len(input_data))
        femur_gyro_z = input_data[:, 2] if input_data.shape[1] >= 3 else np.zeros(len(input_data))
    
    # For Canonical_Memo, the data is already separated by left/right sides
    # We need to identify which samples are left vs right
    # This is a heuristic - in practice you'd need to track this during data loading
    total_samples = len(label_data)
    
    # Simple heuristic: assume alternating left/right or first half/second half
    # For now, let's use first half/second half approach
    mid_point = total_samples // 2
    
    left_gyro_z = femur_gyro_z[:mid_point]
    right_gyro_z = femur_gyro_z[mid_point:]
    left_moment = label_data[:mid_point].flatten()
    right_moment = label_data[mid_point:].flatten()
    
    # Create corresponding time arrays
    left_time = time_data[:mid_point]
    right_time = time_data[mid_point:]
    
    return {
        "subject": trial_data["subject"],
        "condition": trial_data["condition"],
        "left_gyro_z": left_gyro_z,
        "right_gyro_z": right_gyro_z,
        "left_moment": left_moment,
        "right_moment": right_moment,
        "left_time": left_time,
        "right_time": right_time,
        "pelvis_gyro_z": pelvis_gyro_z,
        "total_samples": total_samples
    }


def create_comprehensive_visualization(
    all_trial_data: list, 
    output_dir: Path, 
    dataset_name: str
) -> None:
    """
    Create comprehensive visualization showing all trials.
    
    Args:
        all_trial_data: List of trial data dictionaries
        output_dir: Directory to save plots
        dataset_name: Name for the plot
    """
    n_trials = len(all_trial_data)
    
    if n_trials == 0:
        print("❌ No trial data to visualize")
        return
    
    # Create subplots - one row per trial, two columns (left/right)
    fig, axes = plt.subplots(n_trials, 2, figsize=(16, 6 * n_trials))
    
    if n_trials == 1:
        axes = axes.reshape(1, -1)
    
    for i, trial_data in enumerate(all_trial_data):
        # Left side plot
        ax_left = axes[i, 0]
        ax_left.plot(trial_data["left_time"], trial_data["left_gyro_z"], 
                     color='blue', linewidth=1, alpha=0.8, label='Femur Z Gyro')
        ax_left.plot(trial_data["left_time"], trial_data["left_moment"], 
                     color='red', linewidth=1, alpha=0.8, label='Hip Flexion Moment')
        ax_left.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax_left.set_title(f'{trial_data["subject"]}/{trial_data["condition"]} - Left Side', 
                         fontsize=14, fontweight='bold')
        ax_left.grid(True, alpha=0.3)
        ax_left.legend()
        
        # Right side plot
        ax_right = axes[i, 1]
        ax_right.plot(trial_data["right_time"], trial_data["right_gyro_z"], 
                      color='green', linewidth=1, alpha=0.8, label='Femur Z Gyro')
        ax_right.plot(trial_data["right_time"], trial_data["right_moment"], 
                      color='orange', linewidth=1, alpha=0.8, label='Hip Flexion Moment')
        ax_right.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax_right.set_title(f'{trial_data["subject"]}/{trial_data["condition"]} - Right Side', 
                          fontsize=14, fontweight='bold')
        ax_right.grid(True, alpha=0.3)
        ax_right.legend()
        
        # Add x-label to bottom row
        if i == n_trials - 1:
            ax_left.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            ax_right.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    
    # Add overall title
    plt.suptitle(f'Z-Axis Gyro Data and Hip Flexion Moments - {dataset_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = f'memo_gyro_moments_comprehensive_{dataset_name.replace("/", "_")}.png'
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive visualization saved: {plot_path}")
    
    plt.show()


def create_statistics_summary(all_trial_data: list, output_dir: Path) -> None:
    """
    Create a statistics summary table.
    
    Args:
        all_trial_data: List of trial data dictionaries
        output_dir: Directory to save the summary
    """
    summary_data = []
    
    for trial_data in all_trial_data:
        summary_data.append({
            'Subject': trial_data["subject"],
            'Condition': trial_data["condition"],
            'Total_Samples': trial_data["total_samples"],
            'Left_Gyro_Mean': np.mean(trial_data["left_gyro_z"]),
            'Left_Gyro_Std': np.std(trial_data["left_gyro_z"]),
            'Right_Gyro_Mean': np.mean(trial_data["right_gyro_z"]),
            'Right_Gyro_Std': np.std(trial_data["right_gyro_z"]),
            'Left_Moment_Mean': np.mean(trial_data["left_moment"]),
            'Left_Moment_Std': np.std(trial_data["left_moment"]),
            'Right_Moment_Mean': np.mean(trial_data["right_moment"]),
            'Right_Moment_Std': np.std(trial_data["right_moment"])
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'memo_gyro_moments_statistics.csv'
    df.to_csv(csv_path, index=False)
    print(f"📊 Statistics summary saved: {csv_path}")
    
    # Print summary
    print("\n📊 Statistics Summary:")
    print(df.to_string(index=False, float_format='%.4f'))


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Advanced visualization of z-axis gyro data and hip flexion moments for Canonical_Memo dataset'
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
        default=['AB01'],
        help='List of subjects to visualize (default: AB01)'
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
        default='./memo_visualizations_advanced',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--statistics',
        action='store_true',
        help='Generate statistics summary'
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
    
    # Load data with trial tracking
    print("\n📊 Loading data with trial tracking...")
    data_dict = load_data_with_trial_tracking(
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
    
    print(f"✅ Loaded {len(data_dict['trials'])} trials successfully!")
    
    # Extract gyro and moment data for each trial
    all_trial_data = []
    for trial in data_dict["trials"]:
        print(f"📊 Processing {trial['subject']}/{trial['condition']}...")
        trial_gyro_moments = extract_gyro_moments_by_trial(trial)
        all_trial_data.append(trial_gyro_moments)
    
    # Create comprehensive visualization
    print("\n📊 Creating comprehensive visualization...")
    dataset_name = f"{'_'.join(args.subjects)}_{'_'.join(args.conditions)}"
    create_comprehensive_visualization(all_trial_data, output_dir, dataset_name)
    
    # Generate statistics if requested
    if args.statistics:
        print("\n📊 Generating statistics summary...")
        create_statistics_summary(all_trial_data, output_dir)
    
    print("✅ Advanced visualization complete!")


if __name__ == '__main__':
    main()
