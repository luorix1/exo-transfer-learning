#!/usr/bin/env python3
"""
Debug script to plot ground truth hip flexion moment time series for a specific subject/trial.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))
from data.dataloader import DataHandler
from config.hyperparameters import DEFAULT_TCN_CONFIG


def plot_gt_hip_moment(data_root: str, subject: str, condition: str, trial: str, 
                      imu_segments: list, label_filter_hz: float = 6.0, 
                      save_path: str = None, experiment_dir: str = None):
    """Plot ground truth hip flexion moment for a specific trial using DataHandler for proper normalization."""
    
    # Create minimal config for dataloader
    config = DEFAULT_TCN_CONFIG.copy()
    config.update({
        'data_root': data_root,
        'window_size': 100,
        'batch_size': 32,
        'number_of_workers': 0,
        'validation_split': 0.2,
        'dataset_proportion': 1.0,
        'transfer_learning': False,
        'imu_segments': imu_segments,
        'input_size': 6 if len(imu_segments) == 2 else 3,
        'output_size': 1,
        'label_filter_hz': label_filter_hz
    })
    
    # Initialize DataHandler
    data_handler = DataHandler(
        data_root=data_root,
        hyperparam_config=config,
        pretrained_model_path=experiment_dir  # Use experiment dir for normalization stats
    )
    
    # Load data for the specific subject - use a dummy test subject to avoid empty partition error
    try:
        data_handler.load_data(
            train_data_partition=[subject],
            train_data_condition=[condition],
            test_data_partition=[subject]  # Use same subject to avoid empty partition
        )
        
        # Get the loaded data
        if data_handler.train_data is None or len(data_handler.train_data.label_list) == 0:
            print(f"Error: No data loaded for {subject}/{condition}")
            return
        
        # Use the first trial's data (since we're loading just one subject/condition)
        hip_moment_normalized = data_handler.train_data.label_list[0].flatten()
        
        # Denormalize to get original units
        hip_moment = hip_moment_normalized * data_handler.train_data.label_std[0] + data_handler.train_data.label_mean[0]
        
        # Create time axis (assuming 100 Hz sampling rate)
        time_axis = np.arange(len(hip_moment)) / 100.0  # Convert to seconds
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        ax.plot(time_axis, hip_moment, 'b-', linewidth=1.5, label='Hip Flexion Moment (GT)')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Hip Flexion Moment (N-m/kg)', fontsize=12)
        ax.set_title(f'Ground Truth Hip Flexion Moment: {subject}/{condition}/{trial}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Add statistics text
        mean_moment = np.mean(hip_moment)
        std_moment = np.std(hip_moment)
        min_moment = np.min(hip_moment)
        max_moment = np.max(hip_moment)
        span_moment = max_moment - min_moment
        
        stats_text = f'Mean: {mean_moment:.4f} N-m/kg\n'
        stats_text += f'Std: {std_moment:.4f} N-m/kg\n'
        stats_text += f'Range: [{min_moment:.4f}, {max_moment:.4f}]\n'
        stats_text += f'Span: {span_moment:.4f} N-m/kg\n'
        stats_text += f'Filter: {label_filter_hz} Hz'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
        
        # Print summary statistics
        print(f"\nTrial: {subject}/{condition}/{trial}")
        print(f"Duration: {len(hip_moment)/100.0:.2f} seconds ({len(hip_moment)} samples)")
        print(f"Mean: {mean_moment:.4f} N-m/kg")
        print(f"Std: {std_moment:.4f} N-m/kg")
        print(f"Min: {min_moment:.4f} N-m/kg")
        print(f"Max: {max_moment:.4f} N-m/kg")
        print(f"Span: {span_moment:.4f} N-m/kg")
        print(f"Label filter: {label_filter_hz} Hz")
        if experiment_dir:
            print(f"Using normalization from: {experiment_dir}")
            print(f"Label mean: {data_handler.train_data.label_mean[0]:.4f}")
            print(f"Label std: {data_handler.train_data.label_std[0]:.4f}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return


def main():
    parser = argparse.ArgumentParser(description="Plot ground truth hip flexion moment for a specific trial")
    parser.add_argument("--data_root", required=True, help="Path to Canonical dataset")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., AB01_Jimin)")
    parser.add_argument("--condition", required=True, help="Condition (e.g., 1p2mps)")
    parser.add_argument("--trial", required=True, help="Trial name (e.g., trial_1)")
    parser.add_argument("--imu_segments", nargs="+", default=["pelvis", "femur"], 
                       help="IMU segments to use (default: ['pelvis', 'femur'])")
    parser.add_argument("--label_filter_hz", type=float, default=6.0,
                    help="Low-pass cutoff frequency for label filtering (default: 6.0)")
    parser.add_argument("--save", type=str, default=None,
                       help="Save plot to file (e.g., gt_plot.png)")
    parser.add_argument("--experiment_dir", type=str, default=None,
                       help="Path to experiment directory with normalization stats (e.g., 20251003_1/)")
    
    args = parser.parse_args()
    
    # Verify data path exists
    trial_path = Path(args.data_root) / args.subject / args.condition / args.trial
    if not trial_path.exists():
        print(f"Error: Trial path does not exist: {trial_path}")
        return
    
    # Check for required files
    imu_file = trial_path / "Input" / "imu_data.csv"
    label_file = trial_path / "Label" / "joint_moment.csv"
    
    if not imu_file.exists():
        print(f"Error: IMU file not found: {imu_file}")
        return
    if not label_file.exists():
        print(f"Error: Label file not found: {label_file}")
        return
    
    print(f"Plotting GT hip flexion moment for: {args.subject}/{args.condition}/{args.trial}")
    print(f"Data root: {args.data_root}")
    print(f"IMU segments: {args.imu_segments}")
    print(f"Label filter: {args.label_filter_hz} Hz")
    
    plot_gt_hip_moment(
        data_root=args.data_root,
        subject=args.subject,
        condition=args.condition,
        trial=args.trial,
        imu_segments=args.imu_segments,
        label_filter_hz=args.label_filter_hz,
        save_path=args.save,
        experiment_dir=args.experiment_dir
    )


if __name__ == "__main__":
    main()
