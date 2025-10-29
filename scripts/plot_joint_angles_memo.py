#!/usr/bin/env python3
"""
One-off script to plot joint angles from MetaMobility data to check for signage issues.
Includes hip, knee, and ankle angles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

def load_joint_angles(file_path: Path) -> pd.DataFrame:
    """Load joint angle data from MetaMobility CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from: {file_path}")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_joint_angles(df: pd.DataFrame, output_dir: Path, subject: str, condition: str, trial: str):
    """Plot joint angles to check for signage issues."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find joint angle columns
    hip_columns = [col for col in df.columns if 'Hip' in col and 'Angle' in col]
    knee_columns = [col for col in df.columns if 'Knee' in col and 'Angle' in col]
    ankle_columns = [col for col in df.columns if 'Ankle' in col and 'Angle' in col]
    
    print(f"Found hip angle columns: {hip_columns}")
    print(f"Found knee angle columns: {knee_columns}")
    print(f"Found ankle angle columns: {ankle_columns}")
    
    if not hip_columns and not knee_columns and not ankle_columns:
        print("No joint angle columns found!")
        return
    
    # Create time column if not present
    if 'time' not in df.columns:
        if 'Frame' in df.columns:
            df['time'] = df['Frame'] / 100.0  # Assuming 100 Hz
        elif 'frame' in df.columns:
            df['time'] = df['frame'] / 100.0
        else:
            df['time'] = np.arange(len(df)) / 100.0  # Default 100 Hz
    
    # Plot hip angles
    if hip_columns:
        plot_joint_group(df, hip_columns, 'Hip', output_dir, subject, condition, trial)
    
    # Plot knee angles
    if knee_columns:
        plot_joint_group(df, knee_columns, 'Knee', output_dir, subject, condition, trial)
    
    # Plot ankle angles
    if ankle_columns:
        plot_joint_group(df, ankle_columns, 'Ankle', output_dir, subject, condition, trial)
    
    # Create comparison plots for left vs right
    create_comparison_plots(df, output_dir, subject, condition, trial)

def plot_joint_group(df: pd.DataFrame, columns: list, joint_name: str, output_dir: Path, 
                    subject: str, condition: str, trial: str):
    """Plot a group of joint angles (e.g., all hip angles)."""
    
    n_angles = len(columns)
    fig, axes = plt.subplots(n_angles, 1, figsize=(12, 4*n_angles))
    if n_angles == 1:
        axes = [axes]
    
    for i, col in enumerate(columns):
        ax = axes[i]
        
        # Plot the angle data
        ax.plot(df['time'], df[col], 'b-', linewidth=1.5, label=col)
        
        # Add statistics
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        
        ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        ax.set_title(f'{col} - {subject}_{condition}_{trial}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'Mean: {mean_val:.2f}°\nStd: {std_val:.2f}°\nMin: {min_val:.2f}°\nMax: {max_val:.2f}°'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f"{joint_name.lower()}_angles_{subject}_{condition}_{trial}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved {joint_name} plot: {output_file}")
    
    plt.show()

def create_comparison_plots(df: pd.DataFrame, output_dir: Path, subject: str, condition: str, trial: str):
    """Create left vs right comparison plots for each joint."""
    
    # Hip flexion comparison
    if 'RightHipAngles_X' in df.columns and 'LeftHipAngles_X' in df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.plot(df['time'], df['RightHipAngles_X'], 'b-', linewidth=1.5, label='Right Hip Flexion')
        ax.plot(df['time'], df['LeftHipAngles_X'], 'r-', linewidth=1.5, label='Left Hip Flexion')
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_title(f'Hip Flexion Comparison - {subject}_{condition}_{trial}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        comparison_file = output_dir / f"hip_flexion_comparison_{subject}_{condition}_{trial}.png"
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        print(f"Saved hip flexion comparison: {comparison_file}")
        plt.show()
    
    # Knee flexion comparison
    if 'RightKneeAngles_X' in df.columns and 'LeftKneeAngles_X' in df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.plot(df['time'], df['RightKneeAngles_X'], 'b-', linewidth=1.5, label='Right Knee Flexion')
        ax.plot(df['time'], df['LeftKneeAngles_X'], 'r-', linewidth=1.5, label='Left Knee Flexion')
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_title(f'Knee Flexion Comparison - {subject}_{condition}_{trial}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        comparison_file = output_dir / f"knee_flexion_comparison_{subject}_{condition}_{trial}.png"
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        print(f"Saved knee flexion comparison: {comparison_file}")
        plt.show()
    
    # Ankle dorsiflexion comparison
    if 'RightAnkleAngles_X' in df.columns and 'LeftAnkleAngles_X' in df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.plot(df['time'], df['RightAnkleAngles_X'], 'b-', linewidth=1.5, label='Right Ankle Dorsiflexion')
        ax.plot(df['time'], df['LeftAnkleAngles_X'], 'r-', linewidth=1.5, label='Left Ankle Dorsiflexion')
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_title(f'Ankle Dorsiflexion Comparison - {subject}_{condition}_{trial}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        comparison_file = output_dir / f"ankle_dorsiflexion_comparison_{subject}_{condition}_{trial}.png"
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        print(f"Saved ankle dorsiflexion comparison: {comparison_file}")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot joint angles from MetaMobility data')
    parser.add_argument('--input-root', type=str, required=True,
                       help='Path to MetaMobility data root directory')
    parser.add_argument('--subject', type=str, required=True,
                       help='Subject ID (e.g., AB01_Jimin)')
    parser.add_argument('--condition', type=str, required=True,
                       help='Condition (e.g., 1p2mps)')
    parser.add_argument('--trial', type=str, required=True,
                       help='Trial (e.g., trial_1)')
    parser.add_argument('--output-dir', type=str, default='debug_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Construct file path
    input_root = Path(args.input_root)
    label_file = input_root / args.subject / args.condition / args.trial / "Label" / f"{args.subject}_{args.condition}_{args.trial.split('_')[1]}.csv"
    
    print(f"Looking for file: {label_file}")
    
    if not label_file.exists():
        print(f"❌ File not found: {label_file}")
        print("Available files in Label directory:")
        label_dir = input_root / args.subject / args.condition / args.trial / "Label"
        if label_dir.exists():
            for f in label_dir.glob("*.csv"):
                print(f"  - {f.name}")
        return
    
    # Load and plot data
    df = load_joint_angles(label_file)
    if df is not None:
        output_dir = Path(args.output_dir)
        plot_joint_angles(df, output_dir, args.subject, args.condition, args.trial)
    else:
        print("❌ Failed to load data")

if __name__ == "__main__":
    main()
