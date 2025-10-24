#!/usr/bin/env python3
"""
Compare original IMU gyro data with OpenSim-aligned gyro data for femur_r segment.
This script loads both datasets and creates comparison plots to verify alignment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# Add the project root to the path so we can import from analysis
sys.path.append(str(Path(__file__).parent.parent))

from analysis.imu_orientation_optimization import (
    load_real_imu_data_for_segment, 
    generate_simulated_imu_data,
    align_time_series,
    robust_plane_aware_orientation
)


def load_original_imu_data(imu_file):
    """Load original IMU data from CSV file."""
    try:
        df = pd.read_csv(imu_file, dtype=str)  # Read as strings first to avoid dtype issues
        # Convert to numeric, coercing errors to NaN
        for col in df.columns:
            if col != 'time':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with any NaN values
        df = df.dropna()
        
        print(f"✓ Loaded original IMU data: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading original IMU data: {e}")
        return None


def load_canonical_imu_data(canonical_file):
    """Load canonical (transformed) IMU data from CSV file."""
    try:
        df = pd.read_csv(canonical_file, dtype=str)  # Read as strings first to avoid dtype issues
        # Convert to numeric, coercing errors to NaN
        for col in df.columns:
            if col != 'time':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with any NaN values
        df = df.dropna()
        
        print(f"✓ Loaded canonical IMU data: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading canonical IMU data: {e}")
        return None


def extract_gyro_data(df, segment="femur_r", is_canonical=False):
    """Extract gyro data for a specific segment."""
    if segment == "femur_r":
        # Both original and canonical datasets use femur_r_gyro_x format
        gyro_cols = ["femur_r_gyro_x", "femur_r_gyro_y", "femur_r_gyro_z"]
    elif segment == "pelvis":
        gyro_cols = ["pelvis_gyro_x", "pelvis_gyro_y", "pelvis_gyro_z"]
    elif segment == "tibia_r":
        # Both original and canonical datasets use tibia_r_gyro_x format
        gyro_cols = ["tibia_r_gyro_x", "tibia_r_gyro_y", "tibia_r_gyro_z"]
    else:
        print(f"❌ Unknown segment: {segment}")
        return None, None
    
    # Check if columns exist
    missing_cols = [col for col in gyro_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns for {segment}: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return None, None
    
    gyro_data = df[gyro_cols].values
    time_data = pd.to_numeric(df['time'], errors='coerce').values
    
    print(f"✓ Extracted {segment} gyro data: {gyro_data.shape}")
    return gyro_data, time_data


def load_opensim_simulation_data(canonical_root, subject, condition, trial, segment, max_frames=2000):
    """Load OpenSim simulation data for comparison."""
    try:
        from analysis.imu_orientation_optimization import generate_simulated_imu_data
        
        # Find OpenSim files in canonical dataset
        model_file = Path(canonical_root) / subject / "opensim" / f"{subject}.osim"
        motion_file = Path(canonical_root) / subject / condition / trial / "opensim" / "motion.sto"
        
        if not model_file.exists():
            print(f"❌ Model file not found: {model_file}")
            return None, None, None
            
        if not motion_file.exists():
            print(f"❌ Motion file not found: {motion_file}")
            return None, None, None
        
        # Generate simulated IMU data
        sim_a, sim_g, sim_t, ok = generate_simulated_imu_data(
            model_file, motion_file, segment, f"{segment}_imu", max_frames
        )
        
        if not ok:
            print("❌ Could not generate simulated IMU data")
            return None, None, None
        
        print(f"✓ Generated OpenSim simulation data: {sim_g.shape}")
        return sim_g, sim_t, True
        
    except Exception as e:
        print(f"❌ Error loading OpenSim simulation data: {e}")
        return None, None, False


def create_comparison_plots(canonical_gyro, canonical_time, 
                          opensim_gyro, opensim_time, segment, output_dir):
    """Create comparison plots between canonical and OpenSim gyro data."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Align time series for comparison
    from analysis.imu_orientation_optimization import align_time_series
    
    # Align canonical with OpenSim  
    canon_aligned, opensim_aligned, time_grid, ok = align_time_series(
        opensim_time, opensim_gyro, canonical_time, canonical_gyro
    )
    
    if not ok:
        print("❌ Could not align time series")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    axis_names = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    
    for i in range(3):
        ax = axes[i]
        ax.plot(time_grid, canon_aligned[:, i], color=colors[i], alpha=0.7, 
                label=f'Canonical {axis_names[i]}', linewidth=2)
        ax.plot(time_grid, opensim_aligned[:, i], color=colors[i], 
                linestyle='--', alpha=0.9, label=f'OpenSim {axis_names[i]}', linewidth=2)
        ax.set_title(f'Canonical vs OpenSim - {axis_names[i]}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Gyro Data Alignment: {segment.upper()}\n'
                f'Subject: AB06, Condition: treadmill, Trial: treadmill_01_01', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'gyro_alignment_{segment}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved alignment plot: {output_file}")
    
    # Create 3D axes visualization
    create_axes_visualization(canon_aligned, opensim_aligned, segment, output_dir)
    
    # Calculate and print correlation coefficients
    print(f"\n📊 Alignment Analysis for {segment}:")
    for i, axis in enumerate(axis_names):
        # Canonical vs OpenSim correlation
        canon_corr = np.corrcoef(canon_aligned[:, i], opensim_aligned[:, i])[0, 1]
        
        print(f"  {axis}-axis correlation: {canon_corr:.4f}")
    
    # Overall alignment quality
    overall_corr = np.mean([np.corrcoef(canon_aligned[:, i], opensim_aligned[:, i])[0, 1] 
                           for i in range(3)])
    print(f"  Overall alignment: {overall_corr:.4f}")
    
    if overall_corr > 0.8:
        print("  ✅ Excellent alignment!")
    elif overall_corr > 0.6:
        print("  ✅ Good alignment")
    elif overall_corr > 0.4:
        print("  ⚠️  Moderate alignment")
    else:
        print("  ❌ Poor alignment - may need optimization review")


def create_axes_visualization(canonical_gyro, opensim_gyro, segment, output_dir):
    """Create 3D axes visualization comparing canonical and OpenSim frames."""
    
    # Calculate dominant axes using PCA
    from analysis.imu_orientation_optimization import dominant_axis, orthonormal_basis_from_z
    
    # Get dominant axes for both datasets
    canon_z = dominant_axis(canonical_gyro)
    opensim_z = dominant_axis(opensim_gyro)
    
    # Create orthonormal bases
    canon_basis = orthonormal_basis_from_z(canon_z)
    opensim_basis = orthonormal_basis_from_z(opensim_z)
    
    # Create 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # Define colors and labels
    colors = ["#FF4444", "#44AA44", "#4444FF"]  # Red, Green, Blue
    axis_names = ["X (Medial-Lateral)", "Y (Anterior-Posterior)", "Z (Superior-Inferior)"]
    canon_labels = ["Canonical X", "Canonical Y", "Canonical Z"]
    opensim_labels = ["OpenSim X", "OpenSim Y", "OpenSim Z"]
    
    origin = np.zeros(3)
    
    # Plot OpenSim axes (dashed, thicker)
    for i in range(3):
        ax.quiver(
            *origin,
            *opensim_basis[:, i],
            color=colors[i],
            alpha=0.6,
            linestyle="--",
            linewidth=3,
            label=opensim_labels[i],
            arrow_length_ratio=0.1
        )
    
    # Plot Canonical axes (solid, thicker)
    for i in range(3):
        ax.quiver(
            *origin,
            *canon_basis[:, i],
            color=colors[i],
            alpha=0.9,
            linestyle="-",
            linewidth=4,
            label=canon_labels[i],
            arrow_length_ratio=0.1
        )
    
    # Set equal aspect ratio and limits
    max_range = 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Add axis labels
    ax.set_xlabel("X (Medial-Lateral)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Y (Anterior-Posterior)", fontsize=12, fontweight='bold')
    ax.set_zlabel("Z (Superior-Inferior)", fontsize=12, fontweight='bold')
    
    # Enhanced title with segment info
    ax.set_title(f"IMU Frame Comparison: {segment.upper()}\n"
                f"OpenSim Canonical Frame (dashed) vs Transformed IMU Frame (solid)", 
                fontsize=14, fontweight='bold', pad=20)
    
    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates while preserving order
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    
    ax.legend(unique_handles, unique_labels, 
              loc="upper left", fontsize=10, 
              frameon=True, fancybox=True, shadow=True)
    
    # Add grid and improve appearance
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    
    # Set viewing angle for better perspective
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'axes_comparison_{segment}.png'
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    
    print(f"✓ Saved axes comparison plot: {output_file}")
    
    # Create 2D projection comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    projections = [
        (0, 1, "X-Y Plane (Right View)", "X (Medial-Lateral)", "Y (Anterior-Posterior)"),
        (2, 0, "Z-X Plane (Top View)", "Z (Superior-Inferior)", "X (Medial-Lateral)"), 
        (2, 1, "Z-Y Plane (Front View)", "Z (Superior-Inferior)", "Y (Anterior-Posterior)")
    ]
    
    for idx, (i, j, title, xlabel, ylabel) in enumerate(projections):
        ax = axes[idx]
        
        # Special handling for front view to match OpenSim convention
        if idx == 2:  # Front view
            # Plot OpenSim axes (dashed lines) - Y up, Z left
            for axis_idx in range(3):
                if axis_idx == 1:  # Y axis - should point up
                    ax.arrow(0, 0, opensim_basis[axis_idx, i], opensim_basis[axis_idx, j], 
                            head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                            alpha=0.6, linestyle='--', linewidth=2, 
                            label=f"OpenSim {['X', 'Y', 'Z'][axis_idx]}")
                elif axis_idx == 2:  # Z axis - should point left (negative horizontal)
                    ax.arrow(0, 0, -opensim_basis[axis_idx, i], opensim_basis[axis_idx, j], 
                            head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                            alpha=0.6, linestyle='--', linewidth=2, 
                            label=f"OpenSim {['X', 'Y', 'Z'][axis_idx]}")
                else:  # X axis
                    ax.arrow(0, 0, opensim_basis[axis_idx, i], opensim_basis[axis_idx, j], 
                            head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                            alpha=0.6, linestyle='--', linewidth=2, 
                            label=f"OpenSim {['X', 'Y', 'Z'][axis_idx]}")
            
            # Plot Canonical axes (solid lines) - same transformation
            for axis_idx in range(3):
                if axis_idx == 1:  # Y axis - should point up
                    ax.arrow(0, 0, canon_basis[axis_idx, i], canon_basis[axis_idx, j], 
                            head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                            alpha=0.9, linestyle='-', linewidth=3, 
                            label=f"Canonical {['X', 'Y', 'Z'][axis_idx]}")
                elif axis_idx == 2:  # Z axis - should point left (negative horizontal)
                    ax.arrow(0, 0, -canon_basis[axis_idx, i], canon_basis[axis_idx, j], 
                            head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                            alpha=0.9, linestyle='-', linewidth=3, 
                            label=f"Canonical {['X', 'Y', 'Z'][axis_idx]}")
                else:  # X axis
                    ax.arrow(0, 0, canon_basis[axis_idx, i], canon_basis[axis_idx, j], 
                            head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                            alpha=0.9, linestyle='-', linewidth=3, 
                            label=f"Canonical {['X', 'Y', 'Z'][axis_idx]}")
        else:
            # Plot OpenSim axes (dashed lines) - normal projection
            for axis_idx in range(3):
                ax.arrow(0, 0, opensim_basis[axis_idx, i], opensim_basis[axis_idx, j], 
                        head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                        alpha=0.6, linestyle='--', linewidth=2, 
                        label=f"OpenSim {['X', 'Y', 'Z'][axis_idx]}")
            
            # Plot Canonical axes (solid lines) - normal projection
            for axis_idx in range(3):
                ax.arrow(0, 0, canon_basis[axis_idx, i], canon_basis[axis_idx, j], 
                        head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                        alpha=0.9, linestyle='-', linewidth=3, 
                        label=f"Canonical {['X', 'Y', 'Z'][axis_idx]}")
        
        # Set limits and aspect
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        
        # Add proper axis labels with anatomical directions
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        
        # Add origin point
        ax.plot(0, 0, 'ko', markersize=4, alpha=0.7)
    
    plt.suptitle(f"IMU Frame Projections: {segment.upper()}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save projection plots
    output_file = output_dir / f'axes_projections_{segment}.png'
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    
    print(f"✓ Saved axes projection plots: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare gyro data alignment")
    parser.add_argument("--canonical-root", required=True, help="Path to Final/Canonical_Camargo dataset")
    parser.add_argument("--subject", default="AB06", help="Subject ID")
    parser.add_argument("--condition", default="treadmill", help="Condition")
    parser.add_argument("--trial", default="treadmill_01_01", help="Trial")
    parser.add_argument("--segment", default="femur_r", help="Segment to analyze")
    parser.add_argument("--output-dir", default="comparison_plots", help="Output directory for plots")
    parser.add_argument("--max-frames", type=int, default=2000, help="Max frames for OpenSim simulation")
    
    args = parser.parse_args()
    
    print(f"🔍 Comparing gyro data for {args.segment}")
    print(f"   Subject: {args.subject}")
    print(f"   Condition: {args.condition}")
    print(f"   Trial: {args.trial}")
    print("=" * 60)
    
    # Load canonical IMU data
    canonical_file = Path(args.canonical_root) / args.subject / args.condition / args.trial / "Input" / "imu_data.csv"
    print(f"📁 Loading canonical data: {canonical_file}")
    canonical_df = load_canonical_imu_data(canonical_file)
    if canonical_df is None:
        return
    
    # Extract gyro data
    print(f"\n📊 Extracting {args.segment} gyro data...")
    canonical_gyro, canonical_time = extract_gyro_data(canonical_df, args.segment, is_canonical=True)
    if canonical_gyro is None:
        return
    
    # Load OpenSim simulation data
    print(f"\n🎯 Loading OpenSim simulation data...")
    opensim_gyro, opensim_time, ok = load_opensim_simulation_data(
        args.canonical_root, args.subject, args.condition, args.trial, args.segment, args.max_frames
    )
    if not ok:
        return
    
    # Create comparison plots
    print(f"\n📈 Creating comparison plots...")
    create_comparison_plots(
        canonical_gyro, canonical_time,
        opensim_gyro, opensim_time,
        args.segment, args.output_dir
    )
    
    print(f"\n✅ Comparison complete!")
    print(f"   📁 Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
