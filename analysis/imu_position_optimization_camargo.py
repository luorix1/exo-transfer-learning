#!/usr/bin/env python3
"""
IMU Position Optimization for Camargo Processed Dataset

This script optimizes the 3D position of an IMU on a body segment
using a pre-determined orientation (from orientation optimization).
The position is optimized to minimize error between simulated and real
accelerometer data.

Works with Camargo_processed dataset structure:
- <dataset>/<subject>/<date>/<condition>/imu/<trial>.csv (real IMU data)
- <dataset>/<subject>/<date>/<condition>/opensim/<trial>/walking_motion_states.sto
- <dataset>/<subject>/osimxml/<subject>.osim (OpenSim model)
- Orientation results from prior optimization (JSON file)

Usage:
    python imu_position_optimization_camargo.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Camargo_processed" \
        --subject AB21 \
        --date 01_27_2019 \
        --condition treadmill \
        --trial treadmill_03_01 \
        --segment femur_r \
        --orientation-results results/camargo_multisegment/femur_r/optimization_results.json \
        --output results/position_optimization_camargo/
"""

import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, minimize
from scipy.signal import butter, filtfilt
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Import core functions from main script
sys.path.insert(0, str(Path(__file__).parent))
from imu_position_optimization import (
    quaternion_to_rotation_matrix,
    get_segment_bounds,
    generate_simulated_imu_data_at_position,
    align_time_series,
    PositionOptimizer,
    create_visualizations,
    save_results,
    load_orientation_results,
    apply_lowpass_filter
)


def find_camargo_dataset_files(dataset_root, subject, date, condition, trial):
    """Find the required files in the Camargo_processed dataset structure"""
    dataset_path = Path(dataset_root)
    
    # Find model file
    model_file = dataset_path / subject / "osimxml" / f"{subject}.osim"
    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        return None, None, None
    
    # Find motion file
    motion_file = dataset_path / subject / date / condition / "opensim" / trial / "walking_motion_states.sto"
    if not motion_file.exists():
        print(f"❌ Motion file not found: {motion_file}")
        return None, None, None
    
    # Find IMU data file
    imu_file = dataset_path / subject / date / condition / "imu" / f"{trial}.csv"
    if not imu_file.exists():
        print(f"❌ IMU data file not found: {imu_file}")
        return None, None, None
    
    return model_file, motion_file, imu_file


def load_camargo_imu_acceleration(real_imu_file, segment_name):
    """
    Load real IMU acceleration data from Camargo processed CSV file
    
    Camargo column naming:
    - thigh_Accel_X/Y/Z (for femur_r and femur_l)
    - shank_Accel_X/Y/Z (for tibia_r and tibia_l)
    - trunk_Accel_X/Y/Z (for pelvis)
    - foot_Accel_X/Y/Z (for foot/calcn)
    """
    try:
        imu_df = pd.read_csv(real_imu_file)
        
        # Build case-insensitive column name map
        lower_to_actual = {c.lower(): c for c in imu_df.columns}
        
        def get_cols(candidates):
            actual = []
            for name in candidates:
                key = name.lower()
                if key in lower_to_actual:
                    actual.append(lower_to_actual[key])
                else:
                    return None
            return actual
        
        # Map OpenSim segment names to Camargo IMU names
        # Camargo uses: thigh, shank, trunk, foot (no left/right distinction in column names)
        segment_to_camargo = {
            "femur_r": "thigh",
            "femur_l": "thigh",
            "tibia_r": "shank",
            "tibia_l": "shank",
            "pelvis": "trunk",
            "calcn_r": "foot",
            "calcn_l": "foot",
        }
        
        camargo_name = segment_to_camargo.get(segment_name, segment_name)
        
        # Extract accelerometer data (try multiple naming conventions)
        acc_candidates = [
            [f"{camargo_name}_Accel_X", f"{camargo_name}_Accel_Y", f"{camargo_name}_Accel_Z"],
            [f"{camargo_name}_accel_x", f"{camargo_name}_accel_y", f"{camargo_name}_accel_z"],
        ]
        
        acc_cols_actual = None
        for candidate_set in acc_candidates:
            acc_cols_actual = get_cols(candidate_set)
            if acc_cols_actual is not None:
                print(f"✓ Found accelerometer columns: {acc_cols_actual}")
                break
        
        if acc_cols_actual is None:
            print(f"❌ No accelerometer columns found for Camargo segment '{camargo_name}' in {real_imu_file}")
            print(f"   Available columns: {list(imu_df.columns)}")
            return None, None, False
        
        real_acc_signals = imu_df[acc_cols_actual].values
        
        # Time/Header column (case-insensitive)
        header_col = lower_to_actual.get('header', lower_to_actual.get('time', None))
        real_times = imu_df[header_col].values if header_col else np.arange(len(real_acc_signals))
        
        print(f"✓ Loaded real accelerometer data: {len(real_times)} frames")
        print(f"   Time range: {real_times[0]:.3f}s to {real_times[-1]:.3f}s")
        print(f"   Accel range: {np.min(real_acc_signals, axis=0)} to {np.max(real_acc_signals, axis=0)} m/s²")
        
        return real_acc_signals, real_times, True
        
    except Exception as e:
        print(f"❌ Error loading real IMU acceleration data: {e}")
        return None, None, False


def plot_initial_comparison(optimizer, output_dir, initial_position=None):
    """
    Plot simulated vs real acceleration at initial position (before optimization).
    
    Args:
        optimizer: PositionOptimizer instance
        output_dir: Output directory for saving plot
        initial_position: Initial position to test (default: [0, 0, 0] origin)
    """
    print("\n📊 Generating initial position comparison plot...")
    
    if initial_position is None:
        initial_position = np.array([0.0, 0.0, 0.0])
    
    print(f"   Initial position: [{initial_position[0]:.3f}, {initial_position[1]:.3f}, {initial_position[2]:.3f}]m")
    
    # Generate simulated data at initial position
    sim_acc, sim_times, success = generate_simulated_imu_data_at_position(
        optimizer.model_file,
        optimizer.motion_file,
        optimizer.segment_name,
        "imu_initial",
        initial_position,
        optimizer.orientation_matrix,
        optimizer.max_frames
    )
    
    if not success:
        print("   ⚠️  Failed to generate simulated data at initial position")
        return
    
    # Align time series
    sim_aligned, real_aligned, common_times, success = align_time_series(
        sim_times, sim_acc, optimizer.real_times, optimizer.real_acc_signals
    )
    
    if not success:
        print("   ⚠️  Failed to align time series")
        return
    
    # Compute initial MSE
    initial_mse = np.mean((sim_aligned - real_aligned) ** 2)
    print(f"   Initial MSE: {initial_mse:.6f} m²/s⁴")
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    colors = ["red", "green", "blue"]
    labels = ["X", "Y", "Z"]
    
    for i in range(3):
        # Plot simulated
        axes[i].plot(
            common_times,
            sim_aligned[:, i],
            color=colors[i],
            linewidth=1.5,
            alpha=0.7,
            label=f"Simulated {labels[i]}",
        )
        
        # Plot real
        axes[i].plot(
            common_times,
            real_aligned[:, i],
            color=colors[i],
            linewidth=1.5,
            alpha=0.9,
            linestyle="--",
            label=f"Real IMU {labels[i]}",
        )
        
        # Calculate axis-specific MSE
        axis_mse = np.mean((sim_aligned[:, i] - real_aligned[:, i]) ** 2)
        
        axes[i].set_ylabel(f"Acceleration {labels[i]} (m/s²)", fontsize=11)
        axes[i].set_title(
            f"Initial Comparison - Axis {labels[i]} (MSE: {axis_mse:.6f})",
            fontsize=12
        )
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    
    fig.suptitle(
        f"Initial Position Comparison (Segment: {optimizer.segment_name})\n"
        f"Position: [{initial_position[0]:.3f}, {initial_position[1]:.3f}, {initial_position[2]:.3f}]m | "
        f"Total MSE: {initial_mse:.6f}",
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plot_path = Path(output_dir) / "initial_position_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"   ✓ Saved initial comparison plot: {plot_path}")
    
    return initial_mse


class CamargoPositionOptimizer(PositionOptimizer):
    """
    Position optimizer specialized for Camargo processed dataset.
    Overrides the IMU data loading to use Camargo column names.
    """
    
    def __init__(self, model_file, motion_file, real_imu_file, segment_name, 
                 orientation_matrix, bounds, max_frames=2000):
        # Don't call parent __init__ directly since it tries to load data
        self.model_file = model_file
        self.motion_file = motion_file
        self.real_imu_file = real_imu_file
        self.segment_name = segment_name
        self.orientation_matrix = orientation_matrix
        self.bounds = bounds
        self.max_frames = max_frames
        
        # Load real IMU data using Camargo-specific loader
        print(f"\n🔄 Loading real IMU acceleration data (Camargo format)...")
        self.real_acc_signals, self.real_times, success = load_camargo_imu_acceleration(
            real_imu_file, segment_name
        )
        if not success:
            raise ValueError("Failed to load real IMU data")
        
        # Cache for optimization
        self.eval_count = 0
        self.best_cost = float('inf')
        self.best_position = None


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="IMU Position Optimization for Camargo Processed Dataset")
    parser.add_argument(
        "--dataset-root", required=True, 
        help="Path to Camargo_processed dataset root"
    )
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., AB21)")
    parser.add_argument("--date", required=True, help="Date folder (e.g., 01_27_2019)")
    parser.add_argument("--condition", required=True, help="Condition (e.g., treadmill, levelground)")
    parser.add_argument("--trial", required=True, help="Trial name (e.g., treadmill_03_01)")
    parser.add_argument("--segment", required=True, 
                       help="Body segment (e.g., femur_r, tibia_r, pelvis)")
    parser.add_argument(
        "--orientation-results", required=True,
        help="Path to orientation optimization results JSON file"
    )
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument(
        "--max-frames", type=int, default=1000,
        help="Maximum frames to process (default: 1000, lower for position opt)"
    )
    parser.add_argument(
        "--optimization-method", default="two-stage",
        choices=["differential_evolution", "nelder-mead", "two-stage", "circular"],
        help="Optimization method (default: two-stage). "
             "Options: differential_evolution (global), nelder-mead (local), "
             "two-stage (global+local refinement - recommended), "
             "circular (search on 4cm radius circle in XY plane - fast)"
    )
    parser.add_argument(
        "--circle-radius", type=float, default=0.04,
        help="Radius in meters for circular search method (default: 0.04 = 4cm)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(exist_ok=True, parents=True)
    
    print("🚀 IMU Position Optimization for Camargo Processed Dataset")
    print("=" * 70)
    print(f"Dataset: {args.dataset_root}")
    print(f"Subject: {args.subject}")
    print(f"Date: {args.date}")
    print(f"Condition: {args.condition}")
    print(f"Trial: {args.trial}")
    print(f"Segment: {args.segment}")
    print(f"Output: {args.output}")
    print(f"Orientation file: {args.orientation_results}")
    print("=" * 70)
    
    try:
        # Step 1: Find dataset files
        model_file, motion_file, real_imu_file = find_camargo_dataset_files(
            args.dataset_root, args.subject, args.date, args.condition, args.trial
        )
        
        if model_file is None:
            sys.exit(1)
        
        print(f"Model: {model_file}")
        print(f"Motion: {motion_file}")
        print(f"Real IMU: {real_imu_file}")
        
        # Step 2: Load orientation results
        orientation_matrix, success = load_orientation_results(args.orientation_results)
        if not success:
            sys.exit(1)
        
        # Step 3: Get segment bounds from model
        geometry_path = Path("/Applications/OpenSim 4.5/Geometry")
        osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geometry_path))
        model = osim.Model(str(model_file))
        model.setUseVisualizer(False)
        
        bounds = get_segment_bounds(model, args.segment)
        del model
        
        # Step 4: Create optimizer (using Camargo version)
        optimizer = CamargoPositionOptimizer(
            model_file=str(model_file),
            motion_file=str(motion_file),
            real_imu_file=str(real_imu_file),
            segment_name=args.segment,
            orientation_matrix=orientation_matrix,
            bounds=bounds,
            max_frames=args.max_frames
        )
        
        # Step 4.5: Plot initial position comparison (before optimization)
        initial_mse = plot_initial_comparison(optimizer, args.output)
        
        # Step 5: Run optimization
        optimization_result = optimizer.optimize(
            method=args.optimization_method,
            circle_radius=args.circle_radius
        )
        
        if not optimization_result['success']:
            print("❌ Optimization failed")
            sys.exit(1)
        
        # Step 6: Create visualizations
        create_visualizations(optimizer, optimization_result, args.output)
        
        # Step 7: Save results
        save_results(
            optimization_result,
            model_file,
            motion_file,
            real_imu_file,
            args.segment,
            args.orientation_results,
            args.output
        )
        
        print(f"\n🎉 IMU Position Optimization Completed Successfully!")
        print(f"📊 Summary:")
        print(f"   Optimal position: [{optimization_result['optimal_position'][0]:.4f}, "
              f"{optimization_result['optimal_position'][1]:.4f}, "
              f"{optimization_result['optimal_position'][2]:.4f}]m")
        print(f"   MSE at origin: {optimization_result['origin_cost']:.6f}")
        print(f"   MSE at optimal position: {optimization_result['final_cost']:.6f}")
        print(f"   Improvement: {optimization_result['improvement_percent']:.2f}%")
        print(f"   Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

