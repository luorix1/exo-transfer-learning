#!/usr/bin/env python3
"""
IMU Position Optimization for Final Dataset Structure

This script optimizes the 3D position of an IMU on a body segment
using a pre-determined orientation (from orientation optimization).
The position is optimized to minimize error between simulated and real
accelerometer data.

Works with Final dataset structure:
- <dataset>/<subject>/<condition>/<trial>/Input/imu_data.csv (real IMU data)
- <dataset>/<subject>/<condition>/<trial>/opensim/motion.sto (OpenSim motion)
- <dataset>/<subject>/opensim/<subject>.osim (OpenSim model)
- Orientation results from prior optimization (JSON file)

Usage:
    python imu_position_optimization.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Camargo" \
        --subject AB21 \
        --condition treadmill \
        --trial treadmill_03_01 \
        --segment femur_r \
        --orientation-results results/camargo_multisegment/femur_r/optimization_results.json \
        --output results/position_optimization/
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


def apply_lowpass_filter(data, cutoff_freq=10.0, sample_rate=200.0):
    """Apply low-pass Butterworth filter to remove high-frequency artifacts"""
    try:
        if len(data) < 6:
            return data
        nyquist = sample_rate / 2.0
        normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(N=4, Wn=normalized_cutoff, btype="low", analog=False)
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = filtfilt(b, a, data[:, i])
        return filtered_data
    except Exception as e:
        print(f"   ⚠️  Filtering failed: {e}, returning original data")
        return data


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix"""
    w, x, y, z = q
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R


def get_segment_bounds(model, segment_name):
    """
    Extract segment dimensions from OpenSim model to define reasonable search bounds.
    
    Returns bounds in meters as (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    print(f"\n📏 Extracting segment dimensions for {segment_name}...")
    
    try:
        body = model.getBodySet().get(segment_name)
        
        # Default bounds based on typical segment sizes (in meters)
        default_bounds = {
            "femur_r": (-0.05, 0.05, -0.25, 0.05, -0.05, 0.05),  # Long axis along -Y
            "femur_l": (-0.05, 0.05, -0.25, 0.05, -0.05, 0.05),
            "tibia_r": (-0.05, 0.05, -0.30, 0.05, -0.05, 0.05),  # Long axis along -Y
            "tibia_l": (-0.05, 0.05, -0.30, 0.05, -0.05, 0.05),
            "pelvis": (-0.10, 0.10, -0.10, 0.10, -0.10, 0.10),
        }
        
        bounds = default_bounds.get(segment_name, (-0.10, 0.10, -0.20, 0.10, -0.10, 0.10))
        
        # Try to get actual geometry info if available
        try:
            # Get attached geometry
            geom_set = body.getPropertyByName("attached_geometry")
            if geom_set.size() > 0:
                print(f"   Found {geom_set.size()} geometry objects attached to {segment_name}")
        except:
            pass
        
        print(f"   Using bounds: X=[{bounds[0]:.3f}, {bounds[1]:.3f}]m, "
              f"Y=[{bounds[2]:.3f}, {bounds[3]:.3f}]m, "
              f"Z=[{bounds[4]:.3f}, {bounds[5]:.3f}]m")
        
        return bounds
        
    except Exception as e:
        print(f"   ⚠️  Could not extract geometry: {e}")
        # Default conservative bounds
        default = (-0.10, 0.10, -0.20, 0.10, -0.10, 0.10)
        print(f"   Using default bounds: X=[{default[0]:.3f}, {default[1]:.3f}]m, "
              f"Y=[{default[2]:.3f}, {default[3]:.3f}]m, "
              f"Z=[{default[4]:.3f}, {default[5]:.3f}]m")
        return default


def generate_simulated_imu_data_at_position(
    model_file, motion_file, segment_name, imu_name, 
    position_offset, orientation_matrix, max_frames
):
    """
    Generate simulated IMU data at a specific position with known orientation.
    
    Args:
        model_file: Path to OpenSim model
        motion_file: Path to motion file
        segment_name: Name of body segment
        imu_name: Name for IMU
        position_offset: [x, y, z] offset in segment frame (meters)
        orientation_matrix: 3x3 rotation matrix for IMU orientation
        max_frames: Maximum number of frames to process
    
    Returns:
        sim_acc_signals: Nx3 array of accelerometer data
        sim_times: N array of time stamps
        success: boolean
    """
    try:
        # Load model and add IMU
        geometry_path = Path("/Applications/OpenSim 4.5/Geometry")
        osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geometry_path))
        
        model = osim.Model(model_file)
        model.setUseVisualizer(False)
        
        # Create and attach IMU to target segment
        target_body = model.getBodySet().get(segment_name)
        imu = osim.IMU()
        imu.setName(imu_name)
        
        # Create a PhysicalOffsetFrame with both position and orientation
        imu_frame = osim.PhysicalOffsetFrame()
        imu_frame.setName(f"{imu_name}_frame")
        imu_frame.setParentFrame(target_body)
        
        # Create rotation from orientation matrix
        rotation = osim.Rotation(osim.Mat33(
            orientation_matrix[0, 0], orientation_matrix[0, 1], orientation_matrix[0, 2],
            orientation_matrix[1, 0], orientation_matrix[1, 1], orientation_matrix[1, 2],
            orientation_matrix[2, 0], orientation_matrix[2, 1], orientation_matrix[2, 2]
        ))
        
        # Create translation (position offset in segment frame)
        vec3_offset = osim.Vec3(position_offset[0], position_offset[1], position_offset[2])
        
        # Create transform with both rotation and translation
        transform = osim.Transform(rotation, vec3_offset)
        
        imu_frame.setOffsetTransform(transform)
        
        # Attach the frame to the target body and the IMU to the frame
        target_body.addComponent(imu_frame)
        imu.connectSocket_frame(imu_frame)
        model.addComponent(imu)
        
        # Initialize the system
        state = model.initSystem()
        
        # Load motion data
        motion_table = osim.TimeSeriesTable(motion_file)
        time_col = motion_table.getIndependentColumn()
        coord_names = list(motion_table.getColumnLabels())
        
        # Extract simulated IMU data
        sim_acc_signals = []
        sim_times = []
        
        num_frames = min(max_frames, motion_table.getNumRows())
        
        for i in range(num_frames):
            t = time_col[i]
            state.setTime(t)
            
            # Set coordinates and speeds
            for coord in model.getCoordinateSet():
                q_name = coord.getName()
                u_name = q_name + "_u"
                if q_name in coord_names:
                    coord.setValue(state, motion_table.getDependentColumn(q_name)[i])
                if u_name in coord_names:
                    coord.setSpeedValue(state, motion_table.getDependentColumn(u_name)[i])
            
            # Realize acceleration to compute accelerometer signal
            model.realizeAcceleration(state)
            
            acc = imu.calcAccelerometerSignal(state)
            # Convert from mm/s² to m/s² (OpenSim returns accelerations in mm/s²)
            acc_array = np.array([acc.get(j) / 1000.0 for j in range(3)])
            
            sim_acc_signals.append(acc_array)
            sim_times.append(t)
            
            del acc
        
        sim_acc_signals = np.array(sim_acc_signals)
        sim_times = np.array(sim_times)
        
        # Clean up OpenSim objects
        del imu, imu_frame, target_body, state, model
        
        return sim_acc_signals, sim_times, True
        
    except Exception as e:
        print(f"❌ Error generating simulated IMU data: {e}")
        return None, None, False


def load_real_imu_acceleration(real_imu_file, segment_name):
    """Load real IMU acceleration data from CSV file"""
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
        
        # Extract accelerometer data (try multiple naming conventions)
        acc_candidates = [
            [f"{segment_name}_accel_x", f"{segment_name}_accel_y", f"{segment_name}_accel_z"],
            [f"{segment_name}_Accel_X", f"{segment_name}_Accel_Y", f"{segment_name}_Accel_Z"],
            [f"{segment_name}_acc_x", f"{segment_name}_acc_y", f"{segment_name}_acc_z"],
            [f"{segment_name.upper()}_ACCELX", f"{segment_name.upper()}_ACCELY", f"{segment_name.upper()}_ACCELZ"],
        ]
        
        acc_cols_actual = None
        for candidate_set in acc_candidates:
            acc_cols_actual = get_cols(candidate_set)
            if acc_cols_actual is not None:
                print(f"✓ Found accelerometer columns: {acc_cols_actual}")
                break
        
        if acc_cols_actual is None:
            print(f"❌ No accelerometer columns found for segment '{segment_name}' in {real_imu_file}")
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


def align_time_series(sim_times, sim_acc_signals, real_times, real_acc_signals):
    """Align simulated and real IMU data in time"""
    # Find overlapping time range
    overlap_start = max(sim_times[0], real_times[0])
    overlap_end = min(sim_times[-1], real_times[-1])
    
    if overlap_start >= overlap_end:
        print(f"❌ No overlapping time range found!")
        return None, None, None, False
    
    # Create common time grid
    num_points = min(len(sim_times), len(real_times), 1000)
    common_times = np.linspace(overlap_start, overlap_end, num_points)
    
    # Interpolate both datasets to common time grid
    sim_interp = interp1d(
        sim_times,
        sim_acc_signals,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=0,
    )
    real_interp = interp1d(
        real_times,
        real_acc_signals,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=0,
    )
    
    sim_aligned = sim_interp(common_times)
    real_aligned = real_interp(common_times)
    
    return sim_aligned, real_aligned, common_times, True


class PositionOptimizer:
    """
    Optimizer for IMU position on a body segment.
    Uses global optimization (differential evolution) to find optimal position.
    """
    
    def __init__(self, model_file, motion_file, real_imu_file, segment_name, 
                 imu_data_name, orientation_matrix, bounds, max_frames=2000):
        self.model_file = model_file
        self.motion_file = motion_file
        self.real_imu_file = real_imu_file
        self.segment_name = segment_name
        self.imu_data_name = imu_data_name
        self.orientation_matrix = orientation_matrix
        self.bounds = bounds
        self.max_frames = max_frames
        
        # Load real IMU data once
        print(f"\n🔄 Loading real IMU acceleration data...")
        self.real_acc_signals, self.real_times, success = load_real_imu_acceleration(
            real_imu_file, imu_data_name
        )
        if not success:
            raise ValueError("Failed to load real IMU data")
        
        # Cache for optimization
        self.eval_count = 0
        self.best_cost = float('inf')
        self.best_position = None
    
    def objective_function(self, position):
        """
        Objective function for optimization.
        Evaluates MSE between simulated and real acceleration at given position.
        
        Args:
            position: [x, y, z] position offset in meters
        
        Returns:
            cost: Mean squared error
        """
        self.eval_count += 1
        
        # Generate simulated data at this position
        sim_acc, sim_times, success = generate_simulated_imu_data_at_position(
            self.model_file,
            self.motion_file,
            self.segment_name,
            f"imu_pos_opt_{self.eval_count}",
            position,
            self.orientation_matrix,
            self.max_frames
        )
        
        if not success:
            return 1e6  # Large penalty for failed simulation
        
        # Align time series
        sim_aligned, real_aligned, _, success = align_time_series(
            sim_times, sim_acc, self.real_times, self.real_acc_signals
        )
        
        if not success:
            return 1e6
        
        # Compute MSE
        cost = np.mean((sim_aligned - real_aligned) ** 2)
        
        # Track best solution
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_position = position.copy()
            print(f"   Eval {self.eval_count}: New best cost = {cost:.6f} at position {position}")
        
        return cost
    
    def optimize(self, method='differential_evolution', circle_radius=0.04):
        """
        Run optimization to find best IMU position.
        
        Args:
            method: 'differential_evolution' (global) or 'nelder-mead' (local) or 
                    'two-stage' (global+local) or 'circular' (search on circle)
            circle_radius: Radius in meters for circular search (default: 0.04 = 4cm)
        
        Returns:
            result: optimization result dictionary
        """
        print(f"\n🔄 Starting position optimization using {method}...")
        
        if method != 'circular':
            print(f"   Search bounds: X=[{self.bounds[0]:.3f}, {self.bounds[1]:.3f}]m")
            print(f"                  Y=[{self.bounds[2]:.3f}, {self.bounds[3]:.3f}]m")
            print(f"                  Z=[{self.bounds[4]:.3f}, {self.bounds[5]:.3f}]m")
        
        # Prepare bounds for scipy
        opt_bounds = [
            (self.bounds[0], self.bounds[1]),  # x bounds
            (self.bounds[2], self.bounds[3]),  # y bounds
            (self.bounds[4], self.bounds[5]),  # z bounds
        ]
        
        if method == 'differential_evolution':
            # Global optimization - good for finding global minimum
            result = differential_evolution(
                self.objective_function,
                bounds=opt_bounds,
                maxiter=50,  # Limit iterations due to expensive function
                popsize=10,  # Population size
                atol=1e-4,
                tol=1e-4,
                seed=42,
                workers=1,  # Sequential to avoid OpenSim issues
                updating='deferred',
                disp=True
            )
            
            optimal_position = result.x
            final_cost = result.fun
            
        elif method == 'two-stage':
            # Two-stage optimization: global search + local refinement
            print("\n🎯 Stage 1: Global search with differential evolution...")
            print("   This explores the entire search space to find a good starting point")
            
            stage1_result = differential_evolution(
                self.objective_function,
                bounds=opt_bounds,
                maxiter=30,  # Fewer iterations for stage 1
                popsize=8,   # Smaller population
                atol=1e-3,
                tol=1e-3,
                seed=42,
                workers=1,
                updating='deferred',
                disp=True
            )
            
            stage1_position = stage1_result.x
            stage1_cost = stage1_result.fun
            
            print(f"\n✓ Stage 1 complete!")
            print(f"   Best position found: [{stage1_position[0]:.4f}, {stage1_position[1]:.4f}, {stage1_position[2]:.4f}]m")
            print(f"   Cost: {stage1_cost:.6f}")
            
            print("\n🎯 Stage 2: Local refinement with Nelder-Mead...")
            print("   This polishes the solution with fine-grained local search")
            
            stage2_result = minimize(
                self.objective_function,
                stage1_position,  # Start from Stage 1's best result
                method='Nelder-Mead',
                options={'maxiter': 50, 'disp': True, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            optimal_position = stage2_result.x
            final_cost = stage2_result.fun
            
            print(f"\n✓ Stage 2 complete!")
            print(f"   Refined position: [{optimal_position[0]:.4f}, {optimal_position[1]:.4f}, {optimal_position[2]:.4f}]m")
            print(f"   Final cost: {final_cost:.6f}")
            print(f"   Improvement from Stage 1: {((stage1_cost - final_cost) / stage1_cost * 100):.2f}%")
            
            # Use stage2_result as the final result
            result = stage2_result
            
        elif method == 'circular':
            # Circular search: optimize angle θ on a circle of fixed radius in XY plane (z=0)
            print(f"   Searching on circle: radius = {circle_radius:.3f}m, z = 0")
            print(f"   Position formula: x = {circle_radius:.3f} * cos(θ), y = {circle_radius:.3f} * sin(θ), z = 0")
            
            def circular_objective(theta):
                """Objective function parameterized by angle θ (in radians)"""
                # Convert angle to XYZ position on circle
                position = np.array([
                    circle_radius * np.cos(theta[0]),
                    circle_radius * np.sin(theta[0]),
                    0.0  # Fixed z = 0
                ])
                return self.objective_function(position)
            
            # Search angle from 0 to 2π
            result = minimize(
                circular_objective,
                x0=np.array([0.0]),  # Start at θ=0 (position [r, 0, 0])
                method='Nelder-Mead',
                bounds=[(0, 2*np.pi)],
                options={'maxiter': 50, 'disp': True}
            )
            
            # Convert optimal angle back to position
            optimal_theta = result.x[0]
            optimal_position = np.array([
                circle_radius * np.cos(optimal_theta),
                circle_radius * np.sin(optimal_theta),
                0.0
            ])
            final_cost = result.fun
            
            print(f"\n✓ Circular optimization complete!")
            print(f"   Optimal angle: {optimal_theta:.4f} rad ({np.degrees(optimal_theta):.1f}°)")
            print(f"   Optimal position: [{optimal_position[0]:.4f}, {optimal_position[1]:.4f}, {optimal_position[2]:.4f}]m")
            print(f"   Distance from origin: {circle_radius:.4f}m (fixed)")
            
        else:  # nelder-mead or other local methods
            # Start from center of bounds
            x0 = np.array([
                (self.bounds[0] + self.bounds[1]) / 2,
                (self.bounds[2] + self.bounds[3]) / 2,
                (self.bounds[4] + self.bounds[5]) / 2,
            ])
            
            result = minimize(
                self.objective_function,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 100, 'disp': True}
            )
            
            optimal_position = result.x
            final_cost = result.fun
        
        print(f"\n✓ Optimization completed!")
        print(f"   Total evaluations: {self.eval_count}")
        print(f"   Optimal position: [{optimal_position[0]:.4f}, {optimal_position[1]:.4f}, {optimal_position[2]:.4f}]m")
        print(f"   Final cost (MSE): {final_cost:.6f}")
        
        # Evaluate at origin for comparison
        print(f"\n   Computing cost at segment origin for comparison...")
        origin_cost = self.objective_function(np.array([0.0, 0.0, 0.0]))
        improvement = ((origin_cost - final_cost) / origin_cost) * 100
        
        print(f"   Cost at origin: {origin_cost:.6f}")
        print(f"   Improvement: {improvement:.2f}%")
        
        return {
            'success': result.success if hasattr(result, 'success') else True,
            'optimal_position': optimal_position,
            'final_cost': final_cost,
            'origin_cost': origin_cost,
            'improvement_percent': improvement,
            'evaluations': self.eval_count,
            'optimization_result': result
        }


def create_visualizations(optimizer, optimization_result, output_dir):
    """Create visualization plots"""
    print("\n🔄 Creating visualizations...")
    
    optimal_position = optimization_result['optimal_position']
    
    # Generate data at optimal position and at origin for comparison
    print("   Generating comparison data...")
    
    # At optimal position
    sim_acc_opt, sim_times_opt, _ = generate_simulated_imu_data_at_position(
        optimizer.model_file,
        optimizer.motion_file,
        optimizer.segment_name,
        "imu_optimal",
        optimal_position,
        optimizer.orientation_matrix,
        optimizer.max_frames
    )
    
    # At origin
    sim_acc_origin, sim_times_origin, _ = generate_simulated_imu_data_at_position(
        optimizer.model_file,
        optimizer.motion_file,
        optimizer.segment_name,
        "imu_origin",
        np.array([0.0, 0.0, 0.0]),
        optimizer.orientation_matrix,
        optimizer.max_frames
    )
    
    # Align all time series
    sim_opt_aligned, real_aligned, common_times, _ = align_time_series(
        sim_times_opt, sim_acc_opt, optimizer.real_times, optimizer.real_acc_signals
    )
    
    sim_origin_aligned, _, _, _ = align_time_series(
        sim_times_origin, sim_acc_origin, optimizer.real_times, optimizer.real_acc_signals
    )
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    colors = ["red", "green", "blue"]
    labels = ["X", "Y", "Z"]
    
    # Plot individual axes comparison
    for i in range(3):
        # At origin
        axes[i, 0].plot(
            common_times,
            sim_origin_aligned[:, i],
            color=colors[i],
            linewidth=1.5,
            alpha=0.7,
            label=f"Simulated {labels[i]} (Origin)",
        )
        axes[i, 0].plot(
            common_times,
            real_aligned[:, i],
            color=colors[i],
            linewidth=1.5,
            alpha=0.9,
            linestyle="--",
            label=f"Real IMU {labels[i]}",
        )
        
        axes[i, 0].set_ylabel(f"Acceleration {labels[i]} (m/s²)")
        axes[i, 0].set_title(f"At Segment Origin - Axis {labels[i]}")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # At optimal position
        axes[i, 1].plot(
            common_times,
            sim_opt_aligned[:, i],
            color=colors[i],
            linewidth=1.5,
            alpha=0.7,
            label=f"Simulated {labels[i]} (Optimal)",
        )
        axes[i, 1].plot(
            common_times,
            real_aligned[:, i],
            color=colors[i],
            linewidth=1.5,
            alpha=0.9,
            linestyle="--",
            label=f"Real IMU {labels[i]}",
        )
        
        axes[i, 1].set_ylabel(f"Acceleration {labels[i]} (m/s²)")
        axes[i, 1].set_title(f"At Optimal Position - Axis {labels[i]}")
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    
    plt.tight_layout()
    comparison_plot_path = Path(output_dir) / "position_optimization_comparison.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved comparison plot: {comparison_plot_path}")
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Magnitude comparison
    sim_magnitude_origin = np.linalg.norm(sim_origin_aligned, axis=1)
    sim_magnitude_opt = np.linalg.norm(sim_opt_aligned, axis=1)
    real_magnitude = np.linalg.norm(real_aligned, axis=1)
    
    ax1.plot(
        common_times,
        sim_magnitude_origin,
        "b-",
        linewidth=2,
        alpha=0.7,
        label="Simulated (Origin)",
    )
    ax1.plot(
        common_times,
        sim_magnitude_opt,
        "r-",
        linewidth=2,
        alpha=0.7,
        label="Simulated (Optimal Position)",
    )
    ax1.plot(
        common_times, real_magnitude, "k--", linewidth=2, alpha=0.9, label="Real IMU"
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Acceleration Magnitude (m/s²)")
    ax1.set_title("Magnitude Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cost comparison bar chart
    costs = [optimization_result['origin_cost'], optimization_result['final_cost']]
    methods = ["At Segment\nOrigin", "At Optimal\nPosition"]
    colors_bar = ["lightcoral", "lightgreen"]
    
    bars = ax2.bar(methods, costs, color=colors_bar, alpha=0.7)
    ax2.set_ylabel("Mean Squared Error (m²/s⁴)")
    ax2.set_title(
        f'Position Optimization Results\n{optimization_result["improvement_percent"]:.1f}% Improvement'
    )
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{cost:.6f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    
    plt.tight_layout()
    summary_plot_path = Path(output_dir) / "position_optimization_summary.png"
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved summary plot: {summary_plot_path}")
    
    # Create 3D visualization of position
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot segment coordinate frame axes
    origin = np.zeros(3)
    axes_length = 0.1  # 10cm axes
    
    ax.quiver(0, 0, 0, axes_length, 0, 0, color='r', arrow_length_ratio=0.2, linewidth=2, label='X (segment)')
    ax.quiver(0, 0, 0, 0, axes_length, 0, color='g', arrow_length_ratio=0.2, linewidth=2, label='Y (segment)')
    ax.quiver(0, 0, 0, 0, 0, axes_length, color='b', arrow_length_ratio=0.2, linewidth=2, label='Z (segment)')
    
    # Plot optimal IMU position
    ax.scatter(
        optimal_position[0], optimal_position[1], optimal_position[2],
        color='purple', s=200, marker='o', label='Optimal IMU Position', alpha=0.8
    )
    
    # Draw line from origin to optimal position
    ax.plot(
        [0, optimal_position[0]], 
        [0, optimal_position[1]], 
        [0, optimal_position[2]],
        'k--', linewidth=1.5, alpha=0.5
    )
    
    # Add text label with coordinates
    ax.text(
        optimal_position[0], optimal_position[1], optimal_position[2],
        f'  ({optimal_position[0]:.3f}, {optimal_position[1]:.3f}, {optimal_position[2]:.3f})m',
        fontsize=10
    )
    
    # Set equal aspect ratio and labels
    max_range = max(
        abs(optimizer.bounds[1] - optimizer.bounds[0]),
        abs(optimizer.bounds[3] - optimizer.bounds[2]),
        abs(optimizer.bounds[5] - optimizer.bounds[4])
    ) / 2
    
    mid_x = (optimizer.bounds[0] + optimizer.bounds[1]) / 2
    mid_y = (optimizer.bounds[2] + optimizer.bounds[3]) / 2
    mid_z = (optimizer.bounds[4] + optimizer.bounds[5]) / 2
    
    ax.set_xlim([mid_x - max_range, mid_x + max_range])
    ax.set_ylim([mid_y - max_range, mid_y + max_range])
    ax.set_zlim([mid_z - max_range, mid_z + max_range])
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Optimal IMU Position on {optimizer.segment_name}")
    ax.legend()
    
    plt.tight_layout()
    position_plot_path = Path(output_dir) / "optimal_position_3d.png"
    plt.savefig(position_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved 3D position plot: {position_plot_path}")


def save_results(optimization_result, model_file, motion_file, real_imu_file,
                segment_name, orientation_file, output_dir):
    """Save optimization results to files"""
    print("\n💾 Saving results...")
    
    # Save detailed results as JSON
    results_data = {
        "configuration": {
            "model_file": str(model_file),
            "motion_file": str(motion_file),
            "real_imu_file": str(real_imu_file),
            "segment_name": segment_name,
            "orientation_file": str(orientation_file)
        },
        "optimization": {
            "optimal_position": optimization_result['optimal_position'].tolist(),
            "final_cost": float(optimization_result['final_cost']),
            "origin_cost": float(optimization_result['origin_cost']),
            "improvement_percent": float(optimization_result['improvement_percent']),
            "success": bool(optimization_result['success']),
            "evaluations": int(optimization_result['evaluations'])
        }
    }
    
    results_path = Path(output_dir) / "position_optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"✓ Saved results: {results_path}")
    
    # Save position vector
    position_path = Path(output_dir) / "optimal_position.txt"
    np.savetxt(
        position_path,
        optimization_result['optimal_position'].reshape(1, -1),
        header="Optimal IMU position offset [x, y, z] in segment frame (meters)",
        fmt='%.6f'
    )
    
    print(f"✓ Saved position vector: {position_path}")


def find_dataset_files(dataset_root, subject, condition, trial):
    """Find the required files in the Final dataset structure"""
    dataset_path = Path(dataset_root)
    
    # Find model file
    model_file = dataset_path / subject / "opensim" / f"{subject}.osim"
    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        return None, None, None
    
    # Find motion file
    motion_file = dataset_path / subject / condition / trial / "opensim" / "motion.sto"
    if not motion_file.exists():
        print(f"❌ Motion file not found: {motion_file}")
        return None, None, None
    
    # Find IMU data file
    imu_file = dataset_path / subject / condition / trial / "Input" / "imu_data.csv"
    if not imu_file.exists():
        print(f"❌ IMU data file not found: {imu_file}")
        return None, None, None
    
    return model_file, motion_file, imu_file


def load_orientation_results(orientation_file):
    """Load orientation optimization results from JSON file"""
    print(f"\n📂 Loading orientation results from {orientation_file}...")
    
    try:
        with open(orientation_file, 'r') as f:
            data = json.load(f)
        
        # Extract rotation matrix
        rotation_matrix = np.array(data['optimization']['optimal_rotation_matrix'])
        
        print(f"✓ Loaded orientation matrix:")
        print(rotation_matrix)
        
        return rotation_matrix, True
        
    except Exception as e:
        print(f"❌ Error loading orientation results: {e}")
        return None, False


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="IMU Position Optimization for Final Dataset")
    parser.add_argument(
        "--dataset-root", required=True, 
        help="Path to Final dataset root (e.g., /path/to/Final/Camargo)"
    )
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., AB21)")
    parser.add_argument("--condition", required=True, help="Condition (e.g., treadmill)")
    parser.add_argument("--trial", required=True, help="Trial name (e.g., treadmill_03_01)")
    parser.add_argument("--segment", required=True, 
                       help="Body segment (e.g., femur_r, tibia_r, pelvis)")
    parser.add_argument(
        "--orientation-results", required=True,
        help="Path to orientation optimization results JSON file"
    )
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument(
        "--imu-data-name", default=None,
        help="Name of IMU data columns (default: inferred from segment)"
    )
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
    
    # Infer IMU data name from segment if not provided
    segment_to_imu_data = {
        "femur_r": "thigh_r",
        "femur_l": "thigh_l",
        "tibia_r": "shank_r",
        "tibia_l": "shank_l",
        "pelvis": "pelvis"
    }
    
    imu_data_name = args.imu_data_name if args.imu_data_name else segment_to_imu_data.get(args.segment, args.segment)
    
    print("🚀 IMU Position Optimization for Final Dataset")
    print("=" * 70)
    print(f"Dataset: {args.dataset_root}")
    print(f"Subject: {args.subject}")
    print(f"Condition: {args.condition}")
    print(f"Trial: {args.trial}")
    print(f"Segment: {args.segment}")
    print(f"IMU data name: {imu_data_name}")
    print(f"Output: {args.output}")
    print(f"Orientation file: {args.orientation_results}")
    print("=" * 70)
    
    try:
        # Step 1: Find dataset files
        model_file, motion_file, real_imu_file = find_dataset_files(
            args.dataset_root, args.subject, args.condition, args.trial
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
        
        # Step 4: Create optimizer and run optimization
        optimizer = PositionOptimizer(
            model_file=str(model_file),
            motion_file=str(motion_file),
            real_imu_file=str(real_imu_file),
            segment_name=args.segment,
            imu_data_name=imu_data_name,
            orientation_matrix=orientation_matrix,
            bounds=bounds,
            max_frames=args.max_frames
        )
        
        optimization_result = optimizer.optimize(
            method=args.optimization_method,
            circle_radius=args.circle_radius
        )
        
        if not optimization_result['success']:
            print("❌ Optimization failed")
            sys.exit(1)
        
        # Step 5: Create visualizations
        create_visualizations(optimizer, optimization_result, args.output)
        
        # Step 6: Save results
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

