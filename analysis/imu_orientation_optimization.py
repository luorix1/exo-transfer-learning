#!/usr/bin/env python3
"""
IMU Orientation Optimization for Final Dataset Structure

This script optimizes the 3-DoF orientation (quaternion) of an IMU
to align simulated angular velocity with actual IMU gyro data.

Works with Final dataset structure:
- <dataset>/<subject>/<condition>/<trial>/Input/imu_data.csv (real IMU data)
- <dataset>/<subject>/<condition>/<trial>/opensim/motion.sto (OpenSim motion)
- <dataset>/<subject>/opensim/<subject>.osim (OpenSim model)

Usage:
    python imu_orientation_optimization.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Molinaro_Phase1_Phase2" \
        --subject AB01 \
        --condition levelground \
        --trial LG_C0p0_S0p0_BT_1_10 \
        --segment femur_r \
        --output results/
"""

import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import json
from pathlib import Path
from tqdm import tqdm
import sys
import os


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


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion [w, x, y, z]"""
    # Ensure R is a proper rotation matrix
    R = np.array(R)
    
    # Method from Shepperd's algorithm
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def kabsch_algorithm(sim_data, real_data):
    """
    Kabsch algorithm to find optimal rotation matrix using SVD.
    
    This solves the orthogonal Procrustes problem:
    min ||R*P - Q||_F subject to R^T*R = I
    
    where P is sim_data, Q is real_data, and R is the rotation matrix.
    
    Args:
        sim_data: Nx3 array of simulated IMU data
        real_data: Nx3 array of real IMU data
    
    Returns:
        optimal_rotation_matrix: 3x3 rotation matrix
        cost: mean squared error after optimal rotation
    """
    # Center the data (remove mean)
    sim_centered = sim_data - np.mean(sim_data, axis=0)
    real_centered = real_data - np.mean(real_data, axis=0)
    
    # Compute cross-covariance matrix H = P^T * Q
    H = sim_centered.T @ real_centered
    
    # SVD decomposition: H = U * S * V^T
    U, S, Vt = np.linalg.svd(H)
    
    # Optimal rotation matrix: R = V * U^T
    optimal_rotation_matrix = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1, not -1)
    if np.linalg.det(optimal_rotation_matrix) < 0:
        # If determinant is -1, flip the last column of V
        Vt[-1, :] *= -1
        optimal_rotation_matrix = Vt.T @ U.T
    
    # Apply rotation and compute final cost
    rotated_sim = np.array([optimal_rotation_matrix @ sim_frame for sim_frame in sim_data])
    residual = rotated_sim - real_data
    cost = np.mean(residual**2)
    
    return optimal_rotation_matrix, cost


def generate_simulated_imu_data(
    model_file, motion_file, segment_name, imu_name, max_frames
):
    """Generate simulated IMU data from OpenSim model and motion"""
    print("\n🔄 Step 1: Generating simulated IMU data...")

    try:
        # Look for the .vtp files in "/Applications/OpenSim 4.5/Geometry"
        geometry_path = Path("/Applications/OpenSim 4.5/Geometry")
        osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geometry_path))

        # Load model and add IMU
        model = osim.Model(model_file)
        model.setUseVisualizer(False)

        # Create and attach IMU to target segment
        target_body = model.getBodySet().get(segment_name)
        imu = osim.IMU()
        imu.setName(imu_name)

        # Create a PhysicalOffsetFrame on the target body for the IMU
        imu_frame = osim.PhysicalOffsetFrame()
        imu_frame.setName(f"{imu_name}_frame")
        imu_frame.setParentFrame(target_body)
        # Set frame identical to the attached segment
        imu_frame.setOffsetTransform(osim.Transform())

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

        is_states = any(name.endswith("_u") for name in coord_names)

        print(f"✓ Loaded motion file with {motion_table.getNumRows()} frames")
        print(f"✓ Added IMU to {segment_name}")
        print(f"   Detected format: {'STO (states)' if is_states else 'MOT (coordinates)'}")
        print(f"   Available coordinates: {coord_names[:5]}..." if len(coord_names) > 5 else f"   Available coordinates: {coord_names}")
        
        # Check for NaN values in the motion data
        has_nan = False
        for i in range(min(5, motion_table.getNumRows())):
            for coord_name in coord_names:
                if coord_name != 'time':
                    value = motion_table.getDependentColumn(coord_name)[i]
                    if str(value) == 'nan' or np.isnan(float(value)):
                        has_nan = True
                        break
            if has_nan:
                break
        
        if has_nan:
            print(f"⚠️  Warning: Motion file contains NaN values. This may cause zero gyro data.")
            print(f"   This usually means the motion file coordinate names don't match the model.")
            print(f"   Expected model coordinates: pelvis_tilt, pelvis_list, hip_flexion_r, knee_angle_r, etc.")
            print(f"   Found motion coordinates: {coord_names[:10]}...")

        # Extract simulated IMU data
        sim_acc_signals = []
        sim_gyro_signals = []
        sim_times = []

        print("   Processing motion frames...")
        num_frames = min(max_frames, motion_table.getNumRows())

        for i in tqdm(range(num_frames)):
            t = time_col[i]
            state.setTime(t)

            # STO file: directly set coordinates (q) and speeds (u), already in radians
            for coord in model.getCoordinateSet():
                q_name = coord.getName()
                u_name = q_name + "_u"
                if q_name in coord_names:
                    coord.setValue(
                        state, motion_table.getDependentColumn(q_name)[i]
                    )
                if u_name in coord_names:
                    coord.setSpeedValue(
                        state, motion_table.getDependentColumn(u_name)[i]
                    )
            
            # Compute both gyro and accel signals
            model.realizeAcceleration(state)

            acc = imu.calcAccelerometerSignal(state)
            gyro = imu.calcGyroscopeSignal(state)

            acc_array = np.array([acc.get(j) for j in range(3)])
            gyro_array = np.array([gyro.get(j) for j in range(3)])

            sim_acc_signals.append(acc_array)
            sim_gyro_signals.append(gyro_array)
            sim_times.append(t)

            # Clear the acc and gyro objects to prevent memory leak
            del acc, gyro

            if i % 500 == 0:
                print(f"   Progress: {i}/{num_frames} frames ({100*i/num_frames:.1f}%)")

        sim_acc_signals = np.array(sim_acc_signals)
        sim_gyro_signals = np.array(sim_gyro_signals)
        sim_times = np.array(sim_times)

        print(f"✓ Generated simulated gyro data: {len(sim_times)} frames")
        print(f"   Time range: {sim_times[0]:.3f}s to {sim_times[-1]:.3f}s")
        print(f"   Gyro range: {np.min(sim_gyro_signals, axis=0)} to {np.max(sim_gyro_signals, axis=0)} rad/s")
        
        # Check if gyro data is all zeros
        if np.allclose(sim_gyro_signals, 0.0):
            print(f"⚠️  Warning: Simulated gyro data is all zeros!")
            print(f"   This usually indicates:")
            print(f"   1. Motion file has NaN values (coordinate name mismatch)")
            print(f"   2. Model coordinates are not being set properly")
            print(f"   3. IMU attachment point is incorrect")
            print(f"   Check that motion file coordinates match model coordinates.")
            print(f"   Model expects: pelvis_tilt, pelvis_list, hip_flexion_r, knee_angle_r, etc.")
            print(f"   Motion file has: {coord_names[:10]}...")

        # Clean up OpenSim objects
        del imu, imu_frame, target_body, state, model

        return sim_acc_signals, sim_gyro_signals, sim_times, True

    except Exception as e:
        print(f"❌ Error generating simulated IMU data: {e}")
        return None, None, None, False


def load_real_imu_data(real_imu_file):
    """Load real IMU data from CSV file"""
    print("\n🔄 Step 2: Loading real IMU data...")

    try:
        imu_df = pd.read_csv(real_imu_file)

        # Build case-insensitive column name map
        lower_to_actual = {c.lower(): c for c in imu_df.columns}

        def get_cols(candidates):
            # candidates is a list of expected column names (case-insensitive)
            actual = []
            for name in candidates:
                key = name.lower()
                if key in lower_to_actual:
                    actual.append(lower_to_actual[key])
                else:
                    return None
            return actual

        # Extract gyroscope data (try multiple naming conventions)
        gyro_candidates = [
            ["thigh_Gyro_X", "thigh_Gyro_Y", "thigh_Gyro_Z"],  # Standard
            ["thigh_imu_r_gyro_x", "thigh_imu_r_gyro_y", "thigh_imu_r_gyro_z"],  # Molinaro right
            ["thigh_r_gyro_x", "thigh_r_gyro_y", "thigh_r_gyro_z"],  # Molinaro processed
            ["thigh_l_gyro_x", "thigh_l_gyro_y", "thigh_l_gyro_z"],  # Molinaro left
        ]
        
        gyro_cols_actual = None
        for candidate_set in gyro_candidates:
            gyro_cols_actual = get_cols(candidate_set)
            if gyro_cols_actual is not None:
                print(f"✓ Found gyro columns: {gyro_cols_actual}")
                break
        
        if gyro_cols_actual is None:
            print(f"❌ No gyro columns found in {real_imu_file}")
            print(f"   Available columns: {list(imu_df.columns)}")
            return None, None, None, False
        
        real_gyro_signals = imu_df[gyro_cols_actual].values
        
        # Time/Header column (case-insensitive)
        header_col = lower_to_actual.get('header', lower_to_actual.get('time', None))
        real_times = imu_df[header_col].values if header_col else np.arange(len(real_gyro_signals))
        
        print(f"✓ Loaded real IMU data: {len(real_times)} frames")
        print(f"   Time range: {real_times[0]:.3f}s to {real_times[-1]:.3f}s")
        print(f"   Gyro range: {np.min(real_gyro_signals, axis=0)} to {np.max(real_gyro_signals, axis=0)} rad/s")

        # Return None for acc_signals since we only use gyro
        return None, real_gyro_signals, real_times, True

    except Exception as e:
        print(f"❌ Error loading real IMU data: {e}")
        return None, None, None, False


def plot_debug_imu_data(sim_times, sim_gyro_signals, real_times, real_gyro_signals, output_dir):
    """Plot real and simulated IMU data separately for debugging"""
    print("\n🔍 Debug: Creating separate IMU plots...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Create debug plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        axis_names = ['X', 'Y', 'Z']
        
        for i in range(3):
            # Simulated data
            axes[i].plot(sim_times, sim_gyro_signals[:, i], 'b-', label='Simulated', alpha=0.7)
            axes[i].set_title(f'Simulated Gyro {axis_names[i]}')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Angular Velocity (rad/s)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
            # Real data
            axes[i+3].plot(real_times, real_gyro_signals[:, i], 'r-', label='Real', alpha=0.7)
            axes[i+3].set_title(f'Real Gyro {axis_names[i]}')
            axes[i+3].set_xlabel('Time (s)')
            axes[i+3].set_ylabel('Angular Velocity (rad/s)')
            axes[i+3].grid(True, alpha=0.3)
            axes[i+3].legend()
        
        plt.tight_layout()
        
        # Save debug plot
        debug_plot_path = Path(output_dir) / "debug_imu_separate.png"
        plt.savefig(debug_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Debug plot saved: {debug_plot_path}")
        
        # Print statistics
        print("\n📊 IMU Data Statistics:")
        print("Simulated Gyro:")
        for i, axis in enumerate(axis_names):
            print(f"  {axis}: mean={np.mean(sim_gyro_signals[:, i]):.6f}, std={np.std(sim_gyro_signals[:, i]):.6f}")
        
        print("Real Gyro:")
        for i, axis in enumerate(axis_names):
            print(f"  {axis}: mean={np.mean(real_gyro_signals[:, i]):.6f}, std={np.std(real_gyro_signals[:, i]):.6f}")
            
    except Exception as e:
        print(f"⚠️  Debug plotting failed: {e}")


def align_time_series(sim_times, sim_gyro_signals, real_times, real_gyro_signals):
    """Align simulated and real IMU data in time"""
    print("\n🔄 Step 3: Aligning time series...")

    # Find overlapping time range
    overlap_start = max(sim_times[0], real_times[0])
    overlap_end = min(sim_times[-1], real_times[-1])

    if overlap_start >= overlap_end:
        print(f"❌ No overlapping time range found!")
        print(f"   Sim time range: {sim_times[0]:.3f}s to {sim_times[-1]:.3f}s")
        print(f"   Real time range: {real_times[0]:.3f}s to {real_times[-1]:.3f}s")
        return None, None, None, False

    # Create common time grid
    num_points = min(len(sim_times), len(real_times), 1000)
    common_times = np.linspace(overlap_start, overlap_end, num_points)

    # Interpolate both datasets to common time grid
    sim_interp = interp1d(
        sim_times,
        sim_gyro_signals,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=0,
    )
    real_interp = interp1d(
        real_times,
        real_gyro_signals,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=0,
    )

    sim_aligned = sim_interp(common_times)
    real_aligned = real_interp(common_times)

    print(f"✓ Time alignment completed")
    print(f"   Overlap range: {overlap_start:.3f}s to {overlap_end:.3f}s")
    print(f"   Aligned points: {len(common_times)}")

    return sim_aligned, real_aligned, common_times, True


def optimize_orientation(sim_aligned, real_aligned):
    """Optimize the IMU orientation using Kabsch algorithm (SVD)"""
    print("\n🔄 Step 4: Optimizing IMU orientation using Kabsch algorithm...")

    # Compute initial cost (no rotation)
    initial_cost = np.mean((sim_aligned - real_aligned) ** 2)
    print(f"   Initial cost: {initial_cost:.6f}")

    print("   Running Kabsch algorithm (SVD)...")
    
    # Apply Kabsch algorithm to find optimal rotation
    optimal_rotation_matrix, final_cost = kabsch_algorithm(sim_aligned, real_aligned)
    
    improvement = ((initial_cost - final_cost) / initial_cost) * 100

    print(f"✓ Optimization completed!")
    print(f"   Success: True (SVD always converges)")
    print(f"   Iterations: 1 (direct solution)")
    print(f"   Final cost: {final_cost:.6f}")
    print(f"   Improvement: {improvement:.2f}%")
    print(f"   Optimal rotation matrix determinant: {np.linalg.det(optimal_rotation_matrix):.6f}")

    # Convert rotation matrix to quaternion for compatibility with existing code
    optimal_quaternion = rotation_matrix_to_quaternion(optimal_rotation_matrix)
    print(f"   Optimal quaternion [w,x,y,z]: {optimal_quaternion}")

    # Print MSE for each axis of the real IMU and the simulated IMU
    rotated_sim_aligned = np.array([optimal_rotation_matrix @ sim_frame for sim_frame in sim_aligned])

    for i in range(3):
        print(f"   Initial Axis {i} MSE: {np.mean((real_aligned[:, i] - sim_aligned[:, i]) ** 2):.6f}")
        print(f"   Axis {i} MSE: {np.mean((real_aligned[:, i] - rotated_sim_aligned[:, i]) ** 2):.6f}")

    return {
        "success": True,
        "iterations": 1,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "improvement_percent": improvement,
        "optimal_quaternion": optimal_quaternion,
        "optimal_rotation_matrix": optimal_rotation_matrix,
    }


def create_visualizations(
    sim_aligned, real_aligned, common_times, optimization_result, output_dir
):
    """Create visualization plots"""
    print("\n🔄 Step 5: Creating visualizations...")

    optimal_rotation_matrix = optimization_result["optimal_rotation_matrix"]

    # Plot two sets of axes (initial and optimized) with the same origin
    origin = np.zeros(3)
    initial_axes = np.eye(3)  # 3x3 identity: columns are X, Y, Z axes
    optimized_axes = optimal_rotation_matrix @ initial_axes  # rotate each axis

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot initial axes
    colors = ["r", "g", "b"]
    labels = ["X (Initial)", "Y (Initial)", "Z (Initial)"]
    for i in range(3):
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            initial_axes[0, i],
            initial_axes[1, i],
            initial_axes[2, i],
            color=colors[i],
            linewidth=2,
            arrow_length_ratio=0.15,
            alpha=0.7,
            label=labels[i],
        )

    # Plot optimized axes
    opt_labels = ["X (Optimized)", "Y (Optimized)", "Z (Optimized)"]
    for i in range(3):
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            optimized_axes[0, i],
            optimized_axes[1, i],
            optimized_axes[2, i],
            color=colors[i],
            linewidth=2,
            arrow_length_ratio=0.15,
            alpha=0.9,
            linestyle="dashed",
            label=opt_labels[i],
        )

    # Set plot limits for better visualization
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Initial vs Optimized IMU Axes")
    # Only show unique labels in legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left", fontsize=9)

    plt.tight_layout()
    axes_plot_path = Path(output_dir) / "optimization_axes.png"
    plt.savefig(axes_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved axes plot: {axes_plot_path}")

    # Apply optimal rotation to simulated data
    rotated_sim_aligned = np.array(
        [optimal_rotation_matrix @ sim_frame for sim_frame in sim_aligned]
    )

    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    colors = ["red", "green", "blue"]
    labels = ["X", "Y", "Z"]

    # Plot individual axes comparison
    for i in range(3):
        # Before optimization
        axes[i, 0].plot(
            common_times,
            sim_aligned[:, i],
            color=colors[i],
            linewidth=1.5,
            alpha=0.7,
            label=f"Simulated {labels[i]} (Original)",
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

        axes[i, 0].set_ylabel(f"Angular Velocity {labels[i]} (rad/s)")
        axes[i, 0].set_title(f"Before Optimization - Axis {labels[i]}")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        # After optimization
        axes[i, 1].plot(
            common_times,
            rotated_sim_aligned[:, i],
            color=colors[i],
            linewidth=1.5,
            alpha=0.7,
            label=f"Simulated {labels[i]} (Optimized)",
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

        axes[i, 1].set_ylabel(f"Angular Velocity {labels[i]} (rad/s)")
        axes[i, 1].set_title(f"After Optimization - Axis {labels[i]}")
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    comparison_plot_path = Path(output_dir) / "optimization_comparison.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved comparison plot: {comparison_plot_path}")

    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Magnitude comparison
    sim_magnitude_before = np.linalg.norm(sim_aligned, axis=1)
    sim_magnitude_after = np.linalg.norm(rotated_sim_aligned, axis=1)
    real_magnitude = np.linalg.norm(real_aligned, axis=1)

    ax1.plot(
        common_times,
        sim_magnitude_before,
        "b-",
        linewidth=2,
        alpha=0.7,
        label="Simulated (Original)",
    )
    ax1.plot(
        common_times,
        sim_magnitude_after,
        "r-",
        linewidth=2,
        alpha=0.7,
        label="Simulated (Optimized)",
    )
    ax1.plot(
        common_times, real_magnitude, "k--", linewidth=2, alpha=0.9, label="Real IMU"
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angular Velocity Magnitude (rad/s)")
    ax1.set_title("Magnitude Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cost reduction visualization
    costs = [optimization_result["initial_cost"], optimization_result["final_cost"]]
    methods = ["Before\nOptimization", "After\nOptimization"]
    colors_bar = ["lightcoral", "lightgreen"]

    bars = ax2.bar(methods, costs, color=colors_bar, alpha=0.7)
    ax2.set_ylabel("Mean Squared Error")
    ax2.set_title(
        f'Optimization Results\n{optimization_result["improvement_percent"]:.1f}% Improvement'
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
    summary_plot_path = Path(output_dir) / "optimization_summary.png"
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved summary plot: {summary_plot_path}")


def save_results(
    optimization_result,
    model_file,
    motion_file,
    real_imu_file,
    segment_name,
    imu_name,
    output_dir,
):
    """Save optimization results to files"""
    print("\n💾 Saving results...")

    # Save detailed results as JSON
    results_data = {
        "configuration": {
            "model_file": str(model_file),
            "motion_file": str(motion_file),
            "real_imu_file": str(real_imu_file),
            "segment_name": segment_name,
            "imu_name": imu_name,
        },
        "optimization": {
            "optimal_quaternion": optimization_result["optimal_quaternion"].tolist(),
            "optimal_rotation_matrix": optimization_result[
                "optimal_rotation_matrix"
            ].tolist(),
            "initial_cost": float(optimization_result["initial_cost"]),
            "final_cost": float(optimization_result["final_cost"]),
            "improvement_percent": float(optimization_result["improvement_percent"]),
            "success": bool(optimization_result["success"]),
            "iterations": int(optimization_result["iterations"]),
        },
    }

    results_path = Path(output_dir) / "optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"✓ Saved results: {results_path}")

    # Save rotation matrix
    rotation_path = Path(output_dir) / "optimal_rotation_matrix.txt"
    np.savetxt(
        rotation_path,
        optimization_result["optimal_rotation_matrix"],
        header="Optimal rotation matrix for IMU orientation alignment",
    )

    print(f"✓ Saved rotation matrix: {rotation_path}")


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


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="IMU Orientation Optimization for Final Dataset")
    parser.add_argument(
        "--dataset-root", required=True, help="Path to Final dataset root (e.g., /path/to/Final/Molinaro_Phase1_Phase2)"
    )
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., AB01)")
    parser.add_argument("--condition", required=True, help="Condition (e.g., levelground, ramp, stair)")
    parser.add_argument("--trial", required=True, help="Trial name (e.g., LG_C0p0_S0p0_BT_1_10)")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument(
        "--segments",
        default="femur_r",
        help="Comma-separated list of body segments or 'all' (default: femur_r). Available: femur_r, femur_l, tibia_r, tibia_l, pelvis",
    )
    parser.add_argument(
        "--imu-name",
        default=None,
        help="Name for the IMU component (default: auto-generated from segment)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=2000,
        help="Maximum frames to process (default: 2000)",
    )
    parser.add_argument(
        "--gyro-in-degrees",
        action="store_true",
        help="Use degrees for gyroscope data (default: radians)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with separate IMU plots",
    )

    args = parser.parse_args()
    
    # Define available segments and their corresponding IMU data names
    segment_to_imu_data = {
        "femur_r": "thigh_r",
        "femur_l": "thigh_l",
        "tibia_r": "shank_r",
        "tibia_l": "shank_l",
        "pelvis": "pelvis"
    }
    
    # Parse segments
    if args.segments.lower() == "all":
        segments_to_process = list(segment_to_imu_data.keys())
    else:
        segments_to_process = [s.strip() for s in args.segments.split(",")]
        # Validate segments
        invalid_segments = [s for s in segments_to_process if s not in segment_to_imu_data]
        if invalid_segments:
            print(f"❌ Invalid segments: {invalid_segments}")
            print(f"   Available segments: {list(segment_to_imu_data.keys())}")
            return
    
    # Process each segment
    all_results = {}
    for segment in segments_to_process:
        imu_data_name = segment_to_imu_data[segment]
        imu_name = args.imu_name if args.imu_name else f"{segment}_imu"
        
        print(f"\n{'='*70}")
        print(f"🎯 Processing segment: {segment} (IMU data: {imu_data_name})")
        print(f"{'='*70}")
        
        # Create segment-specific output directory
        if len(segments_to_process) > 1:
            segment_output = Path(args.output) / segment
        else:
            segment_output = Path(args.output)
        
        args_copy = argparse.Namespace(**vars(args))
        args_copy.segment = segment
        args_copy.imu_name = imu_name
        args_copy.output = str(segment_output)
        args_copy.imu_data_name = imu_data_name
        
        result = process_single_segment(args_copy)
        if result:
            all_results[segment] = result
    
    # Save combined results if multiple segments
    if len(segments_to_process) > 1 and all_results:
        combined_output = Path(args.output) / "combined_results.json"
        combined_data = {
            "configuration": {
                "dataset_root": args.dataset_root,
                "subject": args.subject,
                "condition": args.condition,
                "trial": args.trial,
                "segments": segments_to_process
            },
            "segments": {}
        }
        
        for segment, result in all_results.items():
            combined_data["segments"][segment] = {
                "optimal_quaternion": result["optimal_quaternion"].tolist(),
                "optimal_rotation_matrix": result["optimal_rotation_matrix"].tolist(),
                "initial_cost": float(result["initial_cost"]),
                "final_cost": float(result["final_cost"]),
                "improvement_percent": float(result["improvement_percent"])
            }
        
        with open(combined_output, "w") as f:
            json.dump(combined_data, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ All segments processed successfully!")
        print(f"📊 Combined results saved to: {combined_output}")
        print(f"{'='*70}")


def process_single_segment(args):
    """Process optimization for a single segment"""

    # Create output directory
    Path(args.output).mkdir(exist_ok=True, parents=True)

    print("🚀 IMU Orientation Optimization for Final Dataset")
    print("=" * 60)
    print(f"Dataset: {args.dataset_root}")
    print(f"Subject: {args.subject}")
    print(f"Condition: {args.condition}")
    print(f"Trial: {args.trial}")
    print(f"Output: {args.output}")
    print(f"Segment: {args.segment}")
    print("=" * 60)

    try:
        # Find required files
        model_file, motion_file, real_imu_file = find_dataset_files(
            args.dataset_root, args.subject, args.condition, args.trial
        )
        
        if model_file is None:
            return None

        print(f"Model: {model_file}")
        print(f"Motion: {motion_file}")
        print(f"Real IMU: {real_imu_file}")

        # Step 1: Generate simulated IMU data
        sim_acc_signals, sim_gyro_signals, sim_times, success = generate_simulated_imu_data(
            str(model_file), str(motion_file), args.segment, args.imu_name, args.max_frames
        )
        if not success:
            return None

        # Step 2: Load real IMU data (gyro only) - use imu_data_name if provided
        imu_data_segment = getattr(args, 'imu_data_name', args.segment)
        _, real_gyro_signals, real_times, success = load_real_imu_data_for_segment(
            str(real_imu_file), imu_data_segment
        )
        if not success:
            return None

        # If real data is in degrees, convert to radians
        if args.gyro_in_degrees:
            real_gyro_signals = real_gyro_signals * np.pi / 180.0

        # Debug: Plot separate IMU data if requested
        if args.debug:
            plot_debug_imu_data(sim_times, sim_gyro_signals, real_times, real_gyro_signals, args.output)

        # Step 3: Align time series
        sim_aligned, real_aligned, common_times, success = align_time_series(
            sim_times, sim_gyro_signals, real_times, real_gyro_signals
        )

        if not success:
            return None

        # Step 4: Optimize orientation
        optimization_result = optimize_orientation(sim_aligned, real_aligned)
        if not optimization_result["success"]:
            print("❌ Optimization failed to converge")
            return None

        # Step 5: Create visualizations
        create_visualizations(
            sim_aligned, real_aligned, common_times, optimization_result, args.output
        )

        # Step 6: Save results
        save_results(
            optimization_result,
            model_file,
            motion_file,
            real_imu_file,
            args.segment,
            args.imu_name,
            args.output,
        )

        print(f"\n🎉 IMU Optimization Completed Successfully!")
        print(f"📊 Summary:")
        print(f"   Initial MSE: {optimization_result['initial_cost']:.6f}")
        print(f"   Final MSE: {optimization_result['final_cost']:.6f}")
        print(f"   Improvement: {optimization_result['improvement_percent']:.2f}%")
        print(f"   Results saved to: {args.output}")
        
        return optimization_result

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_real_imu_data_for_segment(real_imu_file, segment_name):
    """Load real IMU data for a specific segment"""
    print(f"\n🔄 Step 2: Loading real IMU data for segment: {segment_name}...")

    try:
        imu_df = pd.read_csv(real_imu_file)

        # Build case-insensitive column name map
        lower_to_actual = {c.lower(): c for c in imu_df.columns}

        def get_cols(candidates):
            # candidates is a list of expected column names (case-insensitive)
            actual = []
            for name in candidates:
                key = name.lower()
                if key in lower_to_actual:
                    actual.append(lower_to_actual[key])
                else:
                    return None
            return actual

        # Extract gyroscope data for the specific segment
        gyro_candidates = [
            [f"{segment_name}_gyro_x", f"{segment_name}_gyro_y", f"{segment_name}_gyro_z"],
            [f"{segment_name}_Gyro_X", f"{segment_name}_Gyro_Y", f"{segment_name}_Gyro_Z"],
            [f"{segment_name.upper()}_GYROX", f"{segment_name.upper()}_GYROY", f"{segment_name.upper()}_GYROZ"],
        ]
        
        gyro_cols_actual = None
        for candidate_set in gyro_candidates:
            gyro_cols_actual = get_cols(candidate_set)
            if gyro_cols_actual is not None:
                print(f"✓ Found gyro columns: {gyro_cols_actual}")
                break
        
        if gyro_cols_actual is None:
            print(f"❌ No gyro columns found for segment '{segment_name}' in {real_imu_file}")
            print(f"   Available columns: {list(imu_df.columns)}")
            return None, None, None, False
        
        real_gyro_signals = imu_df[gyro_cols_actual].values
        
        # Time/Header column (case-insensitive)
        header_col = lower_to_actual.get('header', lower_to_actual.get('time', None))
        real_times = imu_df[header_col].values if header_col else np.arange(len(real_gyro_signals))
        
        print(f"✓ Loaded real IMU data: {len(real_times)} frames")
        print(f"   Time range: {real_times[0]:.3f}s to {real_times[-1]:.3f}s")
        print(f"   Gyro range: {np.min(real_gyro_signals, axis=0)} to {np.max(real_gyro_signals, axis=0)} rad/s")

        # Return None for acc_signals since we only use gyro
        return None, real_gyro_signals, real_times, True

    except Exception as e:
        print(f"❌ Error loading real IMU data: {e}")
        return None, None, None, False


if __name__ == "__main__":
    main()
