#!/usr/bin/env python3
"""
IMU Position Optimization
------------------------
Optimizes the 3D position of IMU sensors on body segments using pre-determined orientations.
Uses the rotation matrices from orientation optimization results to find optimal IMU positions.

Key Features:
- Loads pre-computed rotation matrices from orientation optimization
- Optimizes 3D position (x, y, z) of IMU on each body segment
- Uses OpenSim's PhysicalOffsetFrame to position IMUs
- Minimizes difference between real and simulated IMU signals
- Creates visualizations showing position optimization results
"""

import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json
from pathlib import Path
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt, lfilter
from scipy.interpolate import interp1d
from tqdm import tqdm


# ---------------- Filters ---------------- #
def butter_lowpass_zero_phase(data: np.ndarray,
                              cutoff_hz: float = 6.0,
                              fs_hz: float = 200.0,
                              order: int = 4) -> np.ndarray:
    """
    Apply zero-phase Butterworth low-pass filter along time axis.
    Handles multi-axis arrays (N×D).
    """
    if data is None or data.size == 0:
        return data
    nyq = 0.5 * fs_hz
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype='low', analog=False)
    try:
        return np.column_stack([
            filtfilt(
                b, a, data[:, i],
                axis=0,
                method='pad',
                padlen=min(
                    3 * max(len(a), len(b)),
                    max(0, len(data) - 1)
                )
            ) for i in range(data.shape[1])
        ])
    except ValueError:
        # If sequence too short for filtfilt padlen, fall back to lfilter twice
        out = np.zeros_like(data)
        for i in range(data.shape[1]):
            y = lfilter(b, a, data[:, i], axis=0)
            y = lfilter(b, a, y[::-1], axis=0)[::-1]
            out[:, i] = y
        return out


# ---------------- Load Orientation Results ---------------- #
def load_orientation_results(results_dir, segment):
    """Load rotation matrix from orientation optimization results."""
    results_file = Path(results_dir) / segment / "optimization_results.json"
    if not results_file.exists():
        print(f"❌ Orientation results not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    rotation_matrix = np.array(data['result']['rotation_matrix'])
    print(f"✓ Loaded orientation results for {segment}")
    print(f"  - Improvement: {data['result']['improvement_percent']:.1f}%")
    print(f"  - Z-correlation: {data['result']['z_correlation']:.3f}")
    
    return rotation_matrix


# ---------------- Real IMU Data Loading ---------------- #
def _find_cols_ci(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        act = []
        ok = True
        for n in cand:
            if n.lower() in lower:
                act.append(lower[n.lower()])
            else:
                ok = False
                break
        if ok:
            return act
    return None


def load_real_imu_data_for_segment(csv_path, segment):
    """Load real IMU accelerometer data for a specific segment."""
    df = pd.read_csv(csv_path)
    
    # Try different column naming conventions for accelerometer data
    accel_cols = _find_cols_ci(
        df,
        [
            [f"{segment}_accel_x", f"{segment}_accel_y", f"{segment}_accel_z"],
            [f"{segment}_Accel_X", f"{segment}_Accel_Y", f"{segment}_Accel_Z"],
            [f"{segment.upper()}_ACCELX", f"{segment.upper()}_ACCELY", f"{segment.upper()}_ACCELZ"],
            ["thigh_r_accel_x", "thigh_r_accel_y", "thigh_r_accel_z"],
            ["thigh_l_accel_x", "thigh_l_accel_y", "thigh_l_accel_z"],
        ],
    )
    
    if accel_cols is None:
        print(f"❌ No accelerometer columns found for segment '{segment}'")
        return None, None, None, False
    
    accel = df[accel_cols].to_numpy()
    lower = {c.lower(): c for c in df.columns}
    tcol = lower.get("header", lower.get("time", None))
    t = df[tcol].to_numpy() if tcol else np.arange(len(accel))
    
    return accel, None, t, True


# ---------------- OpenSim Simulation with Position ---------------- #
def generate_simulated_imu_data_with_position(model_file, motion_file, segment, imu_name, 
                                            rotation_matrix, position, max_frames):
    """
    Generate simulated IMU data with specific orientation and position.
    
    Args:
        model_file: Path to OpenSim model
        motion_file: Path to motion file
        segment: Body segment name
        imu_name: Name for the IMU
        rotation_matrix: 3x3 rotation matrix for orientation
        position: 3D position [x, y, z] in meters
        max_frames: Maximum frames to process
    """
    try:
        osim.ModelVisualizer.addDirToGeometrySearchPaths(
            "/Applications/OpenSim 4.5/Geometry"
        )
        model = osim.Model(str(model_file))
        model.setUseVisualizer(False)
        
        if not model.getBodySet().contains(segment):
            print(f"❌ Segment '{segment}' not found in model")
            return None, None, None, False
        
        body = model.getBodySet().get(segment)
        
        # Create IMU
        imu = osim.IMU()
        imu.setName(imu_name)
        
        # Create PhysicalOffsetFrame with position and orientation
        frame = osim.PhysicalOffsetFrame()
        frame.setName(f"{imu_name}_frame")
        frame.setParentFrame(body)
        
        # Set position (translation)
        translation = osim.Vec3(position[0], position[1], position[2])
        
        # Convert rotation matrix to OpenSim rotation
        # OpenSim uses rotation matrix in row-major order
        rotation_osim = osim.Rotation()
        for i in range(3):
            for j in range(3):
                rotation_osim.set(i, j, rotation_matrix[i, j])
        
        # Create transform with position and rotation
        transform = osim.Transform(rotation_osim, translation)
        frame.setOffsetTransform(transform)
        
        body.addComponent(frame)
        imu.connectSocket_frame(frame)
        model.addComponent(imu)
        
        state = model.initSystem()
        table = osim.TimeSeriesTable(str(motion_file))
        tcol = table.getIndependentColumn()
        names = list(table.getColumnLabels())
        
        accs, gyros, times = [], [], []
        
        for i in tqdm(range(min(max_frames, table.getNumRows())), desc="Simulating IMU"):
            t = tcol[i]
            state.setTime(t)
            
            # Set coordinate values and speeds
            for c in model.getCoordinateSet():
                q = c.getName()
                u = q + "_u"
                if q in names:
                    c.setValue(state, table.getDependentColumn(q)[i])
                if u in names:
                    c.setSpeedValue(state, table.getDependentColumn(u)[i])
            
            model.realizeAcceleration(state)
            
            # Calculate IMU signals
            a = imu.calcAccelerometerSignal(state)
            g = imu.calcGyroscopeSignal(state)
            
            # Convert acceleration from mm/s² to m/s²
            accs.append([a.get(j) / 1000.0 for j in range(3)])
            gyros.append([g.get(j) for j in range(3)])
            times.append(t)
        
        return np.array(accs), np.array(gyros), np.array(times), True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return None, None, None, False


# ---------------- Time Series Alignment ---------------- #
def align_time_series(sim_t, sim_g, real_t, real_g, fs=200.0):
    """Align simulated and real time series to common time grid."""
    s0 = max(sim_t[0], real_t[0])
    s1 = min(sim_t[-1], real_t[-1])
    
    if s0 >= s1:
        return None, None, None, False
    
    dur = s1 - s0
    n = int(min(max(dur * fs, 1500), 100000))
    grid = np.linspace(s0, s1, n)
    
    # Interpolate to common time grid
    S = interp1d(sim_t, sim_g, axis=0, bounds_error=False, fill_value="extrapolate")(grid)
    R = interp1d(real_t, real_g, axis=0, bounds_error=False, fill_value="extrapolate")(grid)
    
    # Apply smoothing
    S = butter_lowpass_zero_phase(S, cutoff_hz=6.0, fs_hz=fs, order=4)
    R = butter_lowpass_zero_phase(R, cutoff_hz=6.0, fs_hz=fs, order=4)
    
    return S, R, grid, True


# ---------------- Position Optimization ---------------- #
def position_cost_function(position, rotation_matrix, model_file, motion_file, 
                          segment, imu_name, real_accel, real_time, max_frames, cost_type='mse'):
    """
    Cost function for position optimization.
    Returns cost between real and simulated accelerometer data.
    """
    try:
        # Generate simulated data with current position
        sim_accel, _, sim_time, success = generate_simulated_imu_data_with_position(
            model_file, motion_file, segment, imu_name, 
            rotation_matrix, position, max_frames
        )
        
        if not success:
            return 1e6  # Large cost for failed simulation
        
        # Align time series
        S, R, _, aligned = align_time_series(sim_time, sim_accel, real_time, real_accel)
        
        if not aligned:
            return 1e6  # Large cost for failed alignment
        
        if cost_type == 'mse':
            # Calculate MSE
            mse = np.mean((S - R) ** 2)
            return mse
        elif cost_type == 'correlation':
            # Calculate negative correlation (we want to maximize correlation)
            correlations = []
            for i in range(3):
                if np.std(S[:, i]) > 1e-6 and np.std(R[:, i]) > 1e-6:
                    corr = np.corrcoef(S[:, i], R[:, i])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            if len(correlations) == 0:
                return 1e6
            return -np.mean(correlations)  # Negative because we want to maximize
        elif cost_type == 'weighted_mse':
            # Weighted MSE with higher weight for Z-axis
            weights = np.array([1.0, 1.0, 2.0])  # Higher weight for Z-axis
            weighted_mse = np.mean(weights * (S - R) ** 2)
            return weighted_mse
        else:
            return np.mean((S - R) ** 2)
        
    except Exception as e:
        print(f"⚠️ Position optimization error: {e}")
        return 1e6


def get_segment_initial_position(segment):
    """Get realistic initial position based on segment type."""
    if segment in ['femur_r', 'femur_l']:
        return [0.0, -0.3, 0.05]  # Mid-femur, 5cm outward
    elif segment in ['tibia_r', 'tibia_l']:
        return [0.0, -0.2, 0.05]  # Mid-tibia
    elif segment == 'pelvis':
        return [0.0, -0.1, 0.05]  # Pelvis is shorter
    else:
        return [0.0, -0.25, 0.05]  # Default


def multi_start_optimization(rotation_matrix, model_file, motion_file, segment, 
                            real_accel, real_time, max_frames, n_starts=5, cost_type='mse'):
    """Try multiple random starting positions to avoid local minima."""
    print(f"🎯 Multi-start optimization for {segment} ({n_starts} starts)")
    
    # Define bounds
    bounds = [
        (-0.2, 0.2),   # x: ±20cm (forward/backward)
        (-0.5, 0.0),   # y: -50cm to 0cm (distal to proximal along segment)
        (-0.1, 0.3),   # z: -10cm to +30cm (left/right, mostly positive for outward placement)
    ]
    
    best_result = None
    best_cost = float('inf')
    imu_name = f"{segment}_imu"
    
    for i in range(n_starts):
        print(f"  Start {i+1}/{n_starts}...")
        
        # Random initial position within bounds
        initial_pos = [
            np.random.uniform(bounds[0][0], bounds[0][1]),    # X
            np.random.uniform(bounds[1][0], bounds[1][1]),    # Y  
            np.random.uniform(bounds[2][0], bounds[2][1])     # Z
        ]
        
        try:
            result = minimize(
                position_cost_function,
                initial_pos,
                args=(rotation_matrix, model_file, motion_file, segment, imu_name, 
                      real_accel, real_time, max_frames, cost_type),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'disp': False}
            )
            
            if result.success and result.fun < best_cost:
                best_cost = result.fun
                best_result = result
                print(f"    ✓ New best cost: {result.fun:.6f}")
            else:
                print(f"    - Cost: {result.fun:.6f}")
                
        except Exception as e:
            print(f"    ❌ Start {i+1} failed: {e}")
            continue
    
    if best_result is not None:
        print(f"✓ Multi-start optimization successful")
        print(f"  Best position: {best_result.x}")
        print(f"  Best cost: {best_cost:.6f}")
        return best_result.x, best_cost
    else:
        print(f"❌ All multi-start attempts failed")
        # Fall back to default position
        default_pos = get_segment_initial_position(segment)
        return default_pos, 1e6


def optimize_imu_position(rotation_matrix, model_file, motion_file, segment, 
                         real_accel, real_time, max_frames, initial_position=None, 
                         use_multi_start=True, cost_type='mse'):
    """
    Optimize IMU position using pre-determined orientation.
    
    Args:
        rotation_matrix: 3x3 rotation matrix from orientation optimization
        model_file: Path to OpenSim model
        motion_file: Path to motion file
        segment: Body segment name
        real_accel: Real accelerometer data
        real_time: Real time data
        max_frames: Maximum frames to process
        initial_position: Initial position guess [x, y, z] in meters
        use_multi_start: Whether to use multi-start optimization
        cost_type: Cost function type ('mse', 'correlation', 'weighted_mse')
    
    Returns:
        optimal_position: Optimized 3D position
        cost: Final cost value
    """
    if use_multi_start:
        return multi_start_optimization(rotation_matrix, model_file, motion_file, 
                                      segment, real_accel, real_time, max_frames, 
                                      n_starts=5, cost_type=cost_type)
    
    # Single-start optimization (original method)
    if initial_position is None:
        initial_position = get_segment_initial_position(segment)
    
    imu_name = f"{segment}_imu"
    
    print(f"🎯 Single-start optimization for {segment}")
    print(f"  Initial position: {initial_position}")
    print(f"  Cost type: {cost_type}")
    
    # Define bounds for position optimization (in meters)
    bounds = [
        (-0.2, 0.2),   # x: ±20cm (forward/backward)
        (-0.5, 0.0),   # y: -50cm to 0cm (distal to proximal along segment)
        (-0.1, 0.3),   # z: -10cm to +30cm (left/right, mostly positive for outward placement)
    ]
    
    # Optimize position
    result = minimize(
        position_cost_function,
        initial_position,
        args=(rotation_matrix, model_file, motion_file, segment, imu_name, 
              real_accel, real_time, max_frames, cost_type),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 50, 'disp': True}
    )
    
    if result.success:
        print(f"✓ Position optimization successful")
        print(f"  Optimal position: {result.x}")
        print(f"  Final cost: {result.fun:.6f}")
    else:
        print(f"❌ Position optimization failed: {result.message}")
        result.x = initial_position  # Fall back to initial position
    
    return result.x, result.fun


# ---------------- Visualization ---------------- #
def create_position_visualizations(real_accel, sim_accel, time, segment, outdir, 
                                 initial_position, optimal_position, cost):
    """Create visualizations for position optimization results."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Time series comparison
    total = time[-1] - time[0]
    if total > 22:
        mid = 0.5 * (time[0] + time[-1])
        t0, t1 = mid - 10, mid + 10
    else:
        t0, t1 = time[0], time[-1]
    
    s = np.argmin(np.abs(time - t0))
    e = np.argmin(np.abs(time - t1))
    tt = time[s:e]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    colors = ['red', 'green', 'blue']
    labels = ['X', 'Y', 'Z']
    
    for i in range(3):
        ax = axes[i]
        ax.plot(tt, real_accel[s:e, i], color=colors[i], alpha=0.7, 
                label='Real IMU', linewidth=2)
        ax.plot(tt, sim_accel[s:e, i], color=colors[i], linestyle='--', alpha=0.9, 
                label='Simulated IMU', linewidth=2)
        ax.set_title(f'{labels[i]}-axis Accelerometer Data')
        ax.set_ylabel('Linear Acceleration (m/s²)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'IMU Position Optimization: {segment.upper()}\n'
                f'Initial: {initial_position} → Optimal: {optimal_position}\n'
                f'Final Cost: {cost:.6f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    p = outdir / f"position_comparison_{segment}.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved position comparison plot: {p}")
    
    # 3D position visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot initial and optimal positions
    ax.scatter(*initial_position, color='red', s=200, label='Initial Position', alpha=0.8)
    ax.scatter(*optimal_position, color='green', s=200, label='Optimal Position', alpha=0.8)
    
    # Draw line between positions
    ax.plot([initial_position[0], optimal_position[0]], 
            [initial_position[1], optimal_position[1]], 
            [initial_position[2], optimal_position[2]], 
            'k--', alpha=0.5, linewidth=2)
    
    # Set equal aspect ratio
    max_range = max(np.abs(optimal_position).max(), np.abs(initial_position).max()) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    ax.set_xlabel('X (Forward) [m]')
    ax.set_ylabel('Y (Up) [m]')
    ax.set_zlabel('Z (Right) [m]')
    ax.set_title(f'IMU Position Optimization: {segment.upper()}\n'
                f'Movement: {np.linalg.norm(np.array(optimal_position) - np.array(initial_position)):.3f}m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    p = outdir / f"position_3d_{segment}.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved 3D position plot: {p}")


# ---------------- Save Results ---------------- #
def save_position_results(optimal_position, cost, initial_position, segment, outdir):
    """Save position optimization results."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Calculate position change
    position_change = np.array(optimal_position) - np.array(initial_position)
    distance_moved = np.linalg.norm(position_change)
    
    result = {
        "segment": segment,
        "initial_position": list(initial_position),
        "optimal_position": list(optimal_position),
        "position_change": position_change.tolist(),
        "distance_moved_m": float(distance_moved),
        "final_cost": float(cost),
        "optimization_success": True
    }
    
    with open(outdir / f"position_results_{segment}.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Saved position results: {outdir}/position_results_{segment}.json")


# ---------------- Main Processing ---------------- #
def find_dataset_files(dataset_root, subject, condition, trial):
    """Find required dataset files."""
    root = Path(dataset_root)
    model = root / subject / "opensim" / f"{subject}.osim"
    motion = root / subject / condition / trial / "opensim" / "motion.sto"
    imu = root / subject / condition / trial / "Input" / "imu_data.csv"
    
    if not (model.exists() and motion.exists() and imu.exists()):
        print("❌ Missing dataset files.")
        return None, None, None
    return model, motion, imu


def process_segment_position(args, segment, imu_csv, model_file, motion_file, orientation_results_dir):
    """Process position optimization for a single segment."""
    seg_out = Path(args.output) / f"{segment}_position"
    seg_out.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 Processing position optimization for: {segment}")
    
    # Load orientation results
    rotation_matrix = load_orientation_results(orientation_results_dir, segment)
    if rotation_matrix is None:
        return
    
    # Load real IMU data
    real_accel, _, real_time, ok = load_real_imu_data_for_segment(imu_csv, segment)
    if not ok:
        return
    
    # Set initial position based on segment type
    initial_position = get_segment_initial_position(segment)
    
    # Optimize position with user-specified options
    if args.multi_start:
        optimal_position, cost = multi_start_optimization(
            rotation_matrix, model_file, motion_file, segment,
            real_accel, real_time, args.max_frames, 
            n_starts=args.n_starts, cost_type=args.cost_type
        )
    else:
        optimal_position, cost = optimize_imu_position(
            rotation_matrix, model_file, motion_file, segment,
            real_accel, real_time, args.max_frames, initial_position,
            use_multi_start=False, cost_type=args.cost_type
        )
    
    # Generate final simulated data with optimal position
    sim_accel, _, sim_time, success = generate_simulated_imu_data_with_position(
        model_file, motion_file, segment, f"{segment}_imu",
        rotation_matrix, optimal_position, args.max_frames
    )
    
    if success:
        # Align time series for visualization
        S, R, t, aligned = align_time_series(sim_time, sim_accel, real_time, real_accel)
        if aligned:
            create_position_visualizations(R, S, t, segment, seg_out, 
                                         initial_position, optimal_position, cost)
    
    # Save results
    save_position_results(optimal_position, cost, initial_position, segment, seg_out)
    print(f"✓ Completed position optimization for {segment}")


# ---------------- CLI ---------------- #
def main():
    parser = argparse.ArgumentParser(description="IMU Position Optimization")
    parser.add_argument("--dataset-root", required=True, help="Path to dataset root")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--condition", required=True, help="Condition name")
    parser.add_argument("--trial", required=True, help="Trial name")
    parser.add_argument("--segments", default="femur_r", help="Comma-separated segments or 'all'")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--orientation-results", required=True, help="Path to orientation optimization results")
    parser.add_argument("--max-frames", type=int, default=2000, help="Maximum frames to process")
    parser.add_argument("--cost-type", default="correlation", choices=["mse", "correlation", "weighted_mse"], 
                       help="Cost function type for optimization")
    parser.add_argument("--multi-start", action="store_true", default=True, 
                       help="Use multi-start optimization (default: True)")
    parser.add_argument("--n-starts", type=int, default=5, 
                       help="Number of random starts for multi-start optimization")
    
    args = parser.parse_args()
    
    # Find dataset files
    model_file, motion_file, imu_csv = find_dataset_files(
        args.dataset_root, args.subject, args.condition, args.trial
    )
    if model_file is None:
        return
    
    # Determine segments to process
    if args.segments.lower() == "all":
        # Try to detect available segments
        try:
            df = pd.read_csv(imu_csv)
            available_segments = []
            for seg in ["pelvis", "femur_r", "femur_l", "tibia_r", "tibia_l"]:
                if any(f"{seg}_gyro_" in col.lower() for col in df.columns):
                    available_segments.append(seg)
            segments = available_segments
        except:
            segments = ["femur_r"]  # Default fallback
    else:
        segments = [s.strip() for s in args.segments.split(",")]
    
    if not segments:
        print("❌ No valid segments found.")
        return
    
    print(f"🎯 Processing segments: {segments}")
    
    # Process each segment
    for segment in segments:
        process_segment_position(args, segment, imu_csv, model_file, motion_file, 
                                args.orientation_results)


if __name__ == "__main__":
    main()
