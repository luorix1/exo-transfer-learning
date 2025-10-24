#!/usr/bin/env python3
"""
Plane-Aware Robust IMU Orientation Optimization (with plots)
------------------------------------------------------------
- Saves under output/<segment>/
- Auto-detects valid IMU segments
- Disambiguates z-axis sign after PCA
- Includes before/after and 3D frame visualization
"""

import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse, json
from pathlib import Path
from numpy.linalg import svd
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



# ---------------- Geometry utils ---------------- #
def dominant_axis(X):
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = svd(Xc, full_matrices=False)
    return Vt[0] / np.linalg.norm(Vt[0])


def orthonormal_basis_from_z(z):
    z = z / np.linalg.norm(z)
    tmp = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def weighted_2d_procrustes(A, B, w):
    w = np.clip(w, 0, None)
    if w.sum() < 1e-12:
        return np.eye(2)
    w = w / (w.sum() + 1e-12)
    Ac = A - (w[:, None] * A).sum(0, keepdims=True)
    Bc = B - (w[:, None] * B).sum(0, keepdims=True)
    H = (Ac * w[:, None]).T @ Bc
    U, _, Vt = svd(H)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:
        Vt[-1, :] *= -1
        R2 = Vt.T @ U.T
    return R2


# ---------------- Simulation ---------------- #
def generate_simulated_imu_data(model_file, motion_file, segment, imu_name, max_frames):
    try:
        osim.ModelVisualizer.addDirToGeometrySearchPaths(
            "/Applications/OpenSim 4.5/Geometry"
        )
        model = osim.Model(str(model_file))  # ensure string path
        model.setUseVisualizer(False)
        if not model.getBodySet().contains(segment):
            print(f"❌ Segment '{segment}' not found in model")
            return None, None, None, False
        body = model.getBodySet().get(segment)
        imu = osim.IMU()
        imu.setName(imu_name)
        frame = osim.PhysicalOffsetFrame()
        frame.setName(f"{imu_name}_frame")
        frame.setParentFrame(body)
        frame.setOffsetTransform(osim.Transform())
        body.addComponent(frame)
        imu.connectSocket_frame(frame)
        model.addComponent(imu)
        state = model.initSystem()
        table = osim.TimeSeriesTable(str(motion_file))
        tcol = table.getIndependentColumn()
        names = list(table.getColumnLabels())
        accs, gyros, times = [], [], []
        for i in tqdm(range(min(max_frames, table.getNumRows()))):
            t = tcol[i]
            state.setTime(t)
            for c in model.getCoordinateSet():
                q = c.getName()
                u = q + "_u"
                if q in names:
                    c.setValue(state, table.getDependentColumn(q)[i])
                if u in names:
                    c.setSpeedValue(state, table.getDependentColumn(u)[i])
            model.realizeAcceleration(state)
            a = imu.calcAccelerometerSignal(state)
            g = imu.calcGyroscopeSignal(state)
            accs.append([a.get(j) for j in range(3)])
            gyros.append([g.get(j) for j in range(3)])
            times.append(t)
        return np.array(accs), np.array(gyros), np.array(times), True
    except Exception as e:
        print(f"❌ Sim failed: {e}")
        return None, None, None, False


# ---------------- Real IMU ---------------- #
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


def load_real_imu_data_for_segment(csv_path, seg):
    df = pd.read_csv(csv_path)
    gyro_cols = _find_cols_ci(
        df,
        [
            [f"{seg}_gyro_x", f"{seg}_gyro_y", f"{seg}_gyro_z"],
            [f"{seg}_Gyro_X", f"{seg}_Gyro_Y", f"{seg}_Gyro_Z"],
            [f"{seg.upper()}_GYROX", f"{seg.upper()}_GYROY", f"{seg.upper()}_GYROZ"],
            ["thigh_r_gyro_x", "thigh_r_gyro_y", "thigh_r_gyro_z"],
            ["thigh_l_gyro_x", "thigh_l_gyro_y", "thigh_l_gyro_z"],
        ],
    )
    if gyro_cols is None:
        return None, None, None, False
    gyro = df[gyro_cols].to_numpy()
    lower = {c.lower(): c for c in df.columns}
    tcol = lower.get("header", lower.get("time", None))
    t = df[tcol].to_numpy() if tcol else np.arange(len(gyro))
    return None, gyro, t, True


def check_available_segments_in_imu_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        lower = {c.lower(): c for c in df.columns}
        segs = []
        for seg in [
            "pelvis",
            "femur_r",
            "femur_l",
            "tibia_r",
            "tibia_l",
            "thigh_r",
            "thigh_l",
        ]:
            pats = [
                [f"{seg}_gyro_x", f"{seg}_gyro_y", f"{seg}_gyro_z"],
                [f"{seg}_Gyro_X", f"{seg}_Gyro_Y", f"{seg}_Gyro_Z"],
                [
                    f"{seg.upper()}_GYROX",
                    f"{seg.upper()}_GYROY",
                    f"{seg.upper()}_GYROZ",
                ],
            ]
            for pat in pats:
                if all(p.lower() in lower for p in pat):
                    segs.append(seg)
                    break
        print(f"✓ Segments present in IMU data: {segs}")
        return segs
    except Exception as e:
        print(f"❌ Segment scan failed: {e}")
        return []


# ---------------- Alignment ---------------- #
def align_time_series(sim_t, sim_g, real_t, real_g, fs=200.0):
    s0 = max(sim_t[0], real_t[0])
    s1 = min(sim_t[-1], real_t[-1])
    if s0 >= s1:
        return None, None, None, False
    dur = s1 - s0
    n = int(min(max(dur * fs, 1500), 1000000))
    grid = np.linspace(s0, s1, n)
    S = interp1d(sim_t, sim_g, axis=0, bounds_error=False, fill_value="extrapolate")(
        grid
    )
    R = interp1d(real_t, real_g, axis=0, bounds_error=False, fill_value="extrapolate")(
        grid
    )
    
    # Apply zero-phase Butterworth smoothing
    S = butter_lowpass_zero_phase(S, cutoff_hz=6.0, fs_hz=fs, order=4)
    R = butter_lowpass_zero_phase(R, cutoff_hz=6.0, fs_hz=fs, order=4)

    return S, R, grid, True


# ---------------- Orientation Solver ---------------- #
def robust_plane_aware_orientation(real_g, sim_g):
    """Estimate rotation real->sim with z-axis sign check."""
    rg = real_g.copy()
    sg = sim_g.copy()
    zr = dominant_axis(rg)
    zs = dominant_axis(sg)

    # Test all sign combinations
    def mse_z(zr_sign, zs_sign):
        Rr = orthonormal_basis_from_z(zr_sign * zr)
        Rs = orthonormal_basis_from_z(zs_sign * zs)
        rg_loc = rg @ Rr
        sg_loc = sg @ Rs
        return np.mean((rg_loc[:, 2] - sg_loc[:, 2]) ** 2), Rr, Rs

    _, Rr, Rs = min([mse_z(a, b) for a in [1, -1] for b in [1, -1]], key=lambda x: x[0])
    rg_loc = rg @ Rr
    sg_loc = sg @ Rs
    w = np.abs(rg_loc[:, 2]) + np.abs(sg_loc[:, 2])
    w /= w.max() + 1e-12
    R2 = weighted_2d_procrustes(rg_loc[:, :2], sg_loc[:, :2], w)
    Rloc = np.eye(3)
    Rloc[:2, :2] = R2
    return Rs @ Rloc @ Rr.T


# ---------------- Visualization ---------------- #
def create_visualizations(Rg, Sg, t, R, outdir, seg):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rot = Rg @ R.T
    total = t[-1] - t[0]
    if total > 22:
        mid = 0.5 * (t[0] + t[-1])
        t0, t1 = mid - 10, mid + 10
    else:
        t0, t1 = t[0], t[-1]
    s = np.argmin(np.abs(t - t0))
    e = np.argmin(np.abs(t - t1))
    tt = t[s:e]
    labs = ["X", "Y", "Z"]
    cols = ["r", "g", "b"]
    fig, ax = plt.subplots(3, 2, figsize=(16, 11))
    for i in range(3):
        ax[i, 0].plot(tt, Rg[s:e, i], cols[i], alpha=0.7, label="Real (orig)")
        ax[i, 0].plot(tt, Sg[s:e, i], cols[i], ls="--", alpha=0.9, label="Sim (target)")
        ax[i, 0].set_title(f"Before - {labs[i]}")
        ax[i, 0].grid(True)
        ax[i, 0].legend()
        ax[i, 1].plot(tt, rot[s:e, i], cols[i], alpha=0.7, label="Real (rotated)")
        ax[i, 1].plot(tt, Sg[s:e, i], cols[i], ls="--", alpha=0.9, label="Sim (target)")
        ax[i, 1].set_title(f"After - {labs[i]}")
        ax[i, 1].grid(True)
        ax[i, 1].legend()
    ax[-1, 0].set_xlabel("Time (s)")
    ax[-1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    p = outdir / f"comparison_{seg}.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved plot: {p}")

    # Enhanced 3D axes visualization
    origin = np.zeros(3)
    I = np.eye(3)
    IMU = R @ I
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # Define colors and labels with better descriptions
    colors = ["#FF4444", "#44AA44", "#4444FF"]  # Red, Green, Blue
    axis_names = ["X (Medial-Lateral)", "Y (Anterior-Posterior)", "Z (Superior-Inferior)"]
    osim_labels = ["OpenSim X", "OpenSim Y", "OpenSim Z"]
    imu_labels = ["IMU X", "IMU Y", "IMU Z"]
    
    # Plot OpenSim axes (dashed, thicker)
    for i in range(3):
        ax.quiver(
            *origin,
            *I[:, i],
            color=colors[i],
            alpha=0.6,
            linestyle="--",
            linewidth=3,
            label=osim_labels[i],
            arrow_length_ratio=0.1
        )
    
    # Plot IMU axes (solid, thicker)
    for i in range(3):
        ax.quiver(
            *origin,
            *IMU[:, i],
            color=colors[i],
            alpha=0.9,
            linestyle="-",
            linewidth=4,
            label=imu_labels[i],
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
    ax.set_title(f"IMU Orientation Optimization: {seg.upper()}\n"
                f"OpenSim Canonical Frame (dashed) vs Optimized IMU Frame (solid)", 
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
    p = outdir / f"axes_{seg}.png"
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"✓ Saved enhanced axes plot: {p}")
    
    # Create a 2D projection comparison for better understanding
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
                    ax.arrow(0, 0, I[axis_idx, i], I[axis_idx, j], 
                            head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                            alpha=0.6, linestyle='--', linewidth=2, 
                            label=f"OpenSim {['X', 'Y', 'Z'][axis_idx]}")
                elif axis_idx == 2:  # Z axis - should point left (negative horizontal)
                    ax.arrow(0, 0, -I[axis_idx, i], I[axis_idx, j], 
                            head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                            alpha=0.6, linestyle='--', linewidth=2, 
                            label=f"OpenSim {['X', 'Y', 'Z'][axis_idx]}")
                else:  # X axis
                    ax.arrow(0, 0, I[axis_idx, i], I[axis_idx, j], 
                            head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                            alpha=0.6, linestyle='--', linewidth=2, 
                            label=f"OpenSim {['X', 'Y', 'Z'][axis_idx]}")
            
            # Plot IMU axes (solid lines) - same transformation
            for axis_idx in range(3):
                if axis_idx == 1:  # Y axis - should point up
                    ax.arrow(0, 0, IMU[axis_idx, i], IMU[axis_idx, j], 
                            head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                            alpha=0.9, linestyle='-', linewidth=3, 
                            label=f"IMU {['X', 'Y', 'Z'][axis_idx]}")
                elif axis_idx == 2:  # Z axis - should point left (negative horizontal)
                    ax.arrow(0, 0, -IMU[axis_idx, i], IMU[axis_idx, j], 
                            head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                            alpha=0.9, linestyle='-', linewidth=3, 
                            label=f"IMU {['X', 'Y', 'Z'][axis_idx]}")
                else:  # X axis
                    ax.arrow(0, 0, IMU[axis_idx, i], IMU[axis_idx, j], 
                            head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                            alpha=0.9, linestyle='-', linewidth=3, 
                            label=f"IMU {['X', 'Y', 'Z'][axis_idx]}")
        else:
            # Plot OpenSim axes (dashed lines) - normal projection
            for axis_idx in range(3):
                ax.arrow(0, 0, I[axis_idx, i], I[axis_idx, j], 
                        head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                        alpha=0.6, linestyle='--', linewidth=2, 
                        label=f"OpenSim {['X', 'Y', 'Z'][axis_idx]}")
            
            # Plot IMU axes (solid lines) - normal projection
            for axis_idx in range(3):
                ax.arrow(0, 0, IMU[axis_idx, i], IMU[axis_idx, j], 
                        head_width=0.05, head_length=0.1, fc=colors[axis_idx], ec=colors[axis_idx],
                        alpha=0.9, linestyle='-', linewidth=3, 
                        label=f"IMU {['X', 'Y', 'Z'][axis_idx]}")
        
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
    
    plt.suptitle(f"IMU Orientation Projections: {seg.upper()}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = outdir / f"projections_{seg}.png"
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"✓ Saved projection plots: {p}")


# ---------------- Save ---------------- #
def save_results(R, Rg, Sg, outdir, meta):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rot = Rg @ R.T
    resid0 = Rg - Sg
    resid1 = rot - Sg
    initial = np.mean(np.sum(resid0 * resid0, 1))
    final = np.mean(np.sum(resid1 * resid1, 1))
    improve = 100 * (initial - final) / (initial + 1e-12)
    zcorr = np.corrcoef(rot[:, 2], Sg[:, 2])[0, 1]
    result = {
        "rotation_matrix": R.tolist(),
        "initial_cost": float(initial),
        "final_cost": float(final),
        "improvement_percent": float(improve),
        "z_correlation": float(zcorr),
    }
    with open(outdir / "optimization_results.json", "w") as f:
        json.dump({"meta": meta, "result": result}, f, indent=2)
    print(f"✓ Saved results to {outdir/'optimization_results.json'}")


# ---------------- Dataset + Process ---------------- #
def find_dataset_files(dataset_root, subject, condition, trial):
    root = Path(dataset_root)
    model = root / subject / "opensim" / f"{subject}.osim"
    motion = root / subject / condition / trial / "opensim" / "motion.sto"
    imu = root / subject / condition / trial / "Input" / "imu_data.csv"
    if not (model.exists() and motion.exists() and imu.exists()):
        print("❌ Missing dataset files.")
        return None, None, None
    return model, motion, imu


def process_segment(args, seg, imu_csv, model_file, motion_file):
    seg_out = Path(args.output) / seg
    seg_out.mkdir(parents=True, exist_ok=True)
    print(f"\n🎯 Segment: {seg}")
    _, real_g, real_t, ok = load_real_imu_data_for_segment(imu_csv, seg)
    if not ok:
        return
    if args.gyro_in_degrees:
        real_g = np.deg2rad(real_g)
    sim_a, sim_g, sim_t, ok = generate_simulated_imu_data(
        model_file, motion_file, seg, f"{seg}_imu", args.max_frames
    )
    if not ok:
        return
    Sg, Rg, t, ok = align_time_series(sim_t, sim_g, real_t, real_g)
    if not ok:
        return
    R = robust_plane_aware_orientation(Rg, Sg)
    create_visualizations(Rg, Sg, t, R, seg_out, seg)
    save_results(R, Rg, Sg, seg_out, {"segment": seg})
    print(f"✓ Completed {seg}")


# ---------------- CLI ---------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--trial", required=True)
    p.add_argument("--segments", default="femur_r")
    p.add_argument("--output", required=True)
    p.add_argument("--max-frames", type=int, default=2000)
    p.add_argument("--gyro-in-degrees", action="store_true")
    args = p.parse_args()
    model_file, motion_file, imu_csv = find_dataset_files(
        args.dataset_root, args.subject, args.condition, args.trial
    )
    if model_file is None:
        return
    available = check_available_segments_in_imu_data(imu_csv)
    if args.segments.lower() == "all":
        segs = available
    else:
        req = [s.strip() for s in args.segments.split(",")]
        segs = [s for s in req if s in available]
    if not segs:
        print("❌ No valid segments found.")
        return
    for seg in segs:
        process_segment(args, seg, imu_csv, model_file, motion_file)


if __name__ == "__main__":
    main()
