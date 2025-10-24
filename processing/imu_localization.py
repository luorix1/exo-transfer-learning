#!/usr/bin/env python3
"""
IMU Localization (6-DoF): Orientation + Position

Extends canonical_frame_converter orientation fitting by additionally searching for
the IMU 3-DoF position relative to each segment using accelerometer data.

Approach
- Orientation: fit 3x3 rotation (Kabsch) using gyroscope triplets, like canonical_frame_converter.
- Position: grid-search a cylindrical workspace per segment; for each candidate translation
  relative to the body frame, simulate linear acceleration and minimize MSE to the real (rotated) accelerometer.

Notes
- Uses OpenSim IMU components attached via a PhysicalOffsetFrame with pure translation relative to body.
- Cylindrical axes are heuristically mapped per segment (longitudinal axis):
  femur/tibia: local Y; foot: local X; pelvis: local Z.
- Radius/axial search bounds chosen conservatively; override via CLI if needed.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import opensim as osim
import pandas as pd

# Reuse helpers from canonical_frame_converter
from processing.canonical_frame_converter import (
    read_imu_csv,
    parse_imu_columns,
    load_model_and_motion,
    fit_rotation,
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def simulate_signals_for_offset(
    model: osim.Model,
    base_state: osim.State,
    table: osim.TimeSeriesTable,
    time_col,
    coord_labels: List[str],
    segment_name: str,
    imu_name: str,
    offset_xyz: Tuple[float, float, float],
    max_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate gyro and linear accel at an offset relative to a segment body.

    Returns (t, gyro[Nx3], accel[Nx3])
    """
    # Clone model to avoid accumulating components
    mdl = osim.Model(model)
    mdl.setUseVisualizer(False)
    state = mdl.initSystem()

    # Resolve body
    body = None
    names = [mdl.getBodySet().get(i).getName() for i in range(mdl.getBodySet().getSize())]
    for n in names:
        if n.lower() == segment_name.lower():
            body = mdl.getBodySet().get(n)
            break
    if body is None:
        # fallback: substring
        for n in names:
            if segment_name.replace("_r",""").replace("_l",""").split("_")[0] in n.lower():
                body = mdl.getBodySet().get(n)
                break
    if body is None:
        raise RuntimeError(f"Segment not found in model clone: {segment_name}")

    # Create offset frame and IMU
    frame = osim.PhysicalOffsetFrame()
    frame.setName(f"{imu_name}_frame")
    frame.setParentFrame(body)
    T = osim.Transform()
    T.setP(osim.Vec3(*[float(v) for v in offset_xyz]))  # pure translation
    frame.setOffsetTransform(T)
    body.addComponent(frame)
    imu = osim.IMU()
    imu.setName(imu_name)
    imu.connectSocket_frame(frame)
    mdl.addComponent(imu)
    _ = mdl.initSystem()

    num_frames = min(max_frames, table.getNumRows())
    gyro_list: List[np.ndarray] = []
    acc_list: List[np.ndarray] = []
    t_list: List[float] = []

    for i in range(num_frames):
        t = time_col[i]
        state.setTime(t)
        # set coordinates and speeds
        for coord in mdl.getCoordinateSet():
            q = coord.getName()
            u = q + "_u"
            if q in coord_labels:
                coord.setValue(state, table.getDependentColumn(q)[i])
            if u in coord_labels:
                coord.setSpeedValue(state, table.getDependentColumn(u)[i])
        mdl.realizeAcceleration(state)
        g = imu.calcGyroscopeSignal(state)
        a = imu.calcLinearAcceleration(state)
        gyro_list.append(np.array([g.get(0), g.get(1), g.get(2)], dtype=float))
        acc_list.append(np.array([a.get(0), a.get(1), a.get(2)], dtype=float))
        t_list.append(float(t))

    return np.asarray(t_list), np.vstack(gyro_list), np.vstack(acc_list)


def axis_basis(segment: str) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Return (axis_key, ex, ey, ez) body-frame unit vectors and which axis is longitudinal.

    We assume body frame axes are close to OpenSim defaults; this is heuristic.
    Longitudinal axis key: 'x'|'y'|'z'.
    """
    seg = segment.lower()
    if any(k in seg for k in ["femur", "tibia", "thigh", "shank"]):
        return "y", np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    if "foot" in seg:
        return "x", np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    # pelvis or others
    return "z", np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])


def cylindrical_offsets(
    segment: str,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    r_steps: int,
    theta_steps: int,
    z_steps: int,
) -> List[Tuple[float, float, float]]:
    axis_key, ex, ey, ez = axis_basis(segment)
    # radial plane basis
    if axis_key == "x":
        u1, u2, ax = ey, ez, ex
    elif axis_key == "y":
        u1, u2, ax = ex, ez, ey
    else:
        u1, u2, ax = ex, ey, ez

    rs = np.linspace(r_min, r_max, max(1, r_steps))
    thetas = np.linspace(0.0, 2.0 * np.pi, max(1, theta_steps), endpoint=False)
    zs = np.linspace(z_min, z_max, max(1, z_steps))
    offsets: List[Tuple[float, float, float]] = []
    for r in rs:
        for th in thetas:
            radial = r * (np.cos(th) * u1 + np.sin(th) * u2)
            for z in zs:
                off = radial + z * ax
                offsets.append((float(off[0]), float(off[1]), float(off[2])))
    return offsets


def default_bounds_for_segment(segment: str) -> Tuple[float, float, float, float]:
    s = segment.lower()
    if any(k in s for k in ["femur", "thigh"]):
        return 0.03, 0.06, -0.20, 0.20
    if any(k in s for k in ["tibia", "shank"]):
        return 0.025, 0.05, -0.20, 0.20
    if "foot" in s:
        return 0.02, 0.05, -0.15, 0.15
    # pelvis and others
    return 0.04, 0.10, -0.15, 0.15


def main():
    parser = argparse.ArgumentParser(description="IMU localization: fit orientation (gyro) and position (accel)")
    parser.add_argument("--model", required=True, help="Path to OpenSim model (.osim)")
    parser.add_argument("--motion", required=True, help="Path to motion (.sto with states)")
    parser.add_argument("--imu", required=True, help="Path to real IMU CSV (with gyro+acc)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--segments", help="Comma-separated segments to include (default: auto from CSV)")
    parser.add_argument("--max-frames", type=int, default=1500)
    parser.add_argument("--unit", choices=["rad","deg"], default="rad", help="Gyro unit in IMU CSV")
    parser.add_argument("--unilateral", action="store_true", help="Assume generic columns map to right side")
    parser.add_argument("--r-steps", type=int, default=3)
    parser.add_argument("--theta-steps", type=int, default=12)
    parser.add_argument("--z-steps", type=int, default=5)
    parser.add_argument("--radius-min", type=float, help="Override min radius (m) for all segments")
    parser.add_argument("--radius-max", type=float, help="Override max radius (m) for all segments")
    parser.add_argument("--z-min", type=float, help="Override min axial offset (m) for all segments")
    parser.add_argument("--z-max", type=float, help="Override max axial offset (m) for all segments")
    parser.add_argument("--visualize", action="store_true", help="Save comparison plots for best pose per segment")
    parser.add_argument("--save-marked-model", type=str, default=None, help="Optional path to save a model with IMU markers for the best poses (.osim)")

    args = parser.parse_args()

    model_path = Path(args.model)
    motion_path = Path(args.motion)
    imu_path = Path(args.imu)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load real IMU CSV and mapping
    real_df = read_imu_csv(imu_path)
    mapping = parse_imu_columns(real_df, unilateral=args.unilateral)
    if args.segments:
        wanted = set(s.strip() for s in args.segments.split(",") if s.strip())
        mapping = {k: v for k, v in mapping.items() if k in wanted}
    if not mapping:
        raise RuntimeError("No segments with full gyro triplets found in IMU CSV")

    # Convert gyro units if needed
    if args.unit == "deg":
        for seg, m in mapping.items():
            gcols = m["gyro"]
            real_df[gcols] = np.deg2rad(real_df[gcols].astype(float))

    # Accel columns (optional but recommended)
    # Try to discover accelerometer columns by replacing 'gyro' with 'acc' in names present
    seg_to_acc_cols: Dict[str, List[str]] = {}
    lower_cols = {c.lower(): c for c in real_df.columns}
    for seg, m in mapping.items():
        acc_candidates = []
        for axis, suf in zip(["x","y","z"], ["_x","_y","_z"]):
            found = None
            for gyro_name in m["gyro"]:
                cand = gyro_name.lower().replace("gyro", "acc").replace("gyroscope", "acc")
                if cand in lower_cols:
                    found = lower_cols[cand]
                    break
            if found is None:
                # naive fallback: construct pattern like f"{seg}_acc_{axis}" or similar not robust; skip
                found = None
            acc_candidates.append(found)
        if all(c is not None for c in acc_candidates):
            seg_to_acc_cols[seg] = acc_candidates  # type: ignore

    # Load model/motion
    model, state, table, time_col, coord_labels = load_model_and_motion(model_path, motion_path)

    # Orientation fit using zero-offset IMU per segment to get simulated gyro
    best_results: Dict[str, Dict[str, np.ndarray]] = {}
    vis_cache: Dict[str, Dict[str, np.ndarray]] = {}

    for seg, cols in mapping.items():
        # Simulate gyro at zero offset to estimate orientation R
        t_sim, sim_gyro_zero, _ = simulate_signals_for_offset(
            model, state, table, time_col, coord_labels,
            segment_name=seg,
            imu_name=f"loc_{seg}",
            offset_xyz=(0.0, 0.0, 0.0),
            max_frames=args.max_frames,
        )
        time_len = sim_gyro_zero.shape[0]
        real_gyro = real_df[cols["gyro"]].values.astype(float)
        if real_gyro.shape[0] != time_len:
            xp = np.linspace(0, 1, real_gyro.shape[0])
            xq = np.linspace(0, 1, time_len)
            real_gyro = np.vstack([np.interp(xq, xp, real_gyro[:, i]) for i in range(3)]).T

        R = fit_rotation(sim_gyro_zero, real_gyro)

        # If we have accelerometer for this segment, search position
        if seg not in seg_to_acc_cols:
            best_results[seg] = {"R": R, "offset": np.zeros(3)}
            # Cache for visualization (gyro only)
            vis_cache[seg] = {
                "t": t_sim,
                "sim_gyro": sim_gyro_zero,
                "real_gyro_rot": (R @ real_gyro.T).T,
            }
            continue

        real_acc = real_df[seg_to_acc_cols[seg]].values.astype(float)
        if real_acc.shape[0] != time_len:
            xp = np.linspace(0, 1, real_acc.shape[0])
            xq = np.linspace(0, 1, time_len)
            real_acc = np.vstack([np.interp(xq, xp, real_acc[:, i]) for i in range(3)]).T

        # Rotate real accel to simulated (canonical) axes using R
        real_acc_rot = (R @ real_acc.T).T

        # Build search grid
        rmin, rmax, zmin, zmax = default_bounds_for_segment(seg)
        if args.radius_min is not None: rmin = float(args.radius_min)
        if args.radius_max is not None: rmax = float(args.radius_max)
        if args.z_min is not None: zmin = float(args.z_min)
        if args.z_max is not None: zmax = float(args.z_max)
        offs = cylindrical_offsets(seg, rmin, rmax, zmin, zmax, args.r_steps, args.theta_steps, args.z_steps)

        best_mse = float("inf")
        best_off = np.zeros(3)

        for off in offs:
            _, _, sim_acc = simulate_signals_for_offset(
                model, state, table, time_col, coord_labels,
                segment_name=seg,
                imu_name=f"loc_{seg}",
                offset_xyz=off,
                max_frames=args.max_frames,
            )
            # sim_acc already in body-frame axes; compare to rotated real_acc
            if sim_acc.shape[0] != real_acc_rot.shape[0]:
                continue
            mse = float(np.mean((sim_acc - real_acc_rot) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_off = np.array(off)

        best_results[seg] = {"R": R, "offset": best_off}

        # Save per-segment outputs
        np.save(outdir / f"{seg}_R.npy", R)
        np.savetxt(outdir / f"{seg}_R.txt", R, fmt="%.8f")
        np.savetxt(outdir / f"{seg}_offset_xyz.txt", best_off.reshape(1, -1), fmt="%.6f")

        # Cache for visualization (gyro+accel)
        if args.visualize:
            # Simulate at best offset for both gyro/accel
            t_best, sim_gyro_best, sim_acc_best = simulate_signals_for_offset(
                model, state, table, time_col, coord_labels,
                segment_name=seg,
                imu_name=f"loc_{seg}",
                offset_xyz=tuple(best_off.tolist()),
                max_frames=args.max_frames,
            )
            vis_cache[seg] = {
                "t": t_best,
                "sim_gyro": sim_gyro_best,
                "sim_acc": sim_acc_best,
                "real_gyro_rot": (R @ real_gyro.T).T,
                "real_acc_rot": real_acc_rot,
            }

    # Summary
    with open(outdir / "summary.txt", "w") as f:
        for seg, res in best_results.items():
            R = res["R"]
            off = res["offset"]
            f.write(f"{seg}:\nR=\n{R}\noffset_xyz(m)={off.tolist()}\n\n")

    # Visualization
    if args.visualize and plt is not None and vis_cache:
        (outdir / "figures").mkdir(parents=True, exist_ok=True)
        for seg, data in vis_cache.items():
            t = data["t"]
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
            # Gyro: rows[0]
            for i, ax in enumerate(axes[0]):
                ax.plot(t, data["sim_gyro"][:, i], label="sim gyro", color="C0")
                if "real_gyro_rot" in data:
                    ax.plot(t, data["real_gyro_rot"][:, i], label="real gyro (R·real)", color="C1", alpha=0.8)
                ax.grid(True, alpha=0.3)
                ax.set_ylabel(["X","Y","Z"][i])
                if i == 0:
                    ax.set_title(f"{seg} - Gyro")
            # Accel: rows[1]
            for i, ax in enumerate(axes[1]):
                if "sim_acc" in data:
                    ax.plot(t, data["sim_acc"][:, i], label="sim acc", color="C0")
                if "real_acc_rot" in data:
                    ax.plot(t, data["real_acc_rot"][:, i], label="real acc (R·real)", color="C1", alpha=0.8)
                ax.grid(True, alpha=0.3)
                ax.set_ylabel(["X","Y","Z"][i])
                if i == 0:
                    ax.set_title(f"{seg} - Accel")
            axes[-1][-1].legend(loc="upper right")
            axes[-1][1].set_xlabel("Time (s)")
            plt.tight_layout()
            fig.savefig(outdir / "figures" / f"{seg}_comparison.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    # Save marked model (markers at estimated offsets)
    if args.save_marked_model:
        mdl = osim.Model(model)
        mdl.setUseVisualizer(False)
        _ = mdl.initSystem()
        for seg, res in best_results.items():
            off = res["offset"]
            # create an offset frame + marker
            try:
                body = mdl.getBodySet().get(seg)
            except Exception:
                # fallback: approximate name
                body = None
                names = [mdl.getBodySet().get(i).getName() for i in range(mdl.getBodySet().getSize())]
                for n in names:
                    if seg.replace("_r",""").replace("_l",""").split("_")[0] in n.lower():
                        body = mdl.getBodySet().get(n)
                        break
                if body is None:
                    continue
            frame = osim.PhysicalOffsetFrame()
            frame.setName(f"imu_{seg}_frame")
            frame.setParentFrame(body)
            T = osim.Transform()
            T.setP(osim.Vec3(float(off[0]), float(off[1]), float(off[2])))
            frame.setOffsetTransform(T)
            body.addComponent(frame)
            marker = osim.Marker()
            marker.setName(f"imu_{seg}_marker")
            marker.set_location(osim.Vec3(0.0, 0.0, 0.0))
            marker.connectSocket_frame(frame)
            mdl.addComponent(marker)
        mdl.finalizeConnections()
        mdl.print(args.save_marked_model)

    print("\n✓ Completed IMU localization (orientation + position). Results in:", outdir)


if __name__ == "__main__":
    main()


