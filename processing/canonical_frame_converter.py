#!/usr/bin/env python3
"""
Canonical Frame Converter

This script maps real IMU angular velocity data into the OpenSim canonical (segment-aligned)
frames by estimating a 3x3 rotation matrix per segment that best aligns real IMU gyro data
with simulated gyro signals computed from the OpenSim model and motion.

Usage:
  python canonical_frame_converter.py \
    --model path/model.osim \
    --motion path/walking_motion_states.sto \
    --imu path/real_imu.csv \
    --output output_dir \
    [--segments femur_r,tibia_r,pelvis] \
    [--max-frames 2000] \
    [--visualize]

Notes:
- Self-contained (only imports opensim, numpy, pandas, matplotlib).
- Discovers IMU column names case-insensitively and maps to segments.
- Fits rotation matrices via least-squares (Kabsch / Procrustes) on 3-axis gyro.
- Outputs per-segment rotation matrices and converted angular velocities.
"""

import argparse
from pathlib import Path
import os
from typing import Dict, List, Tuple

import numpy as np
import opensim as osim
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ---------------------- I/O helpers ----------------------

def read_imu_csv(imu_path: Path) -> pd.DataFrame:
    df = pd.read_csv(imu_path)
    # Build case-insensitive lookup
    df.columns = [str(c) for c in df.columns]
    return df


def lower_map(columns: List[str]) -> Dict[str, str]:
    return {c.lower(): c for c in columns}


# ---------------------- Column parsing ----------------------

SEGMENT_ALIASES = {
    # canonical opensim segment names mapped from common IMU prefixes
    "pelvis": ["pelvis", "pelv", "trunk", "torso"],
    "femur_r": ["thigh_r", "femur_r", "thigh", "rthigh", "right_thigh"],
    "tibia_r": ["shank_r", "tibia_r", "shank", "rshank", "right_shank"],
    "foot_r": ["foot_r", "rfoot", "right_foot", "foot"],
    "femur_l": ["thigh_l", "femur_l", "lthigh", "left_thigh"],
    "tibia_l": ["shank_l", "tibia_l", "lshank", "left_shank"],
    "foot_l": ["foot_l", "lfoot", "left_foot"],
}

AXIS_ALIASES = {
    # accept a variety of IMU column name styles
    "gyro": ["gyro", "gyroscope", "gyr", "angular_velocity", "angvel"],
    "acc": ["acc", "accelerometer", "accel", "linacc"],
    "x": ["x"],
    "y": ["y"],
    "z": ["z"],
}


def find_segment_from_column(col_name: str, unilateral: bool) -> str:
    """Return canonical segment for a column name.
    If unilateral=True, map generic names to right side by convention.
    """
    lc = col_name.lower()
    if unilateral:
        # direct mapping for unilateral datasets (assume right side)
        if "thigh" in lc or "femur" in lc:
            return "femur_r"
        if "shank" in lc or "tibia" in lc:
            return "tibia_r"
        if "foot" in lc:
            return "foot_r"
        if "trunk" in lc or "pelvis" in lc or "torso" in lc:
            return "pelvis"
    # fallback: alias search (case-insensitive)
    for segment, aliases in SEGMENT_ALIASES.items():
        for a in aliases:
            if a.lower() in lc:
                return segment
    return ""


def parse_imu_columns(df: pd.DataFrame, unilateral: bool) -> Dict[str, Dict[str, List[str]]]:
    """Return mapping segment -> {gyro: [colX,colY,colZ], acc: [colX,colY,colZ], header: col}
    Chooses the first matching triplet found for each type.
    """
    lmap = lower_map(list(df.columns))
    # time/header
    header_col = lmap.get("header", None) or lmap.get("time", None)

    per_segment: Dict[str, Dict[str, List[str]]] = {}
    for col in df.columns:
        seg = find_segment_from_column(col, unilateral)
        if not seg:
            continue
        lc = col.lower()
        is_gyro = any(k in lc for k in [s.lower() for s in AXIS_ALIASES["gyro"]])
        is_acc = any(k in lc for k in [s.lower() for s in AXIS_ALIASES["acc"]])
        axis = None
        if "_x" in lc or lc.endswith("x"): axis = "x"
        if "_y" in lc or lc.endswith("y"): axis = axis or "y"
        if "_z" in lc or lc.endswith("z"): axis = axis or "z"
        if axis is None or (not is_gyro and not is_acc):
            continue
        per_segment.setdefault(seg, {"gyro": [None, None, None], "acc": [None, None, None], "header": header_col})
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        key = "gyro" if is_gyro else "acc"
        if per_segment[seg][key][idx] is None:
            per_segment[seg][key][idx] = col

    # prune segments without full gyro triplet
    pruned = {}
    for seg, m in per_segment.items():
        if all(m["gyro"][i] is not None for i in range(3)):
            pruned[seg] = m
    return pruned


# ---------------------- OpenSim helpers ----------------------

def _set_geometry_paths():
    """Add common geometry paths to suppress VTP warnings in OpenSim GUI/loader."""
    try:
        # Default macOS installation paths
        for p in [
            "/Applications/OpenSim 4.5/Geometry",
            "/Applications/OpenSim 4.4/Geometry",
            "/Applications/OpenSim 4.3/Geometry",
        ]:
            if os.path.isdir(p):
                osim.ModelVisualizer.addDirToGeometrySearchPaths(p)
        # Environment variable override
        env_path = os.environ.get("OPENSIM_GEOMETRY")
        if env_path and os.path.isdir(env_path):
            osim.ModelVisualizer.addDirToGeometrySearchPaths(env_path)
    except Exception:
        pass


def load_model_and_motion(model_path: Path, motion_path: Path):
    _set_geometry_paths()
    model = osim.Model(str(model_path))
    model.setUseVisualizer(False)
    state = model.initSystem()
    table = osim.TimeSeriesTable(str(motion_path))
    time_col = table.getIndependentColumn()
    coord_labels = list(table.getColumnLabels())
    return model, state, table, time_col, coord_labels


def add_segment_aligned_imu(model: osim.Model, segment_name: str, imu_name: str) -> osim.IMU:
    # Resolve segment name to an actual body in the model (models differ: calcn_r vs foot_r, torso vs pelvis)
    def resolve_body_name(model: osim.Model, desired: str) -> str:
        names = [model.getBodySet().get(i).getName() for i in range(model.getBodySet().getSize())]
        if desired in names:
            return desired
        # Candidate aliases per desired key
        alias_map = {
            "foot_r": ["calcn_r", "foot_r", "calcn", "foot"],
            "foot_l": ["calcn_l", "foot_l", "calcn", "foot"],
            "tibia_r": ["tibia_r", "shank_r", "tibia"],
            "tibia_l": ["tibia_l", "shank_l", "tibia"],
            "femur_r": ["femur_r", "thigh_r", "femur"],
            "femur_l": ["femur_l", "thigh_l", "femur"],
            "pelvis": ["pelvis", "torso", "trunk"],  # some models use torso for upper body sensor
        }
        for cand in alias_map.get(desired, []):
            for n in names:
                if n.lower() == cand.lower():
                    return n
        # As a last resort, substring search
        key = desired.replace("_r", "").replace("_l", "")
        for n in names:
            if key.split("_")[0] in n.lower():
                return n
        raise ValueError(f"Segment not found in model: '{desired}'. Available bodies: {names}")

    resolved = resolve_body_name(model, segment_name)
    body = model.getBodySet().get(resolved)
    frame = osim.PhysicalOffsetFrame()
    frame.setName(f"{imu_name}_frame")
    frame.setParentFrame(body)
    frame.setOffsetTransform(osim.Transform())  # identity: aligned with body
    body.addComponent(frame)
    imu = osim.IMU()
    imu.setName(imu_name)
    imu.connectSocket_frame(frame)
    model.addComponent(imu)
    _ = model.initSystem()
    return imu


def simulate_gyro(model: osim.Model, state: osim.State, table: osim.TimeSeriesTable, time_col, coord_labels, seg_to_imu: Dict[str, osim.IMU], max_frames: int) -> Dict[str, np.ndarray]:
    num_frames = min(max_frames, table.getNumRows())
    results: Dict[str, List[np.ndarray]] = {seg: [] for seg in seg_to_imu.keys()}

    for i in range(num_frames):
        t = time_col[i]
        state.setTime(t)
        # set coordinates and speeds if present
        for coord in model.getCoordinateSet():
            q = coord.getName()
            u = q + "_u"
            if q in coord_labels:
                coord.setValue(state, table.getDependentColumn(q)[i])
            if u in coord_labels:
                coord.setSpeedValue(state, table.getDependentColumn(u)[i])
        model.realizeAcceleration(state)
        for seg, imu in seg_to_imu.items():
            g = imu.calcGyroscopeSignal(state)
            results[seg].append(np.array([g.get(0), g.get(1), g.get(2)], dtype=float))
    # stack
    return {seg: np.vstack(vals) for seg, vals in results.items()}


# ---------------------- Fitting ----------------------

def fit_rotation(sim_gyro: np.ndarray, real_gyro: np.ndarray) -> np.ndarray:
    """Fit 3x3 rotation R s.t. R @ real ≈ sim in LS sense.

    This mirrors the orthogonal Procrustes/Kabsch approach used in imu_optimization.py.
    """
    # Center the data like imu_optimization
    P = real_gyro - np.mean(real_gyro, axis=0)
    Q = sim_gyro - np.mean(sim_gyro, axis=0)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


# ---------------------- Visualization ----------------------

def visualize_segments(common_t, sim_map, real_map, R_map, outdir: Path):
    if plt is None:
        print("Visualization skipped (matplotlib unavailable).")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    for seg in sim_map:
        sim = sim_map[seg]
        real = real_map[seg]
        R = R_map[seg]
        real_to_canonical = (R @ real.T).T
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        colors = ["r", "g", "b"]
        for i in range(3):
            axes[i].plot(common_t, sim[:, i], color=colors[i], label=f"Sim {['X','Y','Z'][i]}")
            axes[i].plot(common_t, real_to_canonical[:, i], color=colors[i], linestyle="--", label=f"Real→Canonical {['X','Y','Z'][i]}")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        fig.savefig(outdir / f"{seg}_gyro_alignment.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------- Main ----------------------

def main():
    parser = argparse.ArgumentParser(description="Convert real IMU gyro to OpenSim canonical segment-aligned frames")
    parser.add_argument("--model", required=True, help="Path to OpenSim model (.osim)")
    parser.add_argument("--motion", required=True, help="Path to motion (.sto with states)")
    parser.add_argument("--imu", required=True, help="Path to real IMU CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--segments", help="Comma-separated segments to include (default: auto from CSV)")
    parser.add_argument("--max-frames", type=int, default=2000, help="Max frames to process")
    parser.add_argument("--visualize", action="store_true", help="Plot comparisons")
    parser.add_argument("--unilateral", action="store_true", help="Assume all IMU columns are right side and map by generic segment names")
    parser.add_argument("--unit", choices=["rad","deg"], default="rad", help="Unit of real IMU gyro in CSV (default: rad)")

    args = parser.parse_args()

    model_path = Path(args.model)
    motion_path = Path(args.motion)
    imu_path = Path(args.imu)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Parse IMU CSV
    df = read_imu_csv(imu_path)
    mapping = parse_imu_columns(df, unilateral=args.unilateral)
    if args.segments:
        wanted = set(s.strip() for s in args.segments.split(",") if s.strip())
        mapping = {k: v for k, v in mapping.items() if k in wanted}
    if not mapping:
        raise RuntimeError("No segments with full gyro triplets found in the IMU CSV")

    # 2) Load model/motion
    model, state, table, time_col, coord_labels = load_model_and_motion(model_path, motion_path)

    # 3) Create canonical IMUs per segment (aligned with body)
    seg_to_imu: Dict[str, osim.IMU] = {}
    for seg in mapping.keys():
        imu = add_segment_aligned_imu(model, seg, f"canonical_{seg}")
        seg_to_imu[seg] = imu

    # 4) Simulate gyro per segment
    sim_gyro_map = simulate_gyro(model, state, table, time_col, coord_labels, seg_to_imu, args.max_frames)

    # Extract real gyro per segment and time vector (assume Header/time present or index)
    time_len = min([len(v) for v in sim_gyro_map.values()])
    common_t = np.array(time_col[:time_len])

    real_gyro_map: Dict[str, np.ndarray] = {}
    R_map: Dict[str, np.ndarray] = {}

    for seg, cols in mapping.items():
        gyro_cols = cols["gyro"]
        real = df[gyro_cols].values.astype(float)
        # Convert units if needed
        if args.unit == "deg":
            real = real * (np.pi / 180.0)
        # If CSV length differs, resample or truncate
        if real.shape[0] != time_len:
            # simple linear resample to match sim length
            xp = np.linspace(0, 1, real.shape[0])
            xq = np.linspace(0, 1, time_len)
            real = np.vstack([np.interp(xq, xp, real[:, i]) for i in range(3)]).T
        real_gyro_map[seg] = real.astype(float)
        # 5) Fit R such that R @ real ≈ sim (canonical)
        R = fit_rotation(sim_gyro_map[seg][:time_len], real_gyro_map[seg])
        R_map[seg] = R
        # Save R
        np.save(outdir / f"{seg}_real_to_canonical_R.npy", R)
        np.savetxt(outdir / f"{seg}_real_to_canonical_R.txt", R, fmt="%.8f")
        # Convert and save transformed gyro
        real_conv = (R @ real_gyro_map[seg].T).T
        conv_df = pd.DataFrame(real_conv, columns=[f"{seg}_Gyro_X", f"{seg}_Gyro_Y", f"{seg}_Gyro_Z"])
        conv_df.insert(0, "Time", common_t)
        conv_df.to_csv(outdir / f"{seg}_gyro_canonical.csv", index=False)

    # 6) Visualization
    if args.visualize:
        visualize_segments(common_t, {k: v[:time_len] for k, v in sim_gyro_map.items()}, real_gyro_map, R_map, outdir)

    print("\n✓ Completed canonical frame conversion.")
    print(f"Saved outputs in: {outdir}")


if __name__ == "__main__":
    main()
