#!/usr/bin/env python3
"""
Axis probe utility

Purpose
-------
For a given Canonical-style trial directory (Subject/Condition/Trial), plot:
  - Right thigh IMU Y-axis gyroscope
  - Three axes of right hip moment (X, Y, Z)
in four subplots to visually inspect which hip moment axis corresponds to flexion.

This script is robust to several header naming variants. It normalizes headers
to snake_case and then searches for common patterns.

Usage
-----
python scripts/axis_probe.py \
  --trial "/path/to/Subject/Condition/trial_1" \
  --out "./axis_probe.png"

If --out is not provided, the figure will be saved next to the trial as
{trial_basename}_axis_probe.png
"""

import argparse
from pathlib import Path
from typing import Optional, List
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_csv_flexible(path: Path) -> pd.DataFrame:
    seps = [",", ";", "\t"]
    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
            if df.shape[1] > 1 or df.dropna(how="all").shape[0] > 0:
                return df
        except Exception:
            continue
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def to_snake_case(name: str) -> str:
    s = name.strip()
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.lower()
    return s


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake_case(str(c)) for c in df.columns]
    # time/header normalization (optional)
    for cand in ["time", "header", "timestamp"]:
        if cand in df.columns:
            df.rename(columns={cand: "time"}, inplace=True)
            break
    return df


def find_thigh_r_gyro_axes(df: pd.DataFrame) -> dict:
    """Find column names for right thigh (or femur) gyro x,y,z."""
    axes = {"x": None, "y": None, "z": None}
    for c in df.columns:
        lc = c.lower()
        if ("thigh_r" in lc or "femur_r" in lc) and ("gyro" in lc or "gyr" in lc):
            if lc.endswith("_x") and axes["x"] is None:
                axes["x"] = c
            elif lc.endswith("_y") and axes["y"] is None:
                axes["y"] = c
            elif lc.endswith("_z") and axes["z"] is None:
                axes["z"] = c
    # Prefer specific names when duplicates exist
    for ax, pref in [("x", "thigh_r_gyro_x"), ("y", "thigh_r_gyro_y"), ("z", "thigh_r_gyro_z")]:
        if pref in df.columns:
            axes[ax] = pref
    return axes


def find_right_hip_moment_axes(df: pd.DataFrame) -> List[Optional[str]]:
    """Return columns for right hip moment (x,y,z) best-effort.

    Supports variants like:
      - rhipmoment_x, rhipmoment_y, rhipmoment_z (MeMo-style after normalization)
      - hip_r_moment_x, hip_r_moment_y, hip_r_moment_z
      - right_hip_moment_x, right_hip_moment_y, right_hip_moment_z
      - hip_flexion_r_moment (Y only) – in that case we fill only Y.
    """
    cols = df.columns
    # Exact targets in order of preference
    patterns = [
        ("x", ["rhipmoment_x", "hip_r_moment_x", "right_hip_moment_x"]),
        ("y", ["rhipmoment_y", "hip_r_moment_y", "right_hip_moment_y", "hip_flexion_r_moment"]),
        ("z", ["rhipmoment_z", "hip_r_moment_z", "right_hip_moment_z"]),
    ]

    result: List[Optional[str]] = [None, None, None]
    for idx, (axis, axis_patts) in enumerate(patterns):
        # First exact match
        for patt in axis_patts:
            if patt in cols:
                result[idx] = patt
                break
        if result[idx] is not None:
            continue
        # Fuzzy: any column containing all tokens
        tokens_sets = [patt.split("_") for patt in axis_patts]
        for c in cols:
            lc = c.lower()
            for tokens in tokens_sets:
                if all(tok in lc for tok in tokens):
                    # for the Y-axis, avoid selecting flexion if a full axis set exists
                    result[idx] = c
                    break
            if result[idx] is not None:
                break

    return result


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or np.isnan(s):
        return np.zeros_like(x)
    return x / s


def best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 200) -> tuple:
    """Compute best (max |corr|) Pearson correlation within +/- max_lag samples.

    Returns (best_corr, best_lag). Positive lag means b is shifted forward (b[t+lag]).
    """
    a = np.nan_to_num(a.astype(float))
    b = np.nan_to_num(b.astype(float))
    a = zscore(a)
    b = zscore(b)
    L = min(len(a), len(b))
    a = a[:L]
    b = b[:L]
    best_c = -1.0
    best_l = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: len(aa)]
        elif lag > 0:
            bb = b[lag:]
            aa = a[: len(bb)]
        else:
            aa, bb = a, b
        if len(aa) < 10:
            continue
        c = np.corrcoef(aa, bb)[0, 1]
        if np.isnan(c):
            continue
        if abs(c) > abs(best_c):
            best_c = c
            best_l = lag
    return float(best_c), int(best_l)


def main():
    parser = argparse.ArgumentParser(description="Probe axis correspondence between IMU and hip moment")
    parser.add_argument("--trial", required=True, help="Path to trial directory (Subject/Condition/trial)")
    parser.add_argument("--out", default=None, help="Output PNG path (default beside trial)")
    args = parser.parse_args()

    trial_dir = Path(args.trial)
    # Resolve IMU file (Canonical or MeMo_processed)
    imu_csv = trial_dir / "Input" / "imu_data.csv"
    if not imu_csv.exists():
        # Fallback: first CSV under Input
        input_dir = trial_dir / "Input"
        cand = sorted(input_dir.glob("*.csv")) if input_dir.exists() else []
        if cand:
            imu_csv = cand[0]
    # Resolve label file (Canonical or MeMo_processed)
    label_csv = trial_dir / "Label" / "joint_moment.csv"
    if not label_csv.exists():
        label_dir = trial_dir / "Label"
        cand = sorted(label_dir.glob("*.csv")) if label_dir.exists() else []
        if cand:
            label_csv = cand[0]
    if not imu_csv.exists() or not label_csv.exists():
        raise FileNotFoundError(f"Missing input or label CSV in {trial_dir}")

    imu_df = normalize_headers(read_csv_flexible(imu_csv))
    label_df = normalize_headers(read_csv_flexible(label_csv))

    gyro_cols = find_thigh_r_gyro_axes(imu_df)
    if not all(gyro_cols.values()):
        raise RuntimeError("Could not find right thigh IMU gyro x,y,z columns")

    hip_cols = find_right_hip_moment_axes(label_df)
    # Extract gyro signals
    imu_x = pd.to_numeric(imu_df[gyro_cols['x']], errors='coerce').to_numpy()
    imu_y = pd.to_numeric(imu_df[gyro_cols['y']], errors='coerce').to_numpy()
    imu_z = pd.to_numeric(imu_df[gyro_cols['z']], errors='coerce').to_numpy()

    hip_x = pd.to_numeric(label_df[hip_cols[0]], errors='coerce').to_numpy() if hip_cols[0] else None
    hip_y = pd.to_numeric(label_df[hip_cols[1]], errors='coerce').to_numpy() if hip_cols[1] else None
    hip_z = pd.to_numeric(label_df[hip_cols[2]], errors='coerce').to_numpy() if hip_cols[2] else None

    # Hip moment is in N-mm/kg, convert to N-m/kg
    hip_x = hip_x / 1000.0
    hip_y = hip_y / 1000.0
    hip_z = hip_z / 1000.0

    # Time alignment: truncate to min length to visualize side-by-side
    lengths = [len(imu_x), len(imu_y), len(imu_z)] + [len(s) for s in [hip_x, hip_y, hip_z] if s is not None]
    L = min(lengths)
    imu_x = imu_x[:L]
    imu_y = imu_y[:L]
    imu_z = imu_z[:L]
    if hip_x is not None: hip_x = hip_x[:L]
    if hip_y is not None: hip_y = hip_y[:L]
    if hip_z is not None: hip_z = hip_z[:L]

    # Overlay pairs:
    # 1) gyro Y vs hip X (flex/ext)
    # 2) gyro X vs hip Z (rotation)
    # 3) gyro Z vs hip Y (abd/add)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    t = np.arange(L)

    def overlay(ax, gyro_sig, gyro_name, hip_sig, hip_name):
        if hip_sig is None:
            ax.text(0.5, 0.5, f"{hip_name} not found", ha='center', va='center')
            ax.set_title(f"{gyro_name} vs {hip_name}")
            ax.grid(True, alpha=0.3)
            return 0.0, 0
        ax2 = ax.twinx()
        ax.plot(t, gyro_sig, 'g-', linewidth=1.0, label=gyro_name)
        ax.set_ylabel("Gyro (rad/s)", color='g')
        ax.tick_params(axis='y', labelcolor='g')
        ax2.plot(t, hip_sig, 'b-', linewidth=1.0, alpha=0.8, label=hip_name)
        ax2.set_ylabel("Hip Moment (N-m/kg)", color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax.set_title(f"{gyro_name} vs {hip_name}")
        ax.grid(True, alpha=0.3)
        corr, lag = best_lag_corr(gyro_sig, hip_sig, max_lag=200)
        ax.text(0.01, 0.95, f"corr={corr:+.3f}, lag={lag}", transform=ax.transAxes,
                ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
        return corr, lag

    c1 = overlay(axes[0], imu_y, f"thigh_r_gyro_y ({gyro_cols['y']})", hip_x, f"hip_x ({hip_cols[0] or 'missing'})")
    c2 = overlay(axes[1], imu_x, f"thigh_r_gyro_x ({gyro_cols['x']})", hip_z, f"hip_z ({hip_cols[2] or 'missing'})")
    c3 = overlay(axes[2], imu_z, f"thigh_r_gyro_z ({gyro_cols['z']})", hip_y, f"hip_y ({hip_cols[1] or 'missing'})")

    axes[-1].set_xlabel("Sample Index")

    # Summary in title
    title = f"Axis Probe: {trial_dir.parts[-3]}/{trial_dir.parts[-2]}/{trial_dir.parts[-1]}\n" \
            f"Ygyro↔HipX: corr={c1[0]:+.3f}@lag={c1[1]} | " \
            f"Xgyro↔HipZ: corr={c2[0]:+.3f}@lag={c2[1]} | " \
            f"Zgyro↔HipY: corr={c3[0]:+.3f}@lag={c3[1]}"
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = Path(args.out) if args.out else (trial_dir / f"{trial_dir.name}_axis_probe.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


