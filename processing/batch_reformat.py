#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import sys
import re

import pandas as pd

# Optional import of canonical conversion utilities
CANON_AVAILABLE = False
try:
    from processing.canonical_frame_converter import (
        read_imu_csv,
        parse_imu_columns,
        load_model_and_motion,
        add_segment_aligned_imu,
        simulate_gyro,
        fit_rotation,
    )
    import numpy as np
    import opensim as osim  # noqa: F401
    CANON_AVAILABLE = True
except Exception:
    CANON_AVAILABLE = False


def find_subject_model(subject_dir: Path) -> Optional[Path]:
    for root, dirs, files in os.walk(subject_dir):
        if Path(root).name.lower() == "osimxml":
            for f in files:
                if f.lower().endswith(".osim"):
                    return Path(root) / f
    return None


def is_trial_condition_dir(dir_name: str, allowed: List[str]) -> bool:
    ln = dir_name.lower()
    return any(a in ln for a in allowed)


def normalize_stem(name: str) -> str:
    return name.lower().replace(" ", "_")


def best_match_by_stem(target_stem: str, candidates: List[Path]) -> Optional[Path]:
    if not candidates:
        return None
    tnorm = normalize_stem(target_stem)
    for c in candidates:
        if normalize_stem(c.stem) == tnorm:
            return c
    for c in candidates:
        if tnorm in normalize_stem(c.name):
            return c
    return candidates[0]


# ---------------------- Robust CSV reading ----------------------

def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Try reading with common delimiters and python engine if needed."""
    seps = [",", ";", "\t"]
    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
            # Heuristic: consider success if more than 1 column or any non-empty
            if df.shape[1] > 1 or df.dropna(how="all").shape[0] > 0:
                return df
        except Exception:
            continue
    # Final fallback: default read_csv
    try:
        return pd.read_csv(path)
    except Exception:
        # give an empty df but not None
        return pd.DataFrame()


# ---------------------- Header normalization ----------------------

def to_snake_case(name: str) -> str:
    s = name.strip()
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.lower()
    return s


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        new_cols.append(to_snake_case(str(c)))
    df = df.copy()
    df.columns = new_cols
    for cand in ["time", "header", "timestamp"]:
        if cand in df.columns:
            df.rename(columns={cand: "time"}, inplace=True)
            break
    return df


def write_csv_normalized(df: pd.DataFrame, out_path: Path) -> None:
    df_norm = normalize_headers(df)
    df_norm.to_csv(out_path, index=False)


# ---------------------- Trial discovery ----------------------

def gather_trials(input_root: Path, allowed_conditions: List[str]) -> List[Dict[str, Path]]:
    """Return list of trial dicts with keys: subject, condition_dir, imu_csv, motion_sto, id_csv, model"""
    trials = []
    for subject_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        model_path = find_subject_model(subject_dir)
        for date_dir in sorted([p for p in subject_dir.iterdir() if p.is_dir()]):
            for cond_dir in sorted([p for p in date_dir.iterdir() if p.is_dir()]):
                if not is_trial_condition_dir(cond_dir.name, allowed_conditions):
                    continue
                imu_dir = cond_dir / "imu"
                opensim_dir = cond_dir / "opensim"
                id_dir = cond_dir / "id"
                if not imu_dir.exists() or not opensim_dir.exists():
                    continue
                imu_csvs = sorted(imu_dir.glob("*.csv"))
                motion_stos = list(opensim_dir.rglob("*states.sto"))
                id_csvs = sorted(id_dir.rglob("*.csv")) if id_dir.exists() else []

                for imu_csv in imu_csvs:
                    trial_name = imu_csv.stem
                    motion_sto = best_match_by_stem(trial_name, motion_stos) if motion_stos else None
                    id_csv = best_match_by_stem(trial_name, id_csvs) if id_csvs else None
                    trial = {
                        "subject": subject_dir.name,
                        "condition_dir": cond_dir,
                        "imu_csv": imu_csv,
                        "motion_sto": motion_sto,
                        "id_csv": id_csv,
                        "model": model_path,
                    }
                    trials.append(trial)
    return trials


# ---------------------- Canonical conversion ----------------------

def run_canonical_conversion_if_available(
    model_path: Path,
    motion_sto: Path,
    imu_csv: Path,
    out_csv: Path,
    unilateral: bool,
    unit: str,
    max_frames: int,
) -> bool:
    if not CANON_AVAILABLE:
        return False
    try:
        df = read_imu_csv(imu_csv)
        mapping = parse_imu_columns(df, unilateral=unilateral)
        model, state, table, time_col, coord_labels = load_model_and_motion(model_path, motion_sto)
        seg_to_imu: Dict[str, "osim.IMU"] = {}
        for seg in mapping.keys():
            imu = add_segment_aligned_imu(model, seg, f"canonical_{seg}")
            seg_to_imu[seg] = imu
        sim_map = simulate_gyro(model, state, table, time_col, coord_labels, seg_to_imu, max_frames)
        time_len = min([len(v) for v in sim_map.values()])
        common_t = pd.Series([float(time_col[i]) for i in range(time_len)], name="time")
        cols = {"time": common_t}
        for seg, cols_map in mapping.items():
            gyro_cols = cols_map["gyro"]
            real = df[gyro_cols].values.astype(float)
            if unit == "deg":
                real = real * (np.pi / 180.0)
            if real.shape[0] != time_len:
                xp = np.linspace(0, 1, real.shape[0])
                xq = np.linspace(0, 1, time_len)
                real = np.vstack([np.interp(xq, xp, real[:, i]) for i in range(3)]).T
            R = fit_rotation(sim_map[seg][:time_len], real)
            real_conv = (R @ real.T).T
            cols[to_snake_case(f"{seg}_Gyro_X")] = pd.Series(real_conv[:, 0])
            cols[to_snake_case(f"{seg}_Gyro_Y")] = pd.Series(real_conv[:, 1])
            cols[to_snake_case(f"{seg}_Gyro_Z")] = pd.Series(real_conv[:, 2])
        out_df = pd.DataFrame(cols)
        write_csv_normalized(out_df, out_csv)
        return True
    except Exception:
        return False


# ---------------------- Per-trial processing ----------------------

def process_trial(trial: Dict[str, Path], output_root: Path, canonical: bool, unilateral: bool, unit: str, max_frames: int) -> Optional[Path]:
    subject = trial["subject"]
    cond_dir = trial["condition_dir"]
    imu_csv = trial["imu_csv"]
    motion_sto = trial.get("motion_sto")
    id_csv = trial.get("id_csv")
    model_path = trial.get("model")

    trial_name = imu_csv.stem
    condition_name = cond_dir.name

    out_dir = output_root / subject / condition_name / trial_name
    input_dir = out_dir / "Input"
    label_dir = out_dir / "Label"
    input_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    imu_out = input_dir / "imu_data.csv"
    if canonical and model_path and motion_sto:
        ok = run_canonical_conversion_if_available(model_path, motion_sto, imu_csv, imu_out, unilateral, unit, max_frames)
        if not ok:
            df_raw = read_csv_flexible(imu_csv)
            write_csv_normalized(df_raw, imu_out)
    else:
        df_raw = read_csv_flexible(imu_csv)
        write_csv_normalized(df_raw, imu_out)

    # joint_moment.csv: choose CSV in condition-level id and normalize headers
    out_label = label_dir / "joint_moment.csv"
    if id_csv is not None and id_csv.suffix.lower() == ".csv":
        df_label = read_csv_flexible(id_csv)
        write_csv_normalized(df_label, out_label)
    else:
        id_dir = cond_dir / "id"
        candidates = sorted(id_dir.rglob("*.csv")) if id_dir.exists() else []
        picked = best_match_by_stem(trial_name, candidates) if candidates else None
        if picked is not None:
            df_label = read_csv_flexible(picked)
            write_csv_normalized(df_label, out_label)
        else:
            print(f"[warn] No ID CSV found for trial {trial_name} in {id_dir}")

    return out_dir


# ---------------------- CLI ----------------------

def main():
    parser = argparse.ArgumentParser(description="Reformat dataset into MeMo_processed layout, optionally with canonical IMU conversion")
    parser.add_argument("--input-root", required=True, help="Path to dataset root (e.g., Keaton_processed or subset)")
    parser.add_argument("--output-root", required=True, help="Path to output root (MeMo_processed-like)")
    parser.add_argument("--conditions", default="levelground,treadmill", help="Comma-separated condition filters")
    parser.add_argument("--canonical", action="store_true", help="Perform canonical IMU conversion if possible")
    parser.add_argument("--unilateral", action="store_true", help="Assume unilateral IMU naming (map to right side)")
    parser.add_argument("--unit", choices=["rad", "deg"], default="rad", help="Unit of IMU gyro in source CSV")
    parser.add_argument("--max-frames", type=int, default=2000)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    allowed = [c.strip().lower() for c in args.conditions.split(",") if c.strip()]

    trials = gather_trials(input_root, allowed)
    if not trials:
        print("No trials found matching conditions.")
        sys.exit(1)

    print(f"Found {len(trials)} trial(s). Processing...")
    processed = 0
    for t in trials:
        out_dir = process_trial(
            t,
            output_root,
            canonical=args.canonical,
            unilateral=args.unilateral,
            unit=args.unit,
            max_frames=args.max_frames,
        )
        print(f"✓ {t['subject']} {t['condition_dir'].name} {Path(t['imu_csv']).stem} -> {out_dir}")
        processed += 1
    print(f"Done. {processed} trial(s) processed.")


if __name__ == "__main__":
    main()
