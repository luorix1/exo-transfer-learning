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


def standardize_segment_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize segment names to canonical format.
    
    Transforms:
    - trunk -> pelvis
    - For unilateral data (no _l/_r suffix):
      - thigh -> thigh_r
      - shank -> shank_r
      - tibia -> tibia_r
      - femur -> femur_r
      - foot -> foot_r
    """
    df = df.copy()
    new_columns = []
    
    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        
        # Step 1: Replace 'trunk' with 'pelvis' (case-insensitive)
        # Match trunk as whole word or followed by underscore
        new_col = re.sub(r'\btrunk(?=_|\b)', 'pelvis', col_str, flags=re.IGNORECASE)
        
        # Step 2: For unilateral data, add _r suffix to segment names without side indicators
        # Update col_lower after trunk replacement
        col_lower = new_col.lower()
        # Check if column already has _l, _r, _left, or _right suffix after the segment name
        has_side_suffix = bool(re.search(r'_(l|r|left|right)_', col_lower))
        
        if not has_side_suffix:
            # Add _r suffix to common unilateral segments (assume right side)
            # Pattern: segment_ (followed by underscore but not _l, _r, _left, _right)
            # Note: pelvis doesn't get _r since it's already bilateral/central
            for segment in ['thigh', 'femur', 'shank', 'tibia', 'foot']:
                # Match: segment followed by underscore (but not _l, _r, _left, _right)
                # e.g., 'thigh_gyro' -> 'thigh_r_gyro'
                pattern = rf'\b{segment}_(?!(l|r|left|right)_)'
                if re.search(pattern, col_lower):
                    # Replace with segment_r_
                    new_col = re.sub(pattern, f'{segment}_r_', new_col, flags=re.IGNORECASE)
                    break  # Only replace the first matching segment
        
        new_columns.append(new_col)
    
    df.columns = new_columns
    return df


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


def load_subject_info(subject_dir: Path, subject_name: Optional[str] = None) -> dict:
    """Load subject information including bodyweight for normalization.

    This function is robust to different dataset layouts:
    - Subject-level files: SubjectInfo.csv, subject_info.csv, info.csv, metadata.csv
    - Nested metadata inside subject directory (searched up to 2 levels deep)
    - Dataset-level demographics table with a subject identifier column

    It attempts multiple parsing strategies to extract bodyweight (kg).
    """
    subject_info: Dict[str, float] = {}

    def try_extract_bodyweight(df: pd.DataFrame) -> Optional[float]:
        # Strategy 1: direct numeric column with weight keywords
        weight_cols = [c for c in df.columns if any(k in str(c).lower() for k in ['weight', 'mass', 'bodyweight', 'body_mass'])]
        for c in weight_cols:
            series = pd.to_numeric(df[c], errors='coerce')
            # Prefer single-row tables
            val = series.dropna()
            if not val.empty:
                w = float(val.iloc[0])
                if 20 <= w <= 200:
                    return w
        # Strategy 2: key-value pairs (first col key, second col value)
        if df.shape[1] >= 2:
            key_col = df.columns[0]
            val_col = df.columns[1]
            keys = df[key_col].astype(str).str.lower()
            mask = keys.str.contains('weight') | keys.str.contains('mass') | keys.str.contains('bodyweight') | keys.str.contains('body_mass')
            rows = df[mask]
            if not rows.empty:
                series = pd.to_numeric(rows[val_col], errors='coerce').dropna()
                if not series.empty:
                    w = float(series.iloc[0])
                    if 20 <= w <= 200:
                        return w
        return None

    def read_info_csv(path: Path) -> Optional[pd.DataFrame]:
        try:
            return read_csv_flexible(path)
        except Exception:
            try:
                return pd.read_csv(path)
            except Exception:
                return None

    # 1) Obvious filenames in subject directory
    candidate_files: List[Path] = [
        subject_dir / "SubjectInfo.csv",
        subject_dir / "subject_info.csv",
        subject_dir / "info.csv",
        subject_dir / "metadata.csv",
    ]

    # 2) Nested files inside subject directory (one or two levels deep)
    for pattern in ["**/SubjectInfo.csv", "**/subject_info.csv", "**/*metadata*.csv", "**/*demo*graph*.csv"]:
        candidate_files.extend(subject_dir.glob(pattern))

    # 3) Dataset-level tables that may include per-subject rows
    dataset_root = subject_dir.parent
    for pattern in ["SubjectInfo.csv", "subject_info.csv", "*metadata*.csv", "*demo*graph*.csv"]:
        candidate_files.extend(dataset_root.glob(pattern))

    seen: set = set()
    for info_file in candidate_files:
        if not info_file.exists():
            continue
        if str(info_file) in seen:
            continue
        seen.add(str(info_file))
        df_info = read_info_csv(info_file)
        if df_info is None or df_info.empty:
            continue

        # If table contains subject identifier columns, filter to target subject
        if subject_name is not None:
            subj_cols = [c for c in df_info.columns if any(k in str(c).lower() for k in ['subject', 'id', 'participant'])]
            if subj_cols:
                # Try to match by subject directory name (case-insensitive, ignoring non-alnum)
                def norm(s: str) -> str:
                    return re.sub(r"[^0-9a-zA-Z]", "", s).lower()
                target = norm(subject_name)
                mask = False
                for c in subj_cols:
                    mask = mask | (df_info[c].astype(str).map(norm) == target)
                df_filtered = df_info[mask]
                if not df_filtered.empty:
                    w = try_extract_bodyweight(df_filtered)
                    if w is not None:
                        subject_info['bodyweight'] = w
                        print(f"[info] Loaded bodyweight: {w} kg from {info_file}")
                        return subject_info

        # Fallback: attempt extraction from the whole table
        w = try_extract_bodyweight(df_info)
        if w is not None:
            subject_info['bodyweight'] = w
            print(f"[info] Loaded bodyweight: {w} kg from {info_file}")
            return subject_info

    return subject_info


def normalize_joint_moments(df: pd.DataFrame, subject_info: dict) -> pd.DataFrame:
    """Normalize joint moment columns ("*_moment") by bodyweight from SubjectInfo.csv.

    - Expects bodyweight in kilograms within subject_info
    - Normalizes ONLY columns whose names match "*_moment" (case-insensitive)
    - Safely coerces non-numeric values to NaN before division
    """
    bodyweight = subject_info.get('bodyweight')
    if bodyweight is None or bodyweight <= 0:
        print("[warn] Skipping moment normalization: missing or invalid bodyweight")
        return df

    df_out = df.copy()
    # Header normalization may not be applied yet for the label df, so match case-insensitively
    candidate_cols = []
    for col in df_out.columns:
        col_l = str(col).lower()
        # Match pattern "*_moment" strictly (underscore + moment at end) per requirement
        if col_l.endswith("_moment"):
            candidate_cols.append(col)

    if not candidate_cols:
        # Fallback: any column containing 'moment'
        candidate_cols = [col for col in df_out.columns if 'moment' in str(col).lower()]

    if not candidate_cols:
        print("[warn] No moment columns found to normalize")
        return df

    for col in candidate_cols:
        series = pd.to_numeric(df_out[col], errors='coerce')
        df_out[col] = series / float(bodyweight)
        print(f"[info] Normalized {col} by bodyweight ({bodyweight} kg)")

    return df_out


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

def process_trial(trial: Dict[str, Path], output_root: Path, canonical: bool, unilateral: bool, unit: str, max_frames: int, normalize_moment: bool) -> Optional[Path]:
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
            # Standardize segment names (trunk -> pelvis) for consistency
            df_raw = standardize_segment_names(df_raw)
            write_csv_normalized(df_raw, imu_out)
    else:
        df_raw = read_csv_flexible(imu_csv)
        # Standardize segment names (trunk -> pelvis) for consistency
        df_raw = standardize_segment_names(df_raw)
        write_csv_normalized(df_raw, imu_out)

    # joint_moment.csv: choose CSV in condition-level id and normalize headers
    out_label = label_dir / "joint_moment.csv"
    if id_csv is not None and id_csv.suffix.lower() == ".csv":
        df_label = read_csv_flexible(id_csv)
        # Apply bodyweight normalization if requested
        if normalize_moment:
            subject_dir = cond_dir.parent.parent  # Go up to subject level
            subject_info = load_subject_info(subject_dir)
            df_label = normalize_joint_moments(df_label, subject_info)
        write_csv_normalized(df_label, out_label)
    else:
        id_dir = cond_dir / "id"
        candidates = sorted(id_dir.rglob("*.csv")) if id_dir.exists() else []
        picked = best_match_by_stem(trial_name, candidates) if candidates else None
        if picked is not None:
            df_label = read_csv_flexible(picked)
            # Apply bodyweight normalization if requested
            if normalize_moment:
                subject_dir = cond_dir.parent.parent  # Go up to subject level
                subject_info = load_subject_info(subject_dir)
                df_label = normalize_joint_moments(df_label, subject_info)
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
    parser.add_argument("--normalize-moment", action="store_true", help="Normalize joint moments by bodyweight (requires SubjectInfo.csv)")
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
            normalize_moment=args.normalize_moment,
        )
        print(f"✓ {t['subject']} {t['condition_dir'].name} {Path(t['imu_csv']).stem} -> {out_dir}")
        processed += 1
    print(f"Done. {processed} trial(s) processed.")


if __name__ == "__main__":
    main()
