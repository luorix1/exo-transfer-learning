#!/usr/bin/env python3
"""
Process MeMo_processed dataset into standardized Canonical format.

This script:
1. Reads MeMo_processed structure (already in Subject/Condition/Trial/Input,Label format)
2. Renames Label CSVs from {subject}_{condition}_{trial}.csv to joint_moment.csv
3. Applies column name standardization (lowercase with underscores)
4. Transforms IMU gyroscope data from MeMo frame to OpenSim canonical frame

MeMo IMU Frame Convention:
- X = Up (vertical)
- Y = Left (mediolateral)
- Z = Back (anterior-posterior, pointing backward)

OpenSim Canonical Frame Convention:
- X = Forward (anterior-posterior, pointing forward)
- Y = Up (vertical)
- Z = Right (mediolateral)

Rotation Transformation:
From MeMo [x_up, y_left, z_back] to Canonical [x_fwd, y_up, z_right]:
    x_canonical (forward) = -z_memo (back flipped to forward)
    y_canonical (up)      =  x_memo (up stays up)
    z_canonical (right)   = -y_memo (left flipped to right)

This is a 90-degree rotation + axis flips to align the coordinate systems.
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional
import sys

import pandas as pd
import numpy as np


def to_snake_case(name: str) -> str:
    """Convert column name to lowercase snake_case."""
    s = name.strip()
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.lower()
    return s


def standardize_segment_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize segment names to canonical format.
    
    Transforms:
    - trunk/torso -> pelvis
    - Thigh_L -> thigh_l, Thigh_R -> thigh_r (via snake_case)
    - gyr -> gyro (expand abbreviation)
    - acc -> accel (expand abbreviation)
    - Maintains _l/_r suffixes for bilateral segments
    """
    df = df.copy()
    new_columns = []
    
    for col in df.columns:
        col_str = str(col)
        
        # First apply snake_case normalization
        new_col = to_snake_case(col_str)
        
        # Replace 'trunk' or 'torso' with 'pelvis' (case-insensitive, already lowercase)
        new_col = re.sub(r'\btrunk\b', 'pelvis', new_col)
        new_col = re.sub(r'\btorso\b', 'pelvis', new_col)
        
        # Standardize gyroscope naming: gyr -> gyro (must come before acc to avoid gyr->accelro)
        new_col = re.sub(r'\bgyr_', 'gyro_', new_col)
        new_col = re.sub(r'_gyr_', '_gyro_', new_col)
        
        # Standardize accelerometer naming: acc -> accel
        new_col = re.sub(r'\bacc_', 'accel_', new_col)
        new_col = re.sub(r'_acc_', '_accel_', new_col)
        
        new_columns.append(new_col)
    
    df.columns = new_columns
    return df


def transform_memo_to_canonical(df: pd.DataFrame, segments: Optional[List[str]] = None) -> pd.DataFrame:
    """Transform gyroscope data from MeMo frame to OpenSim canonical frame.
    
    MeMo Frame: [x=up, y=left, z=back]
    Canonical Frame: [x=forward, y=up, z=right]
    
    Transformation:
        canonical_x (forward) = -memo_z (flip back to forward)
        canonical_y (up)      =  memo_x (up stays up)
        canonical_z (right)   = -memo_y (flip left to right)
    
    Args:
        df: DataFrame with gyroscope columns
        segments: List of segments to transform (default: ['pelvis', 'thigh_l', 'thigh_r'])
    
    Returns:
        DataFrame with transformed gyroscope data
    """
    if segments is None:
        segments = ['pelvis', 'thigh_l', 'thigh_r']
    
    df_out = df.copy()
    
    for segment in segments:
        # Find gyroscope columns for this segment (already snake_case and lowercase)
        gyro_x_col = None
        gyro_y_col = None
        gyro_z_col = None
        
        for col in df_out.columns:
            col_lower = col.lower()
            # Match pattern: {segment}_gyro_{x,y,z} or {segment}_gyr_{x,y,z}
            if segment in col_lower and ('gyro' in col_lower or 'gyr' in col_lower):
                if col_lower.endswith('_x'):
                    gyro_x_col = col
                elif col_lower.endswith('_y'):
                    gyro_y_col = col
                elif col_lower.endswith('_z'):
                    gyro_z_col = col
        
        # If all three axes found, apply transformation
        if gyro_x_col and gyro_y_col and gyro_z_col:
            # Extract original MeMo data
            memo_x = df_out[gyro_x_col].values  # up
            memo_y = df_out[gyro_y_col].values  # left
            memo_z = df_out[gyro_z_col].values  # back
            
            # Transform to canonical frame
            canonical_x = -memo_z  # forward = -back
            canonical_y =  memo_x  # up = up
            canonical_z = -memo_y  # right = -left
            
            # Replace with transformed data
            df_out[gyro_x_col] = canonical_x
            df_out[gyro_y_col] = canonical_y
            df_out[gyro_z_col] = canonical_z
            
            print(f"  ✓ Transformed {segment} gyroscope from MeMo frame to canonical frame")
        else:
            if any([gyro_x_col, gyro_y_col, gyro_z_col]):
                print(f"  ⚠ Warning: Incomplete gyroscope axes for {segment} (found: {[gyro_x_col, gyro_y_col, gyro_z_col]})")
    
    return df_out


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Try reading CSV with common delimiters."""
    seps = [",", ";", "\t"]
    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
            if df.shape[1] > 1 or df.dropna(how="all").shape[0] > 0:
                return df
        except Exception:
            continue
    # Final fallback
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def convert_gyro_units(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Convert gyroscope units to radians per second if needed.

    Args:
        df: DataFrame with columns possibly containing gyro signals
        unit: 'deg' or 'rad' indicating current units of gyro in the DataFrame

    Returns:
        DataFrame with gyro signals in rad/s
    """
    if unit.lower() not in {"deg", "rad"}:
        return df

    if unit.lower() == "rad":
        return df  # no change needed

    df_out = df.copy()
    rad_per_deg = np.pi / 180.0
    # Convert any column with '_gyro_' substring
    gyro_cols = [c for c in df_out.columns if "_gyro_" in c.lower()]
    for c in gyro_cols:
        df_out[c] = pd.to_numeric(df_out[c], errors='coerce') * rad_per_deg
    if gyro_cols:
        print(f"  ✓ Converted {len(gyro_cols)} gyroscope column(s) from deg/s to rad/s")
    return df_out


def process_memo_trial(trial_dir: Path, output_root: Path, transform_gyro: bool = True, gyro_unit: str = "deg") -> bool:
    """Process a single MeMo trial.
    
    Args:
        trial_dir: Path to trial directory (e.g., AB01_Jimin/0mps/trial_1)
        output_root: Output root directory
        transform_gyro: Whether to apply gyroscope frame transformation
    
    Returns:
        True if successful, False otherwise
    """
    input_dir = trial_dir / "Input"
    label_dir = trial_dir / "Label"
    
    if not input_dir.exists() or not label_dir.exists():
        print(f"  ⚠ Skipping {trial_dir}: Missing Input or Label directory")
        return False
    
    # Parse trial path to get subject, condition, trial
    # Structure: Subject/Condition/Trial
    trial_name = trial_dir.name
    condition_name = trial_dir.parent.name
    subject_name = trial_dir.parent.parent.name
    
    # Create output directory structure
    out_trial_dir = output_root / subject_name / condition_name / trial_name
    out_input_dir = out_trial_dir / "Input"
    out_label_dir = out_trial_dir / "Label"
    out_input_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Process IMU data
    imu_file = input_dir / "imu_data.csv"
    if imu_file.exists():
        try:
            df_imu = read_csv_flexible(imu_file)
            
            # Standardize column names (snake_case, lowercase)
            df_imu = standardize_segment_names(df_imu)
            
            # Convert gyro units to rad/s if needed
            df_imu = convert_gyro_units(df_imu, unit=gyro_unit)
            
            # Transform gyroscope data from MeMo frame to canonical frame
            if transform_gyro:
                df_imu = transform_memo_to_canonical(df_imu)
            
            # Save processed IMU data
            df_imu.to_csv(out_input_dir / "imu_data.csv", index=False)
            print(f"  ✓ Processed IMU data")
        except Exception as e:
            print(f"  ✗ Error processing IMU data: {e}")
            return False
    else:
        print(f"  ⚠ No imu_data.csv found")
        return False
    
    # Process Label data (rename to joint_moment.csv and standardize)
    label_files = list(label_dir.glob("*.csv"))
    if label_files:
        try:
            # Use the first CSV file found (typically {subject}_{condition}_{trial}.csv)
            label_file = label_files[0]
            df_label = read_csv_flexible(label_file)
            
            # Standardize column names
            df_label = standardize_segment_names(df_label)
            
            # Convert moment units from N·mm/kg to N·m/kg (divide by 1000)
            # Identify moment columns (contain 'moment' in the name)
            moment_cols = [col for col in df_label.columns if 'moment' in col.lower()]
            for col in moment_cols:
                # Convert to numeric, handle any non-numeric values
                df_label[col] = pd.to_numeric(df_label[col], errors='coerce') / 1000.0
                print(f"    • Converted {col} from N·mm/kg to N·m/kg")

            # Map axes to Canonical_Camargo naming with signs per reference table (RIGHT hip):
            #   X axis: +Extension / -Flexion  → hip_flexion_*_moment must be positive for Flexion → multiply by -1
            #   Y axis: +Adduction / -Abduction → hip_adduction_*_moment = +source
            #   Z axis: +External / -Internal rotation → hip_rotation_*_moment = +source
            # Do the same for LEFT hip columns.
            def assign_if_present(src: str, dst: str, sign: float = 1.0):
                if src in df_label.columns and dst not in df_label.columns:
                    df_label[dst] = pd.to_numeric(df_label[src], errors='coerce') * float(sign)
                    s = "(flipped sign)" if sign == -1.0 else ""
                    print(f"    • Added {dst} from {src} {s}")

            # Right hip mappings
            # Flexion positive required -> flip sign if source is extension positive
            assign_if_present('rhipmoment_x', 'hip_flexion_r_moment')
            assign_if_present('rhipmoment_y', 'hip_adduction_r_moment')
            assign_if_present('rhipmoment_z', 'hip_rotation_r_moment')
            # Left hip mappings
            assign_if_present('lhipmoment_x', 'hip_adduction_l_moment')
            assign_if_present('lhipmoment_y', 'hip_flexion_l_moment')
            assign_if_present('lhipmoment_z', 'hip_rotation_l_moment')
            
            # Save as joint_moment.csv
            # Reference note: A prior implementation loaded MeMo Vicon hip moments by
            #   - selecting raw columns [54 (right), 6 (left)]
            #   - dividing by 1000 to convert N·mm/kg → N·m/kg
            #   - flipping the LEFT hip sign so that left/right share the same flexion/extension convention
            # We mirror that behavior here by ensuring the unit conversion above, and flipping
            # left hip flexion sign if present.
            if 'hip_flexion_l_moment' in df_label.columns:
                df_label['hip_flexion_l_moment'] = -pd.to_numeric(df_label['hip_flexion_l_moment'], errors='coerce')
                print("    • Adjusted left hip flexion sign to match right (per reference loader)")

            df_label.to_csv(out_label_dir / "joint_moment.csv", index=False)
            print(f"  ✓ Processed label data: {label_file.name} → joint_moment.csv")
        except Exception as e:
            print(f"  ✗ Error processing label data: {e}")
            return False
    else:
        print(f"  ⚠ No label CSV found")
        return False
    
    return True


def gather_memo_trials(input_root: Path, conditions: Optional[List[str]] = None) -> List[Path]:
    """Gather all trial directories from MeMo_processed structure.
    
    Args:
        input_root: Root directory of MeMo_processed
        conditions: Optional list of conditions to filter (e.g., ['0mps', '1p0mps'])
    
    Returns:
        List of trial directory paths
    """
    trials = []
    
    # Iterate through subjects
    for subject_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        # Skip README and other non-subject directories
        if subject_dir.name.startswith('.') or subject_dir.name == 'README.md':
            continue
        
        # Iterate through conditions
        for condition_dir in sorted([p for p in subject_dir.iterdir() if p.is_dir()]):
            # Filter by conditions if specified
            if conditions and condition_dir.name not in conditions:
                continue
            
            # Iterate through trials
            for trial_dir in sorted([p for p in condition_dir.iterdir() if p.is_dir()]):
                # Verify this is a valid trial directory (has Input and Label subdirs)
                if (trial_dir / "Input").exists() and (trial_dir / "Label").exists():
                    trials.append(trial_dir)
    
    return trials


def main():
    parser = argparse.ArgumentParser(
        description="Process MeMo_processed dataset into standardized Canonical format with coordinate frame transformation"
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Path to MeMo_processed root directory"
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Path to output Canonical directory"
    )
    parser.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated list of conditions to include (default: all conditions)"
    )
    parser.add_argument(
        "--gyro-unit",
        choices=["deg", "rad"],
        default="deg",
        help="Unit of gyroscope data in MeMo (default: deg/s). Will be converted to rad/s.")
    parser.add_argument(
        "--no-transform",
        action="store_true",
        help="Skip gyroscope frame transformation (keep original MeMo frame)"
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated list of subjects to process (default: all subjects)"
    )
    
    args = parser.parse_args()
    
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    
    if not input_root.exists():
        print(f"Error: Input directory not found: {input_root}")
        sys.exit(1)
    
    # Parse conditions filter
    conditions = None
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        print(f"Filtering conditions: {conditions}")
    
    # Parse subjects filter
    subjects_filter = None
    if args.subjects:
        subjects_filter = set(s.strip() for s in args.subjects.split(",") if s.strip())
        print(f"Filtering subjects: {subjects_filter}")
    
    # Gather all trials
    print(f"\nScanning {input_root} for trials...")
    all_trials = gather_memo_trials(input_root, conditions)
    
    # Apply subjects filter if specified
    if subjects_filter:
        all_trials = [t for t in all_trials if t.parent.parent.name in subjects_filter]
    
    if not all_trials:
        print("No trials found matching criteria.")
        sys.exit(1)
    
    print(f"Found {len(all_trials)} trial(s) to process.\n")
    
    # Print coordinate frame transformation info
    if not args.no_transform:
        print("Coordinate Frame Transformation:")
        print("  MeMo Frame:      [x=up, y=left, z=back]")
        print("  Canonical Frame: [x=forward, y=up, z=right]")
        print("  Transformation:  x_canonical = -z_memo")
        print("                   y_canonical =  x_memo")
        print("                   z_canonical = -y_memo")
        print()
    else:
        print("Coordinate frame transformation DISABLED (keeping original MeMo frame)\n")
    
    # Process each trial
    processed = 0
    failed = 0
    
    for trial_dir in all_trials:
        subject = trial_dir.parent.parent.name
        condition = trial_dir.parent.name
        trial = trial_dir.name
        
        print(f"Processing {subject}/{condition}/{trial}...")
        success = process_memo_trial(trial_dir, output_root, transform_gyro=not args.no_transform, gyro_unit=args.gyro_unit)
        
        if success:
            processed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  ✓ Successful: {processed} trials")
    if failed > 0:
        print(f"  ✗ Failed: {failed} trials")
    print(f"  Output directory: {output_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

