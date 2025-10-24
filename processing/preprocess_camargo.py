#!/usr/bin/env python3
"""
Preprocess Camargo dataset from raw format to standardized format.

This script processes raw Camargo data from /Volumes/Samsung_T5/raw_data/Samples/Camargo
and converts it to a standardized format with gyro and accel columns, similar to canonical_frame_converter.py
but without the canonical frame conversion step.

Camargo structure:
- Subject/Date/Condition/imu/trial.csv
- Subject/Date/Condition/id/trial.csv

Usage:
  python preprocess_camargo.py \
    --input-root /Volumes/Samsung_T5/raw_data/Samples/Camargo \
    --output-root /Users/luorix/Desktop/MetaMobility\ Lab\ \(CMU\)/data/Camargo_processed \
    [--conditions treadmill] \
    [--subjects AB01,AB02,AB03] \
    [--unit rad] \
    [--max-frames 40000]
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd


def read_csv_flexible(file_path: Path) -> Optional[pd.DataFrame]:
    """Read CSV file with flexible engine selection to handle NumPy compatibility issues."""
    try:
        # Try with default engine first
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"    Warning: Failed to read {file_path} with default engine: {e}")
        try:
            # Try with python engine
            return pd.read_csv(file_path, engine='python')
        except Exception as e2:
            print(f"    Error: Failed to read {file_path} with python engine: {e2}")
            return None


def standardize_segment_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize segment names to canonical format.
    
    Transforms:
    - foot_Gyro_X -> foot_r_gyro_x (unilateral, assume right side)
    - foot_Accel_X -> foot_r_accel_x (unilateral, assume right side)
    - shank_Gyro_X -> tibia_r_gyro_x (shank -> tibia, unilateral, assume right side)
    - shank_Accel_X -> tibia_r_accel_x (shank -> tibia, unilateral, assume right side)
    - thigh_Gyro_X -> femur_r_gyro_x (thigh -> femur, unilateral, assume right side)
    - thigh_Accel_X -> femur_r_accel_x (thigh -> femur, unilateral, assume right side)
    - trunk_Gyro_X -> pelvis_gyro_x (trunk -> pelvis, no side)
    - trunk_Accel_X -> pelvis_accel_x (trunk -> pelvis, no side)
    """
    df = df.copy()
    new_columns = []
    
    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        
        # Pattern: segment_Sensor_Axis -> segment_side_sensor_axis
        # Handles both gyro and accel
        pattern = r'\b(foot|shank|thigh|trunk)_(gyro|accel)_([xyz])(?:_|$)'
        match = re.search(pattern, col_lower)
        if match:
            segment, sensor, axis = match.groups()
            # Convert segment names to canonical format
            if segment == 'trunk':
                segment = 'pelvis'
                # Pelvis doesn't have sides
                new_col = f'{segment}_{sensor}_{axis}'
            elif segment == 'shank':
                segment = 'tibia'
                # Assume right side for unilateral IMUs
                new_col = f'{segment}_r_{sensor}_{axis}'
            elif segment == 'thigh':
                segment = 'femur'
                # Assume right side for unilateral IMUs
                new_col = f'{segment}_r_{sensor}_{axis}'
            else:
                # foot and others: assume right side for unilateral IMUs
                new_col = f'{segment}_r_{sensor}_{axis}'
        else:
            new_col = col_str
        
        new_columns.append(new_col)
    
    df.columns = new_columns
    return df


def extract_imu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract gyro and accel columns from IMU data."""
    gyro_cols = [col for col in df.columns if 'gyro' in col.lower()]
    accel_cols = [col for col in df.columns if 'accel' in col.lower()]
    
    # Always include time/header column if present
    time_cols = [col for col in df.columns if col.lower() in ['time', 'header']]
    
    selected_cols = time_cols + accel_cols + gyro_cols
    
    print(f"    Found {len(gyro_cols)} gyro columns: {gyro_cols}")
    print(f"    Found {len(accel_cols)} accel columns: {accel_cols}")
    print(f"    Total IMU columns: {len(selected_cols)}")
    
    return df[selected_cols]


def process_trial(imu_file: Path, id_file: Path, output_dir: Path, unit: str, max_frames: int, 
                 subject_weight: Optional[float] = None) -> bool:
    """Process a single trial and save standardized data.
    
    Args:
        imu_file: Path to IMU CSV file
        id_file: Path to ID (joint moment) CSV file
        output_dir: Output directory for processed data
        unit: Unit of gyro data ('rad' or 'deg')
        max_frames: Maximum number of frames to process
        subject_weight: Subject body weight in kg (for normalizing moments)
    """
    try:
        # Read IMU data
        imu_df = read_csv_flexible(imu_file)
        if imu_df is None:
            return False
        
        # Extract gyro and accel columns
        imu_df = extract_imu_columns(imu_df)
        
        # Rename 'Header' to 'time' if present
        if 'Header' in imu_df.columns:
            imu_df = imu_df.rename(columns={'Header': 'time'})
        elif 'header' in imu_df.columns:
            imu_df = imu_df.rename(columns={'header': 'time'})
        # Ensure time is the first column if present
        if 'time' in imu_df.columns:
            cols_order = ['time'] + [c for c in imu_df.columns if c != 'time']
            imu_df = imu_df[cols_order]
        
        # Standardize column names
        imu_df = standardize_segment_names(imu_df)
        
        # Apply unit conversion for gyro data if needed
        if unit == "deg":
            gyro_cols = [col for col in imu_df.columns if 'gyro' in col.lower()]
            for col in gyro_cols:
                imu_df[col] = imu_df[col] * (np.pi / 180.0)
        elif unit == "rad":
            pass
        
        # Limit frames if specified
        if max_frames > 0 and len(imu_df) > max_frames:
            imu_df = imu_df.iloc[:max_frames]
        
        # Read ID data (joint moments)
        id_df = read_csv_flexible(id_file)
        if id_df is None:
            return False
            
        # Rename 'Header' to 'time' if present
        if 'Header' in id_df.columns:
            id_df = id_df.rename(columns={'Header': 'time'})
        elif 'header' in id_df.columns:
            id_df = id_df.rename(columns={'header': 'time'})
        # If time not present in labels but present in IMU and same length, copy over
        if 'time' not in id_df.columns and 'time' in imu_df.columns and len(id_df) == len(imu_df):
            id_df.insert(0, 'time', imu_df['time'].values)
        # Ensure time is the first column if present
        if 'time' in id_df.columns:
            cols_order_lbl = ['time'] + [c for c in id_df.columns if c != 'time']
            id_df = id_df[cols_order_lbl]
        
        # Standardize joint moment column names
        id_df = standardize_joint_moment_names(id_df)
        
        # Normalize joint moments by body weight if provided
        if subject_weight is not None and subject_weight > 0:
            moment_cols = [col for col in id_df.columns if 'moment' in col.lower()]
            for col in moment_cols:
                id_df[col] = id_df[col] / subject_weight
            print(f"    Normalized {len(moment_cols)} moment columns by weight: {subject_weight:.2f} kg")
        
        # Create output directories
        input_dir = output_dir / "Input"
        label_dir = output_dir / "Label"
        input_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        imu_output = input_dir / "imu_data.csv"
        label_output = label_dir / "joint_moment.csv"
        
        imu_df.to_csv(imu_output, index=False)
        id_df.to_csv(label_output, index=False)
        
        print(f"  Processed: {imu_file.name} -> {output_dir.name}")
        return True
        
    except Exception as e:
        print(f"  Error processing {imu_file}: {e}")
        return False


def load_subject_info(dataset_root: Path) -> Dict[str, float]:
    """Load SubjectInfo.csv and return a dict mapping subject ID to body weight.
    
    Args:
        dataset_root: Path to dataset root (should contain SubjectInfo.csv)
    
    Returns:
        Dict mapping subject ID (e.g., 'AB21') to body weight in kg
    """
    subject_info_path = dataset_root / "SubjectInfo.csv"
    if not subject_info_path.exists():
        print(f"⚠️  Warning: SubjectInfo.csv not found at {subject_info_path}")
        print("   Joint moments will NOT be normalized by body weight!")
        return {}
    
    try:
        df = pd.read_csv(subject_info_path)
        
        # Find subject and weight columns (case-insensitive)
        columns_lower = {col.lower(): col for col in df.columns}
        
        subject_col = None
        for candidate in ['subject', 'id', 'participant']:
            if candidate in columns_lower:
                subject_col = columns_lower[candidate]
                break
        
        weight_col = None
        for candidate in ['weight', 'mass', 'body_mass', 'bodyweight']:
            if candidate in columns_lower:
                weight_col = columns_lower[candidate]
                break
        
        if subject_col is None or weight_col is None:
            print(f"⚠️  Warning: Could not find Subject or Weight columns in {subject_info_path}")
            print(f"   Available columns: {list(df.columns)}")
            return {}
        
        # Create mapping
        subject_weights = {}
        for _, row in df.iterrows():
            subject_id = str(row[subject_col]).strip()
            weight = float(row[weight_col])
            subject_weights[subject_id] = weight
        
        print(f"✅ Loaded subject info for {len(subject_weights)} subjects")
        return subject_weights
        
    except Exception as e:
        print(f"⚠️  Error loading SubjectInfo.csv: {e}")
        return {}


def standardize_joint_moment_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize joint moment column names to canonical format.
    
    Camargo joint moment names are already in the correct format:
    - hip_flexion_r_moment -> hip_flexion_r_moment (already correct)
    - hip_flexion_l_moment -> hip_flexion_l_moment (already correct)
    - knee_angle_r_moment -> knee_angle_r_moment (already correct)
    - knee_angle_l_moment -> knee_angle_l_moment (already correct)
    - ankle_angle_r_moment -> ankle_angle_r_moment (already correct)
    - ankle_angle_l_moment -> ankle_angle_l_moment (already correct)
    """
    # Camargo joint moment names are already in the correct format
    return df


def process_camargo_dataset(input_root: str, output_root: str, conditions: List[str], 
                          unit: str, max_frames: int, subjects: Optional[List[str]] = None):
    """Process the entire Camargo dataset."""
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_root}")
    
    # Load subject information (body weights) from source dataset
    subject_weights = load_subject_info(input_path)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy SubjectInfo.csv from source to target if it exists
    source_subject_info = input_path / "SubjectInfo.csv"
    target_subject_info = output_path / "SubjectInfo.csv"
    if source_subject_info.exists():
        shutil.copy2(source_subject_info, target_subject_info)
        print(f"📋 Copied SubjectInfo.csv to output directory")
    else:
        print(f"⚠️  Warning: SubjectInfo.csv not found in source directory: {source_subject_info}")
    
    # Print processing info
    if subjects is not None:
        print(f"🎯 Processing selected subjects: {', '.join(subjects)}")
    else:
        print("🔄 Processing all subjects")
    
    processed_count = 0
    total_trials = 0
    found_subjects = set()
    
    # Process each subject
    for subject_dir in input_path.iterdir():
        if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
            continue
            
        subject_name = subject_dir.name
        
        # Skip subjects not in the selected list if subjects filter is provided
        if subjects is not None and subject_name not in subjects:
            continue
        
        # Track found subjects
        if subjects is not None:
            found_subjects.add(subject_name)
            
        subject_weight = subject_weights.get(subject_name, None)
        
        if subject_weight is None:
            print(f"⚠️  Processing subject: {subject_name} (no weight info - moments will not be normalized)")
        else:
            print(f"Processing subject: {subject_name} (weight: {subject_weight:.2f} kg)")
        
        # Create subject directory in output
        subject_out = output_path / subject_name
        subject_out.mkdir(exist_ok=True)
        
        # Create opensim directory for this subject
        opensim_out = subject_out / "opensim"
        opensim_out.mkdir(exist_ok=True)
        
        # Find and copy .osim files for this subject
        osim_files_found = []
        for date_dir in subject_dir.iterdir():
            if not date_dir.is_dir() or date_dir.name.startswith('.'):
                continue
            # Look for .osim files in any subdirectory
            for osim_file in date_dir.rglob("*.osim"):
                if osim_file.is_file():
                    osim_files_found.append(osim_file)
        
        # Copy .osim files to opensim directory
        for osim_file in osim_files_found:
            target_osim = opensim_out / f"{subject_name}.osim"
            try:
                shutil.copy2(osim_file, target_osim)
                print(f"    Copied .osim file: {osim_file.name} -> {target_osim.name}")
            except Exception as e:
                print(f"    Warning: Failed to copy {osim_file.name}: {e}")
        
        if not osim_files_found:
            print(f"    Warning: No .osim files found for subject {subject_name}")
        
        # Process each date directory
        for date_dir in subject_dir.iterdir():
            if not date_dir.is_dir() or date_dir.name.startswith('.'):
                continue
                
            # Process each condition directory
            for condition_dir in date_dir.iterdir():
                if not condition_dir.is_dir() or condition_dir.name.startswith('.'):
                    continue
                    
                condition_name = condition_dir.name
                if condition_name not in conditions:
                    continue
                    
                print(f"  Processing condition: {condition_name}")
                
                # Create condition directory in output
                condition_out = subject_out / condition_name
                condition_out.mkdir(exist_ok=True)
                
                # Get IMU and ID files
                imu_dir = condition_dir / "imu"
                id_dir = condition_dir / "id"
                
                if not imu_dir.exists() or not id_dir.exists():
                    print(f"    Warning: Missing imu or id directory in {condition_dir}")
                    continue
                
                # Process each trial
                for imu_file in imu_dir.iterdir():
                    if not imu_file.is_file() or not imu_file.suffix == '.csv' or imu_file.name.startswith('.'):
                        continue
                    
                    # Find corresponding ID file
                    id_file = id_dir / imu_file.name
                    if not id_file.exists():
                        print(f"    Warning: No corresponding ID file for {imu_file.name}")
                        continue
                    
                    # Create trial output directory
                    trial_name = imu_file.stem
                    trial_out = condition_out / trial_name
                    trial_out.mkdir(exist_ok=True)
                    
                    total_trials += 1
                    if process_trial(imu_file, id_file, trial_out, unit, max_frames, subject_weight):
                        processed_count += 1
    
    # Check for missing subjects if filter was applied
    if subjects is not None:
        missing_subjects = set(subjects) - found_subjects
        if missing_subjects:
            print(f"\n⚠️  Warning: The following subjects were not found: {', '.join(missing_subjects)}")
    
    print(f"\nCompleted! Processed {processed_count}/{total_trials} trials successfully.")
    print(f"Output directory: {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Camargo dataset")
    parser.add_argument("--input-root", required=True, 
                       help="Path to raw Camargo data directory")
    parser.add_argument("--output-root", required=True, 
                       help="Path to output processed directory")
    parser.add_argument("--conditions", default="treadmill", 
                       help="Comma-separated list of conditions to process")
    parser.add_argument("--unit", choices=["rad", "deg"], default="rad", 
                       help="Unit of IMU gyro in source CSV")
    parser.add_argument("--max-frames", type=int, default=40000, 
                       help="Maximum number of frames to process per trial")
    parser.add_argument("--subjects", 
                       help="Comma-separated list of subjects to process (e.g., AB01,AB02,AB03). If not provided, processes all subjects.")
    
    args = parser.parse_args()
    
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
    
    process_camargo_dataset(
        args.input_root, 
        args.output_root, 
        conditions, 
        args.unit, 
        args.max_frames,
        subjects
    )


if __name__ == "__main__":
    main()