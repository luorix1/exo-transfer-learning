#!/usr/bin/env python3
"""
Canonicalize MetaMobility (MeMo) IMU data to OpenSim reference frame.

This script transforms IMU gyroscope data from MetaMobility sensor frames
to OpenSim canonical frames (x:forward, y:up, z:right for each body segment).

Sensor Frame Orientations:
    - Right Thigh IMU: x:up, y:left, z:back
    - Left Thigh IMU:  x:up, y:left, z:back
    - Pelvis IMU:      x:up, y:right, z:forward

OpenSim Canonical Frame (for all segments):
    - x: forward (direction of motion)
    - y: up (superior direction)
    - z: right (lateral direction)

Transformation Matrices:
    
    Right Thigh: (x:up, y:left, z:back) → (x:forward, y:up, z:right)
        canonical_x (forward) = -sensor_z  (flip back to forward)
        canonical_y (up)      =  sensor_x  (up stays up)
        canonical_z (right)   = -sensor_y  (flip left to right)
    
    Left Thigh: (x:up, y:left, z:back) → (x:forward, y:up, z:right)
        canonical_x (forward) = -sensor_z  (flip back to forward)
        canonical_y (up)      =  sensor_x  (up stays up)
        canonical_z (right)   = -sensor_y  (flip left to right)
    
    Pelvis: (x:up, y:right, z:forward) → (x:forward, y:up, z:right)
        canonical_x (forward) =  sensor_z  (forward stays forward)
        canonical_y (up)      =  sensor_x  (up stays up)
        canonical_z (right)   =  sensor_y  (right stays right)

Usage:
    python processing/canonicalize_memo.py \
        --input-root "/path/to/Final/MetaMobility" \
        --output-root "/path/to/Canonical_Memo" \
        --conditions 1p2mps,transient_30sec \
        --subjects AB01,AB02
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import shutil


def transform_thigh_gyro(gyro_data: np.ndarray) -> np.ndarray:
    """
    Transform thigh IMU data from sensor frame to canonical frame.
    
    Sensor frame: x:up, y:left, z:back
    Canonical frame: x:forward, y:up, z:right
    
    Args:
        gyro_data: Nx3 array [sensor_x, sensor_y, sensor_z]
    
    Returns:
        Nx3 array [canonical_x, canonical_y, canonical_z]
    """
    sensor_x = gyro_data[:, 0]  # up
    sensor_y = gyro_data[:, 1]  # left
    sensor_z = gyro_data[:, 2]  # back
    
    canonical_x = -sensor_z  # forward = -back
    canonical_y = sensor_x   # up = up
    canonical_z = -sensor_y  # right = -left
    
    return np.column_stack([canonical_x, canonical_y, canonical_z])


def transform_pelvis_gyro(gyro_data: np.ndarray) -> np.ndarray:
    """
    Transform pelvis IMU data from sensor frame to canonical frame.
    
    Sensor frame: x:up, y:right, z:forward
    Canonical frame: x:forward, y:up, z:right
    
    Args:
        gyro_data: Nx3 array [sensor_x, sensor_y, sensor_z]
    
    Returns:
        Nx3 array [canonical_x, canonical_y, canonical_z]
    """
    sensor_x = gyro_data[:, 0]  # up
    sensor_y = gyro_data[:, 1]  # right
    sensor_z = gyro_data[:, 2]  # forward
    
    canonical_x = sensor_z   # forward = forward
    canonical_y = sensor_x   # up = up
    canonical_z = sensor_y   # right = right
    
    return np.column_stack([canonical_x, canonical_y, canonical_z])


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names from MetaMobility format to lowercase with underscores.
    
    Handles both formats:
        - Pelvis_Gyr_X -> pelvis_gyro_x
        - pelvis_gyro_x -> pelvis_gyro_x (already standardized)
    """
    rename_map = {}
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'time':
            rename_map[col] = 'time'
        # Pelvis gyro
        elif 'pelvis' in col_lower and ('gyr_x' in col_lower or 'gyro_x' in col_lower):
            rename_map[col] = 'pelvis_gyro_x'
        elif 'pelvis' in col_lower and ('gyr_y' in col_lower or 'gyro_y' in col_lower):
            rename_map[col] = 'pelvis_gyro_y'
        elif 'pelvis' in col_lower and ('gyr_z' in col_lower or 'gyro_z' in col_lower):
            rename_map[col] = 'pelvis_gyro_z'
        # Left thigh gyro
        elif 'thigh_l' in col_lower and ('gyr_x' in col_lower or 'gyro_x' in col_lower):
            rename_map[col] = 'thigh_l_gyro_x'
        elif 'thigh_l' in col_lower and ('gyr_y' in col_lower or 'gyro_y' in col_lower):
            rename_map[col] = 'thigh_l_gyro_y'
        elif 'thigh_l' in col_lower and ('gyr_z' in col_lower or 'gyro_z' in col_lower):
            rename_map[col] = 'thigh_l_gyro_z'
        # Right thigh gyro
        elif 'thigh_r' in col_lower and ('gyr_x' in col_lower or 'gyro_x' in col_lower):
            rename_map[col] = 'thigh_r_gyro_x'
        elif 'thigh_r' in col_lower and ('gyr_y' in col_lower or 'gyro_y' in col_lower):
            rename_map[col] = 'thigh_r_gyro_y'
        elif 'thigh_r' in col_lower and ('gyr_z' in col_lower or 'gyro_z' in col_lower):
            rename_map[col] = 'thigh_r_gyro_z'
    
    return df.rename(columns=rename_map)


def extract_gyro_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only time and gyroscope columns"""
    gyro_cols = ['time']
    gyro_cols.extend([col for col in df.columns if 'gyro' in col.lower()])
    
    return df[gyro_cols].copy()


def extract_subject_id(subject_name: str) -> str:
    """
    Extract subject ID from full name.
    E.g., 'AB01_Jimin' -> 'AB01'
    """
    return subject_name.split('_')[0]


def canonicalize_imu_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize all IMU gyro data in the dataframe.
    
    Args:
        df: DataFrame with columns like thigh_l_gyro_x, pelvis_gyro_y, etc.
    
    Returns:
        DataFrame with canonicalized gyro data (only gyro columns + time)
    """
    # First standardize column names
    df = standardize_column_names(df)
    
    # Extract only gyro columns
    df = extract_gyro_columns(df)
    
    df_canonical = df.copy()
    
    # Transform right thigh
    if all(col in df.columns for col in ['thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z']):
        thigh_r_data = df[['thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z']].values
        thigh_r_canonical = transform_thigh_gyro(thigh_r_data)
        df_canonical['thigh_r_gyro_x'] = thigh_r_canonical[:, 0]
        df_canonical['thigh_r_gyro_y'] = thigh_r_canonical[:, 1]
        df_canonical['thigh_r_gyro_z'] = thigh_r_canonical[:, 2]
        print("    ✓ Transformed right thigh gyro")
    
    # Transform left thigh
    if all(col in df.columns for col in ['thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z']):
        thigh_l_data = df[['thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z']].values
        thigh_l_canonical = transform_thigh_gyro(thigh_l_data)
        df_canonical['thigh_l_gyro_x'] = thigh_l_canonical[:, 0]
        df_canonical['thigh_l_gyro_y'] = thigh_l_canonical[:, 1]
        df_canonical['thigh_l_gyro_z'] = thigh_l_canonical[:, 2]
        print("    ✓ Transformed left thigh gyro")
    
    # Transform pelvis
    if all(col in df.columns for col in ['pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z']):
        pelvis_data = df[['pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z']].values
        pelvis_canonical = transform_pelvis_gyro(pelvis_data)
        df_canonical['pelvis_gyro_x'] = pelvis_canonical[:, 0]
        df_canonical['pelvis_gyro_y'] = pelvis_canonical[:, 1]
        df_canonical['pelvis_gyro_z'] = pelvis_canonical[:, 2]
        print("    ✓ Transformed pelvis gyro")
    
    return df_canonical


def process_trial(
    input_trial_dir: Path,
    output_trial_dir: Path,
    subject: str,
    condition: str,
    trial: str
) -> bool:
    """Process a single trial"""
    try:
        # Check if IMU data exists
        imu_file = input_trial_dir / "Input" / "imu_data.csv"
        if not imu_file.exists():
            print(f"  ⚠️  IMU file not found: {imu_file}")
            return False
        
        # Load IMU data
        imu_df = pd.read_csv(imu_file)
        
        # Canonicalize gyro data
        print(f"  🔄 Processing: {subject}/{condition}/{trial}")
        imu_canonical = canonicalize_imu_data(imu_df)
        
        # Copy entire trial structure
        output_trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy Label directory (unchanged)
        label_src = input_trial_dir / "Label"
        if label_src.exists():
            label_dst = output_trial_dir / "Label"
            if label_dst.exists():
                shutil.rmtree(label_dst)
            shutil.copytree(label_src, label_dst)
        
        # Copy opensim directory if it exists (unchanged)
        opensim_src = input_trial_dir / "opensim"
        if opensim_src.exists():
            opensim_dst = output_trial_dir / "opensim"
            if opensim_dst.exists():
                shutil.rmtree(opensim_dst)
            shutil.copytree(opensim_src, opensim_dst)
        
        # Save canonicalized IMU data
        input_dir = output_trial_dir / "Input"
        input_dir.mkdir(exist_ok=True)
        imu_canonical.to_csv(input_dir / "imu_data.csv", index=False)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {subject}/{condition}/{trial}: {e}")
        import traceback
        traceback.print_exc()
        return False


def canonicalize_dataset(
    input_root: Path,
    output_root: Path,
    conditions: Optional[List[str]] = None,
    subjects: Optional[List[str]] = None
) -> None:
    """Canonicalize entire MeMo dataset"""
    
    print(f"📁 Input root: {input_root}")
    print(f"📁 Output root: {output_root}")
    print(f"🎯 Conditions: {conditions if conditions else 'all'}")
    print(f"👥 Subjects: {subjects if subjects else 'all'}")
    print("=" * 70)
    print("\n📋 Frame Transformations:")
    print("   Right/Left Thigh: (x:up, y:left, z:back) → (x:forward, y:up, z:right)")
    print("   Pelvis:          (x:up, y:right, z:forward) → (x:forward, y:up, z:right)")
    print("=" * 70)
    
    total_trials = 0
    successful_trials = 0
    
    # Copy opensim directory for each subject if it exists
    for subject_dir in sorted(input_root.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
            continue
        
        subject_full_name = subject_dir.name
        subject_id = extract_subject_id(subject_full_name)
        
        # Filter by subjects
        if subjects and subject_id not in subjects:
            continue
        
        # Copy subject-level opensim directory if it exists
        opensim_src = subject_dir / "opensim"
        if opensim_src.exists():
            opensim_dst = output_root / subject_id / "opensim"
            opensim_dst.parent.mkdir(parents=True, exist_ok=True)
            if opensim_dst.exists():
                shutil.rmtree(opensim_dst)
            shutil.copytree(opensim_src, opensim_dst)
            print(f"\n👤 {subject_full_name} (ID: {subject_id}): Copied OpenSim models")
    
    # Process trials
    for subject_dir in sorted(input_root.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
            continue
        
        subject_full_name = subject_dir.name
        subject_id = extract_subject_id(subject_full_name)
        
        # Filter by subjects
        if subjects and subject_id not in subjects:
            continue
        
        print(f"\n👤 Processing subject: {subject_full_name} (ID: {subject_id})")
        
        # Iterate through conditions
        for condition_dir in sorted(subject_dir.iterdir()):
            if not condition_dir.is_dir() or condition_dir.name.startswith('.'):
                continue
            
            condition = condition_dir.name
            
            # Skip opensim directory
            if condition == 'opensim':
                continue
            
            # Filter by conditions
            if conditions and condition not in conditions:
                continue
            
            print(f"  📂 Condition: {condition}")
            
            # Iterate through trials
            for trial_dir in sorted(condition_dir.iterdir()):
                if not trial_dir.is_dir() or trial_dir.name.startswith('.'):
                    continue
                
                trial = trial_dir.name
                total_trials += 1
                
                # Create output path
                output_trial_dir = output_root / subject_id / condition / trial
                
                if process_trial(
                    trial_dir,
                    output_trial_dir,
                    subject_full_name,
                    condition,
                    trial
                ):
                    successful_trials += 1
    
    print(f"\n{'='*70}")
    print(f"✅ Canonicalization complete!")
    print(f"   Total trials: {total_trials}")
    print(f"   Successful: {successful_trials}")
    print(f"   Failed: {total_trials - successful_trials}")
    print(f"\n📊 Output structure:")
    print(f"   {output_root}/")
    print(f"   └── SubjectID/")
    print(f"       ├── opensim/          # OpenSim models (if available)")
    print(f"       └── condition/")
    print(f"           └── trial_N/")
    print(f"               ├── Input/")
    print(f"               │   └── imu_data.csv  # Canonicalized gyro data")
    print(f"               └── Label/")
    print(f"                   └── joint_moment.csv")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Canonicalize MetaMobility IMU data to OpenSim reference frame"
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Path to input Memo dataset (e.g., Final/Memo)"
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Path to output canonical dataset (e.g., Canonical_Memo)"
    )
    parser.add_argument(
        "--conditions",
        help="Comma-separated list of conditions to process (default: all)"
    )
    parser.add_argument(
        "--subjects",
        help="Comma-separated list of subject IDs to process (default: all)"
    )
    
    args = parser.parse_args()
    
    # Parse conditions and subjects
    conditions = [c.strip() for c in args.conditions.split(',')] if args.conditions else None
    subjects = [s.strip() for s in args.subjects.split(',')] if args.subjects else None
    
    # Convert paths
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    
    if not input_root.exists():
        print(f"❌ Input root does not exist: {input_root}")
        return
    
    # Canonicalize dataset
    canonicalize_dataset(
        input_root,
        output_root,
        conditions,
        subjects
    )


if __name__ == "__main__":
    main()

