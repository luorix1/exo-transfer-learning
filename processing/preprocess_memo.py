#!/usr/bin/env python3
"""
Preprocess MetaMobility dataset from raw format to standardized format.

This script processes raw MetaMobility data and converts it to a standardized format
with gyro and accel columns, joint moments, and OpenSim motion files.

MetaMobility structure:
- Subject/Condition/Trial/Input/imu_data.csv
- Subject/Condition/Trial/Label/{subject}_{condition}_{trial}.csv

Usage:
  python preprocess_memo.py \
    --input-root /Users/luorix/Desktop/MetaMobility\ Lab\ \(CMU\)/data/MetaMobility \
    --output-root /Users/luorix/Desktop/MetaMobility\ Lab\ \(CMU\)/data/Final/MetaMobility \
    [--conditions 0mps,1p0mps,1p2mps] \
    [--subjects AB01,AB02,AB03] \
    [--unit deg] \
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
    - Pelvis_Gyr_X -> pelvis_gyro_x
    - Pelvis_Acc_X -> pelvis_accel_x
    - Thigh_R_Gyr_X -> femur_r_gyro_x (thigh -> femur, add side)
    - Thigh_R_Acc_X -> femur_r_accel_x (thigh -> femur, add side)
    - Thigh_L_Gyr_X -> femur_l_gyro_x (thigh -> femur, add side)
    - Thigh_L_Acc_X -> femur_l_accel_x (thigh -> femur, add side)
    - Shank_R_Gyr_X -> tibia_r_gyro_x (shank -> tibia, add side)
    - Shank_R_Acc_X -> tibia_r_accel_x (shank -> tibia, add side)
    - Shank_L_Gyr_X -> tibia_l_gyro_x (shank -> tibia, add side)
    - Shank_L_Acc_X -> tibia_l_accel_x (shank -> tibia, add side)
    """
    df = df.copy()
    new_columns = []
    
    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        
        # Pattern: Segment_Side_Sensor_Axis -> segment_side_sensor_axis
        # Handle both gyro and accel
        pattern = r'\b(pelvis|thigh|shank)_([rl]?)_?(gyr|accel|acc)_([xyz])(?:_|$)'
        match = re.search(pattern, col_lower)
        if match:
            segment, side, sensor, axis = match.groups()
            
            # Convert segment names to canonical format
            if segment == 'thigh':
                segment = 'femur'
            elif segment == 'shank':
                segment = 'tibia'
            
            # Standardize sensor names
            if sensor in ['gyr']:
                sensor = 'gyro'
            elif sensor in ['accel', 'acc']:
                sensor = 'accel'
            
            # Handle side information
            if side and side in ['r', 'l']:
                new_col = f'{segment}_{side}_{sensor}_{axis}'
            else:
                # No side specified, assume right for bilateral segments
                if segment in ['femur', 'tibia']:
                    new_col = f'{segment}_r_{sensor}_{axis}'
                else:
                    new_col = f'{segment}_{sensor}_{axis}'
        else:
            new_col = col_str
        
        new_columns.append(new_col)
    
    df.columns = new_columns
    return df


def extract_imu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract gyro and accel columns from IMU data."""
    gyro_cols = [col for col in df.columns if 'gyro' in col.lower() or 'gyr' in col.lower()]
    accel_cols = [col for col in df.columns if 'accel' in col.lower() or 'acc' in col.lower()]
    
    # Always include time/header column if present
    time_cols = [col for col in df.columns if col.lower() in ['time', 'header', 'frame']]
    
    selected_cols = time_cols + accel_cols + gyro_cols
    
    print(f"    Found {len(gyro_cols)} gyro columns: {gyro_cols}")
    print(f"    Found {len(accel_cols)} accel columns: {accel_cols}")
    print(f"    Total IMU columns: {len(selected_cols)}")
    
    return df[selected_cols]


def process_trial(imu_file: Path, label_file: Path, output_dir: Path, unit: str, max_frames: int, 
                 subject_weight: Optional[float] = None) -> bool:
    """Process a single MetaMobility trial and save standardized data.
    
    Args:
        imu_file: Path to IMU CSV file
        label_file: Path to Label CSV file
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
        
        # Rename 'Frame' to 'time' if present
        if 'Frame' in imu_df.columns:
            imu_df = imu_df.rename(columns={'Frame': 'time'})
        elif 'frame' in imu_df.columns:
            imu_df = imu_df.rename(columns={'frame': 'time'})
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
        
        # Read Label data (joint moments)
        label_df = read_csv_flexible(label_file)
        if label_df is None:
            return False
            
        # Rename 'Frame' to 'time' if present
        if 'Frame' in label_df.columns:
            label_df = label_df.rename(columns={'Frame': 'time'})
        elif 'frame' in label_df.columns:
            label_df = label_df.rename(columns={'frame': 'time'})
        # If time not present in labels but present in IMU and same length, copy over
        if 'time' not in label_df.columns and 'time' in imu_df.columns and len(label_df) == len(imu_df):
            label_df.insert(0, 'time', imu_df['time'].values)
        # Ensure time is the first column if present
        if 'time' in label_df.columns:
            cols_order_lbl = ['time'] + [c for c in label_df.columns if c != 'time']
            label_df = label_df[cols_order_lbl]
        
        # Standardize joint moment column names
        label_df = standardize_joint_moment_names(label_df)
        
        # Convert moment units from N-mm/kg to N-m/kg (divide by 1000)
        moment_cols = [col for col in label_df.columns if 'moment' in col.lower()]
        for col in moment_cols:
            label_df[col] = label_df[col] / 1000.0
        print(f"    Converted {len(moment_cols)} moment columns from N-mm/kg to N-m/kg")
        
        # Separate joint angles from moments for different outputs
        # Get all columns that are not moments or time
        angle_cols = [col for col in label_df.columns if col != 'time' and 'moment' not in col.lower()]
        
        if angle_cols:
            print(f"    Found joint angle columns: {angle_cols}")
        else:
            print(f"    No joint angle columns found in label data")
        
        # For joint_moment.csv, keep only time and moment columns
        moment_columns_to_keep = ['time'] + moment_cols
        moment_df = label_df[moment_columns_to_keep]
        print(f"    Kept only moment columns in joint_moment.csv: {moment_cols}")
        
        # For motion.sto file, we need the full label data (angles + moments)
        full_label_df = label_df.copy()
        
        # Normalize joint moments by body weight if provided
        if subject_weight is not None and subject_weight > 0:
            for col in moment_cols:
                moment_df[col] = moment_df[col] / subject_weight
                full_label_df[col] = full_label_df[col] / subject_weight
            print(f"    Normalized {len(moment_cols)} moment columns by weight: {subject_weight:.2f} kg")
        
        # Create output directories
        input_dir = output_dir / "Input"
        label_dir = output_dir / "Label"
        opensim_dir = output_dir / "opensim"
        input_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        opensim_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        imu_output = input_dir / "imu_data.csv"
        label_output = label_dir / "joint_moment.csv"
        
        imu_df.to_csv(imu_output, index=False)
        moment_df.to_csv(label_output, index=False)
        
        # Generate motion.sto file for OpenSim using full label data (angles + moments)
        motion_sto_path = opensim_dir / "motion.sto"
        success = create_motion_sto_file(imu_df, full_label_df, motion_sto_path)
        
        if success:
            print(f"  Processed: {imu_file.name} -> {output_dir.name}")
            return True
        else:
            print(f"  Warning: Failed to create motion.sto for {imu_file.name}")
            return True  # Still return True as main processing succeeded
        
    except Exception as e:
        print(f"  Error processing {imu_file}: {e}")
        return False


def create_motion_sto_file(imu_df: pd.DataFrame, label_df: pd.DataFrame, output_path: Path) -> bool:
    """Create a single motion.sto file combining IMU and label data in OpenSim format."""
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the proper OpenSim motion file format
        with open(output_path, 'w') as f:
            # Write header in the correct format
            f.write("DataType=double\n")
            f.write("version=3\n")
            f.write("OpenSimVersion=4.5\n")
            f.write("endheader\n")
            
            # Define the exact column order to match OpenSim format
            # time, coordinates, coordinate_velocities, forces_moments
            
            # Get coordinate columns from label data (joint angles, excluding time and moment columns)
            coordinate_columns = []
            moment_columns = []
            
            for col in label_df.columns:
                if col == 'time':
                    continue
                elif 'moment' in col.lower():
                    moment_columns.append(col)
                else:
                    # This is a joint angle coordinate
                    coordinate_columns.append(col)
            
            # Debug: Print what coordinate columns we found
            print(f"    Found coordinate columns: {coordinate_columns}")
            
            # Check if we have any joint angle data
            if not coordinate_columns:
                print("    ⚠️  No joint angle data found - MetaMobility only contains moment data")
                print("    ⚠️  Skipping motion.sto file generation (no joint angles available)")
                return True  # Return success but skip .sto file generation
            
            # Map MetaMobility joint angles to OpenSim coordinate names
            opensim_coordinates = map_to_opensim_coordinates(coordinate_columns)
            print(f"    Mapped to OpenSim coordinates: {opensim_coordinates}")
            
            # Add missing coordinates that are typically in OpenSim models but might not be in MetaMobility data
            # These will be filled with default values (0.0)
            missing_coordinates = []
            all_expected_coords = [
                'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
                'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 
                'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 
                'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', 
                'lumbar_extension', 'lumbar_bending', 'lumbar_rotation'
            ]
            
            for coord in all_expected_coords:
                if coord not in opensim_coordinates:
                    missing_coordinates.append(coord)
            
            print(f"    Missing coordinates (will use 0.0): {missing_coordinates}")
            
            # Combine found and missing coordinates
            all_coordinates = opensim_coordinates + missing_coordinates
            
            # Create the column header in the exact order
            column_names = ['time']

            # Add coordinate names (OpenSim format)
            column_names.extend(all_coordinates)

            # Add coordinate velocity names (with _u suffix)
            column_names.extend([col + '_u' for col in all_coordinates])

            # Add force/moment names
            column_names.extend(moment_columns)

            # Precompute: convert joint angles to correct units (rad for rotations)
            df_pos = label_df.copy()
            translational_coords = {'pelvis_tx', 'pelvis_ty', 'pelvis_tz'}
            for col in coordinate_columns:
                if col not in translational_coords:
                    # Convert degrees to radians for rotational coordinates
                    df_pos[col] = np.radians(df_pos[col].astype(float))
                else:
                    # Keep translational coordinates as-is (already in meters)
                    df_pos[col] = df_pos[col].astype(float)
            
            # Apply sign flips for OpenSim conventions
            # OpenSim knee flexion is negative, but MetaMobility might be positive
            for col in coordinate_columns:
                if 'KneeAngles_X' in col:  # Knee flexion
                    df_pos[col] = -df_pos[col]  # Flip sign for knee flexion
                    print(f"    Flipped sign for {col} (knee flexion - OpenSim convention)")
            
            # Zero out all pelvis angles to assume rigid upper body
            for col in coordinate_columns:
                if 'PelvisAngles' in col or 'pelvis_' in col:
                    df_pos[col] = 0.0  # Set all pelvis angles to zero
                    # Also zero out in the original label_df to ensure it's reflected in output
                    label_df[col] = 0.0
                    print(f"    Zeroed {col} (rigid upper body assumption)")
            
            # Zero out lumbar rotation to assume rigid spine
            # Note: lumbar_rotation is mapped from RightSpineAngles_Z/LeftSpineAngles_Z
            # We'll handle this in the coordinate mapping section below
            
            # Ensure time is properly scaled for 100 Hz
            if 'time' in label_df.columns:
                time_max = label_df['time'].max()
                expected_duration = len(label_df) / 100.0  # Expected duration at 100 Hz
                if time_max > expected_duration * 2:  # Time is too long
                    print(f"    Warning: Time range {time_max:.2f}s seems too long for {len(label_df)} frames at 100 Hz")
                    # Recreate time column with proper 100 Hz scaling
                    label_df['time'] = np.arange(len(label_df)) / 100.0
                    print(f"    Recreated time column with proper 100 Hz scaling")

            # Precompute velocities via finite difference on positions (central differences)
            t = label_df['time'].astype(float).values
            vel_cols: Dict[str, np.ndarray] = {}
            for col in coordinate_columns:
                values = df_pos[col].astype(float).values
                # Unwrap rotational angles to avoid 2π jumps before differencing
                if col not in translational_coords:
                    values = np.unwrap(values)
                vel = np.zeros_like(values)
                if len(values) >= 2:
                    dt = np.diff(t)
                    dv = np.diff(values)
                    inst = dv / dt
                    if len(values) > 2:
                        vel[1:-1] = (inst[:-1] + inst[1:]) / 2.0
                    vel[0] = inst[0]
                    vel[-1] = inst[-1]
                vel_cols[col + '_u'] = vel

            # Write column header
            f.write("\t".join(column_names) + "\n")
            
            # Create a mapping from OpenSim coordinates back to original column names
            coord_to_original = {}
            for i, coord in enumerate(opensim_coordinates):
                if i < len(coordinate_columns):
                    coord_to_original[coord] = coordinate_columns[i]
            
            print(f"    Coordinate mapping: {coord_to_original}")
            
            # Write data rows
            for i in range(len(label_df)):
                # Get time value
                time_val = label_df.iloc[i]['time']
                
                # Get precomputed coordinate values (joint angles) for found coordinates
                coord_vals = []
                for coord in all_coordinates:
                    if coord == 'lumbar_rotation':
                        # Force lumbar_rotation to zero for rigid spine assumption
                        coord_vals.append(0.0)
                    elif coord in coord_to_original:
                        original_col = coord_to_original[coord]
                        coord_vals.append(float(df_pos.iloc[i][original_col]))
                    else:
                        # Missing coordinate, use default value
                        coord_vals.append(0.0)

                # Use precomputed velocities for found coordinates, 0.0 for missing ones
                coord_vel_vals = []
                for coord in all_coordinates:
                    if coord == 'lumbar_rotation':
                        # Force lumbar_rotation velocity to zero for rigid spine assumption
                        coord_vel_vals.append(0.0)
                    elif coord in coord_to_original:
                        original_col = coord_to_original[coord]
                        if (original_col + '_u') in vel_cols:
                            coord_vel_vals.append(float(vel_cols[original_col + '_u'][i]))
                        else:
                            coord_vel_vals.append(0.0)  # Default velocity
                    else:
                        # Missing coordinate, use default velocity
                        coord_vel_vals.append(0.0)
                
                # Get force/moment values from label data
                force_moment_vals = [label_df.iloc[i][col] for col in moment_columns]
                
                # Combine all values in the correct order
                all_vals = [time_val] + coord_vals + coord_vel_vals + force_moment_vals
                
                # Format all values to 6 decimal places
                data_vals = [f"{val:.6f}" for val in all_vals]
                f.write("\t".join(data_vals) + "\n")
        
        return True
        
    except Exception as e:
        print(f"    Error creating motion.sto file {output_path}: {e}")
        return False


def map_to_opensim_coordinates(metamobility_columns: List[str]) -> List[str]:
    """Map MetaMobility joint angle column names to OpenSim coordinate names.
    
    Args:
        metamobility_columns: List of MetaMobility joint angle column names
    
    Returns:
        List of OpenSim coordinate names
    """
    # Define mapping from MetaMobility to OpenSim coordinate names
    # MetaMobility uses format like: LeftAnkleAngles_X, RightHipAngles_Y, etc.
    # Based on the README, we have comprehensive joint angle data
    coordinate_mapping = {
        # Pelvis coordinates
        'RightPelvisAngles_X': 'pelvis_tilt',
        'RightPelvisAngles_Y': 'pelvis_list', 
        'RightPelvisAngles_Z': 'pelvis_rotation',
        'LeftPelvisAngles_X': 'pelvis_tilt',  # Same as right for pelvis
        'LeftPelvisAngles_Y': 'pelvis_list',
        'LeftPelvisAngles_Z': 'pelvis_rotation',
        
        # Hip coordinates
        'RightHipAngles_X': 'hip_flexion_r',
        'RightHipAngles_Y': 'hip_adduction_r',
        'RightHipAngles_Z': 'hip_rotation_r',
        'LeftHipAngles_X': 'hip_flexion_l',
        'LeftHipAngles_Y': 'hip_adduction_l',
        'LeftHipAngles_Z': 'hip_rotation_l',
        
        # Knee coordinates
        'RightKneeAngles_X': 'knee_angle_r',
        'RightKneeAngles_Y': 'knee_adduction_r',  # May not exist in OpenSim model
        'RightKneeAngles_Z': 'knee_rotation_r',   # May not exist in OpenSim model
        'LeftKneeAngles_X': 'knee_angle_l',
        'LeftKneeAngles_Y': 'knee_adduction_l',   # May not exist in OpenSim model
        'LeftKneeAngles_Z': 'knee_rotation_l',    # May not exist in OpenSim model
        
        # Ankle coordinates
        'RightAnkleAngles_X': 'ankle_angle_r',
        'RightAnkleAngles_Y': 'subtalar_angle_r',
        'RightAnkleAngles_Z': 'mtp_angle_r',
        'LeftAnkleAngles_X': 'ankle_angle_l',
        'LeftAnkleAngles_Y': 'subtalar_angle_l',
        'LeftAnkleAngles_Z': 'mtp_angle_l',
        
        # Spine coordinates (if present in OpenSim model)
        'RightSpineAngles_X': 'lumbar_extension',
        'RightSpineAngles_Y': 'lumbar_bending',
        'RightSpineAngles_Z': 'lumbar_rotation',
        'LeftSpineAngles_X': 'lumbar_extension',
        'LeftSpineAngles_Y': 'lumbar_bending',
        'LeftSpineAngles_Z': 'lumbar_rotation',
        
        # Thorax coordinates (if present in OpenSim model)
        'RightThoraxAngles_X': 'lumbar_extension',
        'RightThoraxAngles_Y': 'lumbar_bending',
        'RightThoraxAngles_Z': 'lumbar_rotation',
        'LeftThoraxAngles_X': 'lumbar_extension',
        'LeftThoraxAngles_Y': 'lumbar_bending',
        'LeftThoraxAngles_Z': 'lumbar_rotation',
    }
    
    # Map each column to OpenSim coordinate name
    opensim_coordinates = []
    for col in metamobility_columns:
        if col in coordinate_mapping:
            opensim_coordinates.append(coordinate_mapping[col])
        else:
            # If no mapping found, use the original name
            opensim_coordinates.append(col)
            print(f"    Warning: No OpenSim mapping found for coordinate: {col}")
    
    return opensim_coordinates


def standardize_joint_moment_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize joint moment column names to canonical format.
    
    MetaMobility joint moment names need to be mapped to canonical format:
    - RHipMoment_X -> hip_adduction_r_moment
    - RHipMoment_Y -> hip_flexion_r_moment  
    - RHipMoment_Z -> hip_rotation_r_moment
    - LHipMoment_X -> hip_adduction_l_moment
    - LHipMoment_Y -> hip_flexion_l_moment (with sign flip)
    - LHipMoment_Z -> hip_rotation_l_moment
    
    Note: Left hip moments have flipped signage (extension is +) so we flip the sign.
    """
    df = df.copy()
    new_columns = []
    
    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        
        # Pattern: SideJointMoment_Axis -> joint_axis_side_moment
        pattern = r'\b([rl])hipmoment_([xyz])(?:_|$)'
        match = re.search(pattern, col_lower)
        if match:
            side, axis = match.groups()
            side_full = 'r' if side == 'r' else 'l'
            
            # Map axes to canonical names
            if axis == 'x':
                new_col = f'hip_adduction_{side_full}_moment'
            elif axis == 'y':
                new_col = f'hip_flexion_{side_full}_moment'
                # Flip sign for left hip flexion to match right side convention
                if side == 'l':
                    df[col] = -df[col]  # Flip the sign
                    print(f"    Flipped sign for {col} -> {new_col} (left hip flexion)")
            elif axis == 'z':
                new_col = f'hip_rotation_{side_full}_moment'
            else:
                new_col = col_str
        else:
            new_col = col_str
        
        new_columns.append(new_col)
    
    df.columns = new_columns
    return df


def load_subject_info(dataset_root: Path) -> Dict[str, float]:
    """Load SubjectInfo.csv and return a dict mapping subject ID to body weight.
    
    Args:
        dataset_root: Path to dataset root (should contain SubjectInfo.csv)
    
    Returns:
        Dict mapping subject ID (e.g., 'AB01') to body weight in kg
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


def process_memo_dataset(input_root: str, output_root: str, conditions: List[str], 
                        unit: str, max_frames: int, subjects: Optional[List[str]] = None):
    """Process the entire MetaMobility dataset."""
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
    
    print(f"🚀 Starting MetaMobility preprocessing...")
    print(f"   Input: {input_root}")
    print(f"   Output: {output_root}")
    print(f"   Conditions: {', '.join(conditions)}")
    print(f"   Unit: {unit}")
    print(f"   Max frames: {max_frames}")
    print()
    
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
        for condition_dir in subject_dir.iterdir():
            if not condition_dir.is_dir() or condition_dir.name.startswith('.'):
                continue
            # Look for .osim files in any subdirectory
            for osim_file in condition_dir.rglob("*.osim"):
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
        
        # Process each condition directory
        for condition_dir in subject_dir.iterdir():
            if not condition_dir.is_dir() or condition_dir.name.startswith('.'):
                continue
                
            condition_name = condition_dir.name
            if condition_name not in conditions:
                continue
                
            print(f"  Processing condition: {condition_name}")
            
            # Create condition directory in output
            condition_out = subject_out / condition_name
            condition_out.mkdir(exist_ok=True)
            
            # Process each trial
            for trial_dir in condition_dir.iterdir():
                if not trial_dir.is_dir() or trial_dir.name.startswith('.'):
                    continue
                
                trial_name = trial_dir.name
                
                # Get Input and Label files
                input_dir = trial_dir / "Input"
                label_dir = trial_dir / "Label"
                
                if not input_dir.exists() or not label_dir.exists():
                    print(f"    Warning: Missing Input or Label directory in {trial_dir}")
                    continue
                
                # Find IMU and label files
                imu_file = input_dir / "imu_data.csv"
                label_files = list(label_dir.glob("*.csv"))
                
                if not imu_file.exists() or not label_files:
                    print(f"    Warning: Missing IMU or label files in {trial_dir}")
                    continue
                
                # Use the first label file found
                label_file = label_files[0]
                
                # Create trial output directory
                trial_out = condition_out / trial_name
                trial_out.mkdir(exist_ok=True)
                
                total_trials += 1
                print(f"    Processing trial {total_trials}: {subject_name}/{condition_name}/{trial_name}")
                if process_trial(imu_file, label_file, trial_out, unit, max_frames, subject_weight):
                    processed_count += 1
                    print(f"    ✅ Success: {subject_name}/{condition_name}/{trial_name}")
                else:
                    print(f"    ❌ Failed: {subject_name}/{condition_name}/{trial_name}")
    
    # Check for missing subjects if filter was applied
    if subjects is not None:
        missing_subjects = set(subjects) - found_subjects
        if missing_subjects:
            print(f"\n⚠️  Warning: The following subjects were not found: {', '.join(missing_subjects)}")
    
    print(f"\n🎉 Processing Complete!")
    print(f"   Successfully processed: {processed_count}/{total_trials} trials")
    print(f"   Success rate: {(processed_count/total_trials)*100:.1f}%")
    print(f"   Output directory: {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess MetaMobility dataset")
    parser.add_argument("--input-root", required=True, 
                       help="Path to raw MetaMobility data directory")
    parser.add_argument("--output-root", required=True, 
                       help="Path to output processed directory")
    parser.add_argument("--conditions", default="0mps, 0p2mps, 0p4mps, 0p6mps, 0p8mps, 1p0mps, 1p2mps, 1p4mps, transient_15sec, transient_30sec", 
                       help="Comma-separated list of conditions to process")
    parser.add_argument("--unit", choices=["rad", "deg"], default="deg", 
                       help="Unit of IMU gyro in source CSV")
    parser.add_argument("--max-frames", type=int, default=40000, 
                       help="Maximum number of frames to process per trial")
    parser.add_argument("--subjects", 
                       help="Comma-separated list of subjects to process (e.g., AB01,AB02,AB03). If not provided, processes all subjects.")
    
    args = parser.parse_args()
    
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
    
    process_memo_dataset(
        args.input_root, 
        args.output_root, 
        conditions, 
        args.unit, 
        args.max_frames,
        subjects
    )


if __name__ == "__main__":
    main()
