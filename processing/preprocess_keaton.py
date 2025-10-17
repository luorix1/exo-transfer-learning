#!/usr/bin/env python3
"""
Preprocess Keaton dataset from raw format to standardized format.

This script processes raw Keaton data from /Volumes/Samsung_T5/raw_data/Samples/Keaton
and converts it to a standardized format with only gyro columns, similar to canonical_frame_converter.py
but without the canonical frame conversion step.

Keaton structure:
- Subject/Trial/AB01_trial_imu_real.csv
- Subject/Trial/AB01_trial_moment.csv

Usage:
  python preprocess_keaton.py \
    --input-root /Volumes/Samsung_T5/raw_data/Samples/Keaton \
    --output-root /Users/luorix/Desktop/MetaMobility\ Lab\ \(CMU\)/data/Keaton_processed \
    [--conditions levelground,ramp,stair] \
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


def standardize_segment_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize segment names to canonical format.
    
    Transforms:
    - LShank_GYROX -> shank_l_gyro_x
    - RShank_GYROX -> shank_r_gyro_x
    - LAThigh_GYROX -> thigh_l_gyro_x
    - RAThigh_GYROX -> thigh_r_gyro_x
    - LPThigh_GYROX -> thigh_l_gyro_x (posterior thigh -> thigh)
    - RPThigh_GYROX -> thigh_r_gyro_x (posterior thigh -> thigh)
    - LPelvis_GYROX -> pelvis_l_gyro_x
    - RPelvis_GYROX -> pelvis_r_gyro_x
    """
    df = df.copy()
    new_columns = []
    
    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        
        # Pattern: L/R + Segment + GYRO + Axis -> segment_side_gyro_axis
        keaton_pattern = r'\b(l|r)(shank|athigh|pthigh|pelvis)_(gyro)([xyz])(?:_|$)'
        match = re.search(keaton_pattern, col_lower)
        if match:
            side, segment, sensor, axis = match.groups()
            # Convert L/R to l/r
            side_short = side.lower()
            # Handle compound segment names: athigh/pthigh -> thigh
            if segment in ['athigh', 'pthigh']:
                segment = 'thigh'
            # Reconstruct: segment_side_gyro_axis
            new_col = re.sub(keaton_pattern, rf'{segment}_{side_short}_{sensor}_{axis}', col_str, flags=re.IGNORECASE)
        else:
            new_col = col_str
        
        new_columns.append(new_col)
    
    df.columns = new_columns
    return df


def extract_gyro_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only gyro columns from IMU data."""
    gyro_cols = [col for col in df.columns if 'gyro' in col.lower()]
    
    # Always include time/header column if present
    time_cols = [col for col in df.columns if col.lower() in ['time', 'header']]
    
    selected_cols = time_cols + gyro_cols
    return df[selected_cols]


def process_trial(imu_file: Path, moment_file: Path, output_dir: Path, unit: str, max_frames: int) -> bool:
    """Process a single trial and save standardized data."""
    try:
        # Read IMU data
        imu_df = pd.read_csv(imu_file)
        
        # Extract only gyro columns
        imu_df = extract_gyro_columns(imu_df)
        
        # Standardize column names
        imu_df = standardize_segment_names(imu_df)
        
        # Apply unit conversion for gyro data if needed
        if unit == "deg":
            gyro_cols = [col for col in imu_df.columns if 'gyro' in col.lower()]
            for col in gyro_cols:
                imu_df[col] = imu_df[col] * (np.pi / 180.0)
        
        # Limit frames if specified
        if max_frames > 0 and len(imu_df) > max_frames:
            imu_df = imu_df.iloc[:max_frames]
        
        # Read moment data (joint moments)
        moment_df = pd.read_csv(moment_file)
        
        # Standardize joint moment column names
        moment_df = standardize_joint_moment_names(moment_df)
        
        # Create output directories
        input_dir = output_dir / "Input"
        label_dir = output_dir / "Label"
        input_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        imu_output = input_dir / "imu_data.csv"
        label_output = label_dir / "joint_moment.csv"
        
        imu_df.to_csv(imu_output, index=False)
        moment_df.to_csv(label_output, index=False)
        
        print(f"  Processed: {imu_file.name} -> {output_dir.name}")
        return True
        
    except Exception as e:
        print(f"  Error processing {imu_file}: {e}")
        return False


def standardize_joint_moment_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize joint moment column names to canonical format.
    
    Keaton joint moment names are already in the correct format:
    - hip_flexion_r_moment -> hip_flexion_r_moment (already correct)
    - hip_flexion_l_moment -> hip_flexion_l_moment (already correct)
    """
    # Keaton joint moment names are already in the correct format
    return df


def categorize_condition(trial_name: str) -> str:
    """Categorize trial name into condition type."""
    trial_lower = trial_name.lower()
    
    if 'normal_walk' in trial_lower or 'meander' in trial_lower:
        return 'levelground'
    elif 'incline' in trial_lower or 'ramp' in trial_lower:
        return 'ramp'
    elif 'stairs' in trial_lower or 'stair' in trial_lower:
        return 'stair'
    else:
        return 'other'


def process_keaton_dataset(input_root: str, output_root: str, conditions: List[str], 
                          unit: str, max_frames: int):
    """Process the entire Keaton dataset."""
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_root}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    total_trials = 0
    
    # Process each subject
    for subject_dir in input_path.iterdir():
        if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
            continue
            
        subject_name = subject_dir.name
        print(f"Processing subject: {subject_name}")
        
        # Create subject directory in output
        subject_out = output_path / subject_name
        subject_out.mkdir(exist_ok=True)
        
        # Process each trial directory
        for trial_dir in subject_dir.iterdir():
            if not trial_dir.is_dir() or trial_dir.name.startswith('.'):
                continue
                
            trial_name = trial_dir.name
            condition = categorize_condition(trial_name)
            
            if condition not in conditions and 'other' not in conditions:
                continue
                
            print(f"  Processing trial: {trial_name} (condition: {condition})")
            
            # Create condition directory in output
            condition_out = subject_out / condition
            condition_out.mkdir(exist_ok=True)
            
            # Find IMU and moment files
            imu_file = None
            moment_file = None
            
            for file in trial_dir.iterdir():
                if not file.is_file() or file.name.startswith('.'):
                    continue
                    
                if file.name.endswith('_imu_real.csv'):
                    imu_file = file
                elif file.name.endswith('_moment.csv'):
                    moment_file = file
            
            if not imu_file or not moment_file:
                print(f"    Warning: Missing IMU or moment file in {trial_dir}")
                continue
            
            # Create trial output directory
            trial_out = condition_out / trial_name
            trial_out.mkdir(exist_ok=True)
            
            total_trials += 1
            if process_trial(imu_file, moment_file, trial_out, unit, max_frames):
                processed_count += 1
    
    print(f"\nCompleted! Processed {processed_count}/{total_trials} trials successfully.")
    print(f"Output directory: {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Keaton dataset")
    parser.add_argument("--input-root", required=True, 
                       help="Path to raw Keaton data directory")
    parser.add_argument("--output-root", required=True, 
                       help="Path to output processed directory")
    parser.add_argument("--conditions", default="levelground,ramp,stair", 
                       help="Comma-separated list of conditions to process")
    parser.add_argument("--unit", choices=["rad", "deg"], default="deg", 
                       help="Unit of IMU gyro in source CSV")
    parser.add_argument("--max-frames", type=int, default=40000, 
                       help="Maximum number of frames to process per trial")
    
    args = parser.parse_args()
    
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    
    process_keaton_dataset(
        args.input_root, 
        args.output_root, 
        conditions, 
        args.unit, 
        args.max_frames
    )


if __name__ == "__main__":
    main()
