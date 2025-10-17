#!/usr/bin/env python3
"""
Generate .sto files for Camargo dataset from ik and id files.

This script processes Camargo data from the processed directory and creates .sto files
for each trial in the opensim subdirectory, which are needed for realignment of axes.

Usage:
  python generate_sto_files_camargo.py \
  --input-root /Users/luorix/Desktop/MetaMobility\\ Lab\\ \\(CMU\\)/data/Camargo_processed \\
  --output-root /Users/luorix/Desktop/MetaMobility\\ Lab\\ \\(CMU\\)/data/Final/Camargo \\
    [--conditions treadmill,ramp,stair,levelground,static]
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
        # Try with c engine first (fastest)
        return pd.read_csv(file_path, engine='c')
    except Exception as e:
        print(f"    Warning: Failed to read {file_path} with c engine: {e}")
        try:
            # Try with python engine
            return pd.read_csv(file_path, engine='python')
        except Exception as e2:
            print(f"    Error: Failed to read {file_path} with python engine: {e2}")
            return None


def create_motion_sto_file(ik_df: pd.DataFrame, id_df: pd.DataFrame, output_path: Path) -> bool:
    """Create a single motion.sto file combining IK and ID data in OpenSim format."""
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rename 'Header' column to 'time' in both dataframes
        if 'Header' in ik_df.columns:
            ik_df = ik_df.rename(columns={'Header': 'time'})
        if 'Header' in id_df.columns:
            id_df = id_df.rename(columns={'Header': 'time'})
        
        # Ensure both dataframes have the same time column and length
        if len(ik_df) != len(id_df):
            print(f"    Warning: IK and ID data have different lengths: {len(ik_df)} vs {len(id_df)}")
            return False
        
        # Create the proper OpenSim motion file format
        with open(output_path, 'w') as f:
            # Write header in the correct format
            f.write("DataType=double\n")
            f.write("version=3\n")
            f.write("OpenSimVersion=4.5\n")
            f.write("endheader\n")
            
            # Define the exact column order to match your desired format
            # Based on your example, the order should be:
            # time, coordinates, coordinate_velocities, forces_moments
            
            # Get coordinate columns from IK data (excluding time)
            ik_columns = [col for col in ik_df.columns if col != 'time']
            
            # Get force/moment columns from ID data (excluding time)  
            id_columns = [col for col in id_df.columns if col != 'time']
            
            # Create the column header in the exact order from your example
            column_names = ['time']

            # Add coordinate names (positions)
            column_names.extend(ik_columns)

            # Add coordinate velocity names (with _u suffix)
            column_names.extend([col + '_u' for col in ik_columns])

            # Add force/moment names
            column_names.extend(id_columns)

            # Precompute: convert IK positions to correct units (rad for rotations, m for translations)
            df_pos = ik_df.copy()
            translational_coords = {'pelvis_tx','pelvis_ty','pelvis_tz'}
            for col in ik_columns:
                if col not in translational_coords:
                    df_pos[col] = np.radians(df_pos[col].astype(float))
                else:
                    df_pos[col] = df_pos[col].astype(float)

            # Precompute velocities via finite difference on positions (central differences)
            t = ik_df['time'].astype(float).values
            vel_cols: Dict[str, np.ndarray] = {}
            for col in ik_columns:
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
            
            # Write data rows
            for i in range(len(ik_df)):
                # Get time value
                time_val = ik_df.iloc[i]['time']
                
                # Get precomputed coordinate values
                coord_vals = [float(df_pos.iloc[i][col]) for col in ik_columns]

                # Use precomputed velocities
                coord_vel_vals = [float(vel_cols[col + '_u'][i]) for col in ik_columns]
                
                # Get force/moment values from ID data
                force_moment_vals = [id_df.iloc[i][col] for col in id_columns]
                
                # Combine all values in the correct order
                all_vals = [time_val] + coord_vals + coord_vel_vals + force_moment_vals
                
                # Format all values to 6 decimal places
                data_vals = [f"{val:.6f}" for val in all_vals]
                f.write("\t".join(data_vals) + "\n")
        
        return True
        
    except Exception as e:
        print(f"    Error creating motion.sto file {output_path}: {e}")
        return False


def process_trial_opensim_files(ik_file: Path, id_file: Path, output_dir: Path) -> bool:
    """Process ik and id files for a single trial and create a single motion.sto file."""
    try:
        # Read IK data
        ik_df = read_csv_flexible(ik_file)
        if ik_df is None:
            return False
        
        # Read ID data
        id_df = read_csv_flexible(id_file)
        if id_df is None:
            return False
        
        # Create opensim directory
        opensim_dir = output_dir / "opensim"
        opensim_dir.mkdir(parents=True, exist_ok=True)
        
        # Create single motion.sto file
        motion_sto_path = opensim_dir / "motion.sto"
        
        success = create_motion_sto_file(ik_df, id_df, motion_sto_path)
        
        if success:
            print(f"  Created motion.sto for {ik_file.stem}")
            return True
        else:
            return False
        
    except Exception as e:
        print(f"  Error processing trial {ik_file}: {e}")
        return False


def copy_osim_files(subject_osim_dir: Path, output_dir: Path) -> bool:
    """Copy .osim and setup files to the output directory."""
    try:
        # Create opensim directory
        opensim_dir = output_dir / "opensim"
        opensim_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy .osim file
        osim_file = subject_osim_dir / f"{output_dir.parent.name}.osim"
        if osim_file.exists():
            shutil.copy2(osim_file, opensim_dir / f"{output_dir.parent.name}.osim")
        
        # Copy setup files
        for setup_file in subject_osim_dir.glob("*.xml"):
            shutil.copy2(setup_file, opensim_dir / setup_file.name)
        
        return True
        
    except Exception as e:
        print(f"  Error copying .osim files: {e}")
        return False


def process_camargo_opensim_files(input_root: str, output_root: str, conditions: List[str]):
    """Process the entire Camargo dataset to create .sto files."""
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_root}")
    
    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_root}")
    
    processed_count = 0
    total_trials = 0
    copied_subject_assets: Dict[str, bool] = {}
    
    # Process each subject
    for subject_dir in input_path.iterdir():
        if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
            continue
            
        subject_name = subject_dir.name
        print(f"Processing subject: {subject_name}")
        
        # Check if subject exists in output directory
        subject_out = output_path / subject_name
        if not subject_out.exists():
            print(f"  Warning: Subject {subject_name} not found in output directory")
            continue
        
        # Get .osim files directory
        osim_dir = subject_dir / "osimxml"
        
        # Process each date directory
        for date_dir in subject_dir.iterdir():
            if not date_dir.is_dir() or date_dir.name.startswith('.') or date_dir.name == "osimxml":
                continue
                
            # Process each condition directory
            for condition_dir in date_dir.iterdir():
                if not condition_dir.is_dir() or condition_dir.name.startswith('.'):
                    continue
                    
                condition_name = condition_dir.name
                if condition_name not in conditions:
                    continue
                    
                print(f"  Processing condition: {condition_name}")
                
                # Check if condition exists in output directory
                condition_out = subject_out / condition_name
                if not condition_out.exists():
                    print(f"    Warning: Condition {condition_name} not found in output directory")
                    continue
                
                # Get IK and ID files
                ik_dir = condition_dir / "ik"
                id_dir = condition_dir / "id"
                
                if not ik_dir.exists() or not id_dir.exists():
                    print(f"    Warning: Missing ik or id directory in {condition_dir}")
                    continue
                
                # Process each trial
                for ik_file in ik_dir.iterdir():
                    if not ik_file.is_file() or not ik_file.suffix == '.csv' or ik_file.name.startswith('.'):
                        continue
                    
                    # Find corresponding ID file
                    id_file = id_dir / ik_file.name
                    if not id_file.exists():
                        print(f"    Warning: No corresponding ID file for {ik_file.name}")
                        continue
                    
                    # Find corresponding trial in output directory
                    trial_name = ik_file.stem
                    trial_out = condition_out / trial_name
                    if not trial_out.exists():
                        print(f"    Warning: Trial {trial_name} not found in output directory")
                        continue
                    
                    total_trials += 1
                    if process_trial_opensim_files(ik_file, id_file, trial_out):
                        processed_count += 1
                        # Copy model/setup files once per subject to Final/<subject>/opensim
                        if not copied_subject_assets.get(subject_name, False):
                            try:
                                copy_osim_files(osim_dir, subject_out)
                                copied_subject_assets[subject_name] = True
                            except Exception:
                                pass
    
    print(f"\nCompleted! Processed {processed_count}/{total_trials} trials successfully.")
    print(f"Output directory: {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Generate .sto files for Camargo dataset")
    parser.add_argument("--input-root", required=True, 
                       help="Path to processed Camargo data directory")
    parser.add_argument("--output-root", required=True, 
                       help="Path to output Final directory")
    parser.add_argument("--conditions", default="treadmill,ramp,stair,levelground,static", 
                       help="Comma-separated list of conditions to process")
    
    args = parser.parse_args()
    
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    
    process_camargo_opensim_files(
        args.input_root, 
        args.output_root, 
        conditions
    )


if __name__ == "__main__":
    main()
