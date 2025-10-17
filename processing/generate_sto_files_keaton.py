#!/usr/bin/env python3
"""
Generate .sto files for the Keaton dataset from angle and moment CSV files.

This mirrors the output format and logic used by generate_sto_files_camargo.py, producing
one motion.sto per trial under the trial's opensim folder, at the same depth as Input/Label.

Keaton structure:
- /Volumes/Samsung_T5/raw_data/Keaton/<subject>/<trial>/<subject>_<trial>_angle.csv (IK data)
- /Volumes/Samsung_T5/raw_data/Keaton/<subject>/<trial>/<subject>_<trial>_moment.csv (ID data)

Usage:
  python generate_sto_files_keaton.py \
    --input-root /Volumes/Samsung_T5/raw_data/Keaton \
    --output-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Keaton" \
    --conditions "levelground,ramp,stair"
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shutil
import os


def read_csv_flexible(file_path: Path) -> Optional[pd.DataFrame]:
    """Read CSV file with flexible engine selection."""
    try:
        return pd.read_csv(file_path, engine="c")
    except Exception:
        try:
            return pd.read_csv(file_path, engine="python")
        except Exception as e2:
            print(f"    Error: Failed to read {file_path}: {e2}")
            return None


def create_motion_sto_file(ik_df: pd.DataFrame, id_df: pd.DataFrame, output_path: Path) -> bool:
    """Create a single motion.sto file combining IK and ID data in OpenSim format."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize time column name
        if "Header" in ik_df.columns:
            ik_df = ik_df.rename(columns={"Header": "time"})
        if "Header" in id_df.columns:
            id_df = id_df.rename(columns={"Header": "time"})

        if len(ik_df) != len(id_df):
            print(f"    Warning: IK and ID lengths differ: {len(ik_df)} vs {len(id_df)}")
            return False

        # Define the complete set of OpenSim coordinates in the correct order
        # This matches the OpenSim model coordinate order
        required_coords = [
            'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 
            'subtalar_angle_r', 'mtp_angle_r',
            'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l', 
            'subtalar_angle_l', 'mtp_angle_l',
            'lumbar_extension', 'lumbar_bending', 'lumbar_rotation'
        ]
        
        # Map Keaton coordinate names to OpenSim names
        coord_mapping = {
            'hip_flexion_r': 'hip_flexion_r',
            'hip_adduction_r': 'hip_adduction_r', 
            'hip_rotation_r': 'hip_rotation_r',
            'knee_angle_r': 'knee_angle_r',
            'ankle_angle_r': 'ankle_angle_r',
            'subtalar_angle_r': 'subtalar_angle_r',
            'hip_flexion_l': 'hip_flexion_l',
            'hip_adduction_l': 'hip_adduction_l',
            'hip_rotation_l': 'hip_rotation_l', 
            'knee_angle_l': 'knee_angle_l',
            'ankle_angle_l': 'ankle_angle_l',
            'subtalar_angle_l': 'subtalar_angle_l'
        }
        
        # Create complete dataframe with all required coordinates
        complete_df = pd.DataFrame(index=ik_df.index)
        complete_df['time'] = ik_df['time']
        
        # Add available coordinates from Keaton data
        for keaton_col, opensim_col in coord_mapping.items():
            if keaton_col in ik_df.columns:
                complete_df[opensim_col] = ik_df[keaton_col]
            else:
                complete_df[opensim_col] = 0.0  # Default value for missing coordinates
        
        # Add missing coordinates with default values
        for coord in required_coords:
            if coord not in complete_df.columns:
                if coord.startswith('pelvis_t') and coord != 'pelvis_tilt':
                    # Pelvis translations - use reasonable defaults
                    if coord == 'pelvis_tx':
                        complete_df[coord] = 0.0  # Forward position
                    elif coord == 'pelvis_ty':
                        complete_df[coord] = 0.9  # Height (approximate hip height)
                    elif coord == 'pelvis_tz':
                        complete_df[coord] = 0.0  # Lateral position
                elif coord.startswith('pelvis_'):
                    # Pelvis rotations
                    complete_df[coord] = 0.0
                elif coord.startswith('lumbar_'):
                    # Lumbar coordinates
                    complete_df[coord] = 0.0
                elif coord == 'mtp_angle_r' or coord == 'mtp_angle_l':
                    # MTP angles (toe joints)
                    complete_df[coord] = 0.0
                else:
                    # Other missing coordinates
                    complete_df[coord] = 0.0

        # Reorder columns to match required order
        complete_df = complete_df[['time'] + required_coords]
        
        # Precompute positions (radians for rotations; meters for translations)
        translational_coords = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}
        df_pos = complete_df.copy()
        for col in required_coords:
            if col not in translational_coords:
                df_pos[col] = np.radians(df_pos[col].astype(float))
            else:
                df_pos[col] = df_pos[col].astype(float)

        # Precompute speeds via central differences with unwrap for rotations
        t = complete_df["time"].astype(float).values
        vel_cols: Dict[str, np.ndarray] = {}
        for col in required_coords:
            values = df_pos[col].astype(float).values
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
            vel_cols[col + "_u"] = vel

        # Get ID columns (moments/forces)
        id_columns = [c for c in id_df.columns if c != "time"]

        with open(output_path, "w") as f:
            f.write("DataType=double\n")
            f.write("version=3\n")
            f.write("OpenSimVersion=4.5\n")
            f.write("endheader\n")

            # Create header with correct order: time, coordinates, velocities, forces/moments
            header = ["time"] + required_coords + [c + "_u" for c in required_coords] + id_columns
            f.write("\t".join(header) + "\n")

            for i in range(len(complete_df)):
                time_val = float(complete_df.iloc[i]["time"])
                coord_vals = [float(df_pos.iloc[i][c]) for c in required_coords]
                coord_vels = [float(vel_cols[c + "_u"][i]) for c in required_coords]
                fm_vals = [float(id_df.iloc[i][c]) for c in id_columns]
                row_vals = [time_val] + coord_vals + coord_vels + fm_vals
                f.write("\t".join(f"{v:.6f}" for v in row_vals) + "\n")

        return True
    except Exception as e:
        print(f"    Error creating motion.sto file {output_path}: {e}")
        return False


def process_trial(ik_file: Path, id_file: Path, trial_out_dir: Path) -> bool:
    try:
        ik_df = read_csv_flexible(ik_file)
        if ik_df is None:
            return False
        id_df = read_csv_flexible(id_file)
        if id_df is None:
            return False

        opensim_dir = trial_out_dir / "opensim"
        opensim_dir.mkdir(parents=True, exist_ok=True)
        motion_sto_path = opensim_dir / "motion.sto"
        ok = create_motion_sto_file(ik_df, id_df, motion_sto_path)
        if ok:
            print(f"  Created motion.sto for {ik_file.stem}")
        return ok
    except Exception as e:
        print(f"  Error processing trial {ik_file}: {e}")
        return False


def copy_subject_osim_if_available(subject_out_dir: Path) -> None:
    """If a subject model exists under Final/Keaton/<subject>/opensim, ensure it's present."""
    try:
        # If an .osim already exists in opensim, do nothing. Otherwise, try to find one nearby.
        opensim_dir = subject_out_dir / "opensim"
        opensim_dir.mkdir(parents=True, exist_ok=True)
        existing = list(opensim_dir.glob("*.osim"))
        if existing:
            return
        # Attempt to find a scaled model named <subject>.osim under the same dir (already is opensim)
        cand = opensim_dir / f"{subject_out_dir.name}.osim"
        if cand.exists():
            return
        # Nothing to copy; leave as-is
    except Exception:
        pass


def categorize_condition(trial_name: str) -> str:
    """Categorize trial name into condition type (matching preprocess_keaton.py)."""
    trial_lower = trial_name.lower()
    
    if 'normal_walk' in trial_lower or 'meander' in trial_lower:
        return 'levelground'
    elif 'incline' in trial_lower or 'ramp' in trial_lower:
        return 'ramp'
    elif 'stairs' in trial_lower or 'stair' in trial_lower:
        return 'stair'
    else:
        return 'other'


def process_keaton(input_root: str, output_root: str, conditions: List[str]) -> None:
    """Process Keaton dataset and generate .sto files."""
    in_root = Path(input_root)
    out_root = Path(output_root)
    if not in_root.exists():
        raise ValueError(f"Input directory does not exist: {input_root}")
    if not out_root.exists():
        raise ValueError(f"Output directory does not exist: {output_root}")

    total = 0
    ok = 0

    # Keaton structure: <input>/<subject>/<trial>/<subject>_<trial>_angle.csv and <subject>_<trial>_moment.csv
    for subject_dir in in_root.iterdir():
        if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
            continue
        subject = subject_dir.name
        subject_out = out_root / subject
        if not subject_out.exists():
            print(f"  Warning: Subject {subject} not found in output root; skipping")
            continue
        print(f"Processing subject: {subject}")

        # Process each trial directory directly under subject
        for trial_dir in subject_dir.iterdir():
            if not trial_dir.is_dir() or trial_dir.name.startswith('.'):
                continue
                
            trial_name = trial_dir.name
            condition = categorize_condition(trial_name)
            
            if condition not in conditions and 'other' not in conditions:
                continue
                
            print(f"  Processing trial: {trial_name} (condition: {condition})")
            
            # Check if condition exists in output directory
            condition_out = subject_out / condition
            if not condition_out.exists():
                print(f"    Warning: Condition {condition} not found in output directory")
                continue

            # Find angle (IK) and moment (ID) files
            angle_file = None
            moment_file = None
            
            for file in trial_dir.iterdir():
                if not file.is_file() or file.name.startswith('.'):
                    continue
                    
                if file.name.endswith('_angle.csv'):
                    angle_file = file
                elif file.name.endswith('_moment.csv'):
                    moment_file = file
            
            if not angle_file or not moment_file:
                print(f"    Warning: Missing angle or moment file in {trial_dir}")
                continue

            # Find corresponding trial in output directory
            trial_out = condition_out / trial_name
            if not trial_out.exists():
                print(f"    Warning: Trial {trial_name} not found in output directory")
                continue

            total += 1
            if process_trial(angle_file, moment_file, trial_out):
                ok += 1

        # ensure subject .osim present if available
        copy_subject_osim_if_available(subject_out)

    print(f"\nCompleted! Processed {ok}/{total} trials successfully.")
    print(f"Output directory: {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Generate .sto files for Keaton dataset")
    parser.add_argument("--input-root", required=True, help="Path to Keaton raw data root (e.g., /Volumes/.../Keaton)")
    parser.add_argument("--output-root", required=True, help="Path to Final Keaton root")
    parser.add_argument("--conditions", default="levelground,ramp,stair", help="Comma-separated conditions")

    args = parser.parse_args()
    conditions = [c.strip() for c in args.conditions.split(',') if c.strip()]
    process_keaton(args.input_root, args.output_root, conditions)


if __name__ == "__main__":
    main()


