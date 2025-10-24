#!/usr/bin/env python3
"""
Transform IMU data using batch optimization results.

This script takes batch optimization results and transforms IMU data for each subject
using their individual rotation matrices, creating a canonical dataset.

Usage:
    # Transform all subjects using batch results
    python transform_batch_results.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Camargo" \
        --output-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Canonical_Camargo" \
        --batch-results "results/camargo_batch/batch_results.json"
    
    # Transform specific subjects
    python transform_batch_results.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Camargo" \
        --output-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Canonical_Camargo" \
        --batch-results "results/camargo_batch/batch_results.json" \
        --subjects "AB06,AB07,AB08"
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_batch_results(batch_file: Path) -> Optional[Dict]:
    """
    Load batch optimization results.
    
    Args:
        batch_file: Path to batch_results.json
    
    Returns:
        Batch results dictionary or None on error
    """
    try:
        with open(batch_file, 'r') as f:
            results = json.load(f)
        
        if 'subjects' not in results or 'metadata' not in results:
            print(f"❌ Invalid batch results format: {batch_file}")
            return None
        
        metadata = results['metadata']
        print(f"📊 Loaded batch results:")
        print(f"   Dataset: {metadata.get('dataset', 'Unknown')}")
        print(f"   Condition: {metadata.get('condition', 'Unknown')}")
        print(f"   Trial: {metadata.get('trial', 'Unknown')}")
        print(f"   Total subjects: {metadata.get('total_subjects', 0)}")
        
        return results
    
    except Exception as e:
        print(f"❌ Error loading batch results from {batch_file}: {e}")
        return None


def get_trial_rotation_matrices(batch_results: Dict, subject: str, condition: str, trial: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Get rotation matrices for a specific trial from batch results.
    
    Args:
        batch_results: Batch results dictionary
        subject: Subject ID
        condition: Condition name
        trial: Trial name
    
    Returns:
        Dict mapping segment names to rotation matrices, or None if not found
    """
    if subject not in batch_results['subjects']:
        print(f"❌ Subject {subject} not found in batch results")
        return None
    
    subject_data = batch_results['subjects'][subject]
    if condition not in subject_data:
        print(f"❌ Condition {condition} not found for subject {subject}")
        return None
    
    condition_data = subject_data[condition]
    if trial not in condition_data:
        print(f"❌ Trial {trial} not found for subject {subject}/{condition}")
        return None
    
    trial_data = condition_data[trial]
    if 'segments' not in trial_data:
        print(f"❌ No segments found for subject {subject}/{condition}/{trial}")
        return None
    
    rotation_matrices = {}
    for segment_name, segment_data in trial_data['segments'].items():
        rotation_matrix = np.array(segment_data['optimal_rotation_matrix'])
        
        # Verify it's a valid rotation matrix
        det = np.linalg.det(rotation_matrix)
        if not np.isclose(det, 1.0, atol=0.01):
            print(f"⚠️  Warning: Rotation matrix for {subject}/{condition}/{trial}/{segment_name} has determinant {det:.6f}, expected 1.0")
        
        rotation_matrices[segment_name] = rotation_matrix
    
    return rotation_matrices


def find_gyro_columns(df: pd.DataFrame, segment: str = "thigh_r") -> Optional[List[str]]:
    """Find gyro columns for a specific segment in the dataframe."""
    candidates = [
        [f"{segment}_gyro_x", f"{segment}_gyro_y", f"{segment}_gyro_z"],
        [f"{segment.upper()}_GYROX", f"{segment.upper()}_GYROY", f"{segment.upper()}_GYROZ"],
        [f"{segment}_Gyro_X", f"{segment}_Gyro_Y", f"{segment}_Gyro_Z"],
    ]
    
    lower_cols = {c.lower(): c for c in df.columns}
    
    for candidate_set in candidates:
        actual_cols = []
        for c in candidate_set:
            if c.lower() in lower_cols:
                actual_cols.append(lower_cols[c.lower()])
        
        if len(actual_cols) == 3:
            return actual_cols
    
    return None


def find_accel_columns(df: pd.DataFrame, segment: str = "thigh_r") -> Optional[List[str]]:
    """Find accel columns for a specific segment in the dataframe."""
    candidates = [
        [f"{segment}_accel_x", f"{segment}_accel_y", f"{segment}_accel_z"],
        [f"{segment.upper()}_ACCELX", f"{segment.upper()}_ACCELY", f"{segment.upper()}_ACCELZ"],
        [f"{segment}_Accel_X", f"{segment}_Accel_Y", f"{segment}_Accel_Z"],
    ]
    
    lower_cols = {c.lower(): c for c in df.columns}
    
    for candidate_set in candidates:
        actual_cols = []
        for c in candidate_set:
            if c.lower() in lower_cols:
                actual_cols.append(lower_cols[c.lower()])
        
        if len(actual_cols) == 3:
            return actual_cols
    
    return None


def get_imu_segment_name(opensim_segment: str) -> str:
    """Map OpenSim segment names to IMU data segment names"""
    segment_mapping = {
        "femur_r": "thigh_r",
        "femur_l": "thigh_l",
        "tibia_r": "shank_r",
        "tibia_l": "shank_l",
        "pelvis": "pelvis"
    }
    return segment_mapping.get(opensim_segment, opensim_segment)


def transform_gyro_data(gyro_data: np.ndarray, rotation_matrix: np.ndarray, inverse: bool = True) -> np.ndarray:
    """Transform gyro data using rotation matrix."""
    if inverse:
        transform = rotation_matrix.T
    else:
        transform = rotation_matrix
    
    transformed_data = np.array([transform @ frame for frame in gyro_data])
    return transformed_data


def transform_trial_imu_data(
    input_imu_file: Path,
    output_imu_file: Path,
    rotation_matrices: Dict[str, np.ndarray],
    dry_run: bool = False
) -> bool:
    """Transform IMU data for a single trial using subject-specific rotation matrices."""
    try:
        df = pd.read_csv(input_imu_file)
        transformed_count = 0
        
        for opensim_segment, rotation_matrix in rotation_matrices.items():
            imu_segment = get_imu_segment_name(opensim_segment)
            
            # Transform gyro data
            gyro_cols = find_gyro_columns(df, imu_segment)
            if gyro_cols is not None:
                gyro_data = df[gyro_cols].values
                transformed_gyro = transform_gyro_data(gyro_data, rotation_matrix, inverse=True)
                
                for i, col in enumerate(gyro_cols):
                    df[col] = transformed_gyro[:, i]
                
                print(f"    ✓ Transformed {opensim_segment} gyro data")
                transformed_count += 1
            
            # Transform accel data
            accel_cols = find_accel_columns(df, imu_segment)
            if accel_cols is not None:
                accel_data = df[accel_cols].values
                transformed_accel = transform_gyro_data(accel_data, rotation_matrix, inverse=True)
                
                for i, col in enumerate(accel_cols):
                    df[col] = transformed_accel[:, i]
                
                print(f"    ✓ Transformed {opensim_segment} accel data")
        
        if transformed_count == 0:
            print(f"    ⚠️  No segments were transformed")
            return False
        
        if not dry_run:
            output_imu_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_imu_file, index=False)
            print(f"    ✓ Saved transformed IMU data: {output_imu_file.name}")
        else:
            print(f"    ✓ Dry run: Would transform {transformed_count} segments")
        
        return True
    
    except Exception as e:
        print(f"    ❌ Error transforming {input_imu_file}: {e}")
        return False


def transform_subject_data(
    dataset_root: Path,
    output_root: Path,
    subject: str,
    batch_results: Dict,
    dry_run: bool = False
) -> Tuple[int, int]:
    """Transform all trials for a specific subject using trial-specific rotation matrices."""
    subject_dir = dataset_root / subject
    if not subject_dir.exists():
        print(f"❌ Subject directory not found: {subject}")
        return 0, 0
    
    total = 0
    successful = 0
    
    print(f"\n📁 Processing subject: {subject}")
    
    # Copy opensim directory (models) for this subject
    opensim_src = subject_dir / "opensim"
    if opensim_src.exists() and not dry_run:
        opensim_dst = output_root / subject / "opensim"
        opensim_dst.parent.mkdir(parents=True, exist_ok=True)
        if not opensim_dst.exists():
            shutil.copytree(opensim_src, opensim_dst)
            print(f"  ✓ Copied OpenSim models")
    
    for condition_dir in sorted(subject_dir.iterdir()):
        if not condition_dir.is_dir() or condition_dir.name.startswith('.') or condition_dir.name == 'opensim':
            continue
        
        condition = condition_dir.name
        print(f"  📂 Condition: {condition}")
        
        for trial_dir in sorted(condition_dir.iterdir()):
            if not trial_dir.is_dir() or trial_dir.name.startswith('.'):
                continue
            
            trial = trial_dir.name
            input_imu_file = trial_dir / "Input" / "imu_data.csv"
            
            if not input_imu_file.exists():
                print(f"    ⚠️  IMU file not found: {trial}")
                continue
            
            print(f"    🔄 Trial: {trial}")
            total += 1
            
            # Get trial-specific rotation matrices
            rotation_matrices = get_trial_rotation_matrices(batch_results, subject, condition, trial)
            if rotation_matrices is None:
                print(f"    ❌ No rotation matrices found for {subject}/{condition}/{trial}")
                continue
            
            # Create output trial directory structure
            output_trial_dir = output_root / subject / condition / trial
            
            if not dry_run:
                # Copy all files except Input/imu_data.csv
                for item in trial_dir.rglob('*'):
                    if item.is_file():
                        relative_path = item.relative_to(trial_dir)
                        if relative_path == Path("Input/imu_data.csv"):
                            continue
                        dest_file = output_trial_dir / relative_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest_file)
            
            # Transform and save IMU data using trial-specific matrices
            output_imu_file = output_trial_dir / "Input" / "imu_data.csv"
            if transform_trial_imu_data(input_imu_file, output_imu_file, rotation_matrices, dry_run):
                successful += 1
    
    return successful, total


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Transform IMU data using batch optimization results")
    parser.add_argument("--dataset-root", required=True, help="Path to source dataset root")
    parser.add_argument("--output-root", required=True, help="Path to output dataset root")
    parser.add_argument("--batch-results", required=True, help="Path to batch_results.json")
    parser.add_argument("--subjects", help="Comma-separated subjects (default: all from batch)")
    parser.add_argument("--dry-run", action="store_true", help="Don't save changes, just show what would be done")
    
    args = parser.parse_args()
    
    # Load batch results
    batch_results = load_batch_results(Path(args.batch_results))
    if batch_results is None:
        return
    
    # Determine subjects to process
    if args.subjects:
        subjects_to_process = [s.strip() for s in args.subjects.split(',')]
        # Validate subjects exist in batch results
        available_subjects = list(batch_results['subjects'].keys())
        subjects_to_process = [s for s in subjects_to_process if s in available_subjects]
        if not subjects_to_process:
            print("❌ No valid subjects found in batch results")
            return
    else:
        subjects_to_process = list(batch_results['subjects'].keys())
    
    print(f"\n📋 Processing {len(subjects_to_process)} subjects: {subjects_to_process}")
    if args.dry_run:
        print(f"🔍 DRY RUN MODE - No changes will be saved")
    print("=" * 60)
    
    # Process each subject
    total_successful = 0
    total_trials = 0
    
    for subject in subjects_to_process:
        # Transform subject data (uses trial-specific matrices internally)
        successful, total = transform_subject_data(
            Path(args.dataset_root),
            Path(args.output_root),
            subject,
            batch_results,
            args.dry_run
        )
        
        total_successful += successful
        total_trials += total
        
        print(f"  📊 {subject}: {successful}/{total} trials processed")
    
    print(f"\n✅ Batch transformation complete!")
    print(f"   Processed: {total_successful}/{total_trials} trials successfully")
    if args.dry_run:
        print(f"   ⚠️  Dry run mode - no files were created")
    else:
        print(f"   Canonical dataset created at: {args.output_root}")


if __name__ == "__main__":
    main()
