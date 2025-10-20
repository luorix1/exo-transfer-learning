#!/usr/bin/env python3
"""
Transform real IMU data to OpenSim reference frame using optimized rotation matrices.

This script applies the rotation matrices found during IMU orientation optimization
to transform the real IMU gyro data from the IMU's local frame to the OpenSim
reference frame (x: forward, y: up, z: right for each body segment).

The optimization process finds R such that:
    R * sim_data ≈ real_data
    
To transform real_data to the same frame as sim_data (OpenSim frame), we apply:
    transformed_real = R^T * real_data
    
This is because R transforms from OpenSim frame to IMU frame, so R^T (inverse)
transforms from IMU frame to OpenSim frame.

Usage:
    # Create new canonical dataset
    python transform_imu_to_opensim_frame.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Camargo" \
        --output-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Canonical_Camargo" \
        --results-file "results/camargo_analysis/optimization_results.json"
    
    # Batch transform
    python transform_imu_to_opensim_frame.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Molinaro_Phase1_Phase2" \
        --output-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Canonical_Molinaro" \
        --results-dir "results/molinaro_analysis/"
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_rotation_matrices(results_file: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Load optimal rotation matrices from optimization results JSON.
    Supports both single-segment and multi-segment formats.
    
    Returns:
        Dict mapping segment names to rotation matrices, or None on error
    """
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        rotation_matrices = {}
        
        # Check if this is a combined results file (multi-segment)
        if 'segments' in results:
            print(f"📊 Loading multi-segment rotation matrices from: {results_file}")
            for segment_name, segment_data in results['segments'].items():
                rotation_matrix = np.array(segment_data['optimal_rotation_matrix'])
                
                # Verify it's a valid rotation matrix
                det = np.linalg.det(rotation_matrix)
                if not np.isclose(det, 1.0, atol=0.01):
                    print(f"⚠️  Warning: Rotation matrix for {segment_name} has determinant {det:.6f}, expected 1.0")
                
                rotation_matrices[segment_name] = rotation_matrix
                print(f"   ✓ Loaded rotation matrix for: {segment_name}")
        
        # Single segment format
        elif 'optimization' in results:
            rotation_matrix = np.array(results['optimization']['optimal_rotation_matrix'])
            
            # Verify it's a valid rotation matrix
            det = np.linalg.det(rotation_matrix)
            if not np.isclose(det, 1.0, atol=0.01):
                print(f"⚠️  Warning: Rotation matrix determinant is {det:.6f}, expected 1.0")
            
            # Try to infer segment name from file path or configuration
            segment_name = "femur_r"  # default
            if 'configuration' in results and 'segment_name' in results['configuration']:
                segment_name = results['configuration']['segment_name']
            
            rotation_matrices[segment_name] = rotation_matrix
            print(f"📊 Loading single-segment rotation matrix from: {results_file}")
            print(f"   ✓ Loaded rotation matrix for: {segment_name}")
        
        else:
            print(f"❌ Unrecognized results file format: {results_file}")
            return None
        
        return rotation_matrices
    
    except Exception as e:
        print(f"❌ Error loading rotation matrices from {results_file}: {e}")
        return None


def transform_gyro_data(gyro_data: np.ndarray, rotation_matrix: np.ndarray, inverse: bool = True) -> np.ndarray:
    """
    Transform gyro data using rotation matrix.
    
    Args:
        gyro_data: Nx3 array of gyro data
        rotation_matrix: 3x3 rotation matrix
        inverse: If True, apply R^T (transform from IMU frame to OpenSim frame)
                 If False, apply R (transform from OpenSim frame to IMU frame)
    
    Returns:
        transformed_data: Nx3 array of transformed gyro data
    """
    if inverse:
        # Apply inverse rotation: R^T * data
        transform = rotation_matrix.T
    else:
        # Apply forward rotation: R * data
        transform = rotation_matrix
    
    # Apply transformation to each frame
    transformed_data = np.array([transform @ frame for frame in gyro_data])
    
    return transformed_data


def find_gyro_columns(df: pd.DataFrame, segment: str = "thigh_r") -> Optional[List[str]]:
    """
    Find gyro columns for a specific segment in the dataframe.
    
    Args:
        df: DataFrame with IMU data
        segment: Segment name (e.g., "thigh_r", "thigh_l")
    
    Returns:
        List of column names [x, y, z] or None if not found
    """
    # Try different naming conventions
    candidates = [
        [f"{segment}_gyro_x", f"{segment}_gyro_y", f"{segment}_gyro_z"],
        [f"{segment.upper()}_GYROX", f"{segment.upper()}_GYROY", f"{segment.upper()}_GYROZ"],
        [f"{segment}_Gyro_X", f"{segment}_Gyro_Y", f"{segment}_Gyro_Z"],
    ]
    
    # Case-insensitive search
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


def transform_trial_imu_data(
    input_imu_file: Path,
    output_imu_file: Path,
    rotation_matrices: Dict[str, np.ndarray],
    dry_run: bool = False
) -> bool:
    """
    Transform IMU data for a single trial (all segments) and save to output location.
    
    Args:
        input_imu_file: Path to source imu_data.csv
        output_imu_file: Path to destination imu_data.csv
        rotation_matrices: Dict mapping segment names to rotation matrices
        dry_run: If True, don't save changes
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load IMU data
        df = pd.read_csv(input_imu_file)
        
        transformed_count = 0
        
        # Transform each segment
        for opensim_segment, rotation_matrix in rotation_matrices.items():
            imu_segment = get_imu_segment_name(opensim_segment)
            
            # Find gyro columns for this segment
            gyro_cols = find_gyro_columns(df, imu_segment)
            if gyro_cols is None:
                print(f"  ⚠️  No gyro columns found for segment '{imu_segment}' (OpenSim: {opensim_segment})")
                continue
            
            print(f"  Found gyro columns for {opensim_segment}: {gyro_cols}")
            
            # Extract gyro data
            gyro_data = df[gyro_cols].values
            
            # Check for NaN values
            if np.any(np.isnan(gyro_data)):
                nan_count = np.sum(np.isnan(gyro_data))
                print(f"  ⚠️  Warning: {nan_count} NaN values found in {opensim_segment} gyro data")
            
            # Transform data (apply inverse rotation to go from IMU frame to OpenSim frame)
            transformed_data = transform_gyro_data(gyro_data, rotation_matrix, inverse=True)
            
            # Replace original gyro columns with transformed data
            for i, col in enumerate(gyro_cols):
                df[col] = transformed_data[:, i]
            
            # Print statistics
            print(f"    Original std: {np.std(gyro_data, axis=0)}")
            print(f"    Transformed std: {np.std(transformed_data, axis=0)}")
            
            transformed_count += 1
        
        if transformed_count == 0:
            print(f"  ⚠️  No segments were transformed")
            return False
        
        if not dry_run:
            # Ensure output directory exists
            output_imu_file.parent.mkdir(parents=True, exist_ok=True)
            # Save to output file
            df.to_csv(output_imu_file, index=False)
            print(f"  ✓ Transformed {transformed_count} segments and saved: {output_imu_file.relative_to(output_imu_file.parents[4])}")
        else:
            print(f"  ✓ Dry run: Would transform {transformed_count} segments to {output_imu_file.name}")
        
        return True
    
    except Exception as e:
        print(f"  ❌ Error transforming {input_imu_file}: {e}")
        return False


def copy_directory_structure(src: Path, dst: Path, exclude_patterns: Optional[List[str]] = None):
    """
    Copy directory structure excluding certain patterns.
    
    Args:
        src: Source directory
        dst: Destination directory
        exclude_patterns: List of patterns to exclude (e.g., ['Input/imu_data.csv'])
    """
    exclude_patterns = exclude_patterns or []
    
    for item in src.rglob('*'):
        if item.is_file():
            # Check if this file should be excluded
            relative_path = item.relative_to(src)
            should_exclude = any(pattern in str(relative_path) for pattern in exclude_patterns)
            
            if not should_exclude:
                dest_file = dst / relative_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_file)


def transform_dataset(
    dataset_root: Path,
    output_root: Path,
    rotation_matrices: Dict[str, np.ndarray],
    subjects: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    max_trials: int = -1,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Transform all IMU data in a dataset and create a new canonical dataset.
    
    Args:
        dataset_root: Path to source dataset root (e.g., Final/Camargo)
        output_root: Path to output dataset root (e.g., Canonical_Camargo)
        rotation_matrices: Dict mapping segment names to rotation matrices
        subjects: List of subjects to process (None = all)
        conditions: List of conditions to process (None = all)
        max_trials: Maximum trials to process (-1 = all)
        dry_run: If True, don't save changes
    
    Returns:
        (successful_count, total_count)
    """
    total = 0
    successful = 0
    
    if not dry_run:
        print(f"\n📋 Creating output dataset structure...")
        output_root.mkdir(parents=True, exist_ok=True)
    
    # Iterate through dataset structure: <dataset>/<subject>/<condition>/<trial>/Input/imu_data.csv
    for subject_dir in sorted(dataset_root.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
            continue
        
        subject = subject_dir.name
        
        # Filter by subjects
        if subjects and subject not in subjects:
            continue
        
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
            if not condition_dir.is_dir() or condition_dir.name.startswith('.'):
                continue
            
            condition = condition_dir.name
            
            # Skip opensim directory (already copied)
            if condition == 'opensim':
                continue
            
            # Filter by conditions
            if conditions and condition not in conditions:
                continue
            
            print(f"  📂 Condition: {condition}")
            
            trial_count = 0
            for trial_dir in sorted(condition_dir.iterdir()):
                if not trial_dir.is_dir() or trial_dir.name.startswith('.'):
                    continue
                
                # Check max trials limit
                if max_trials > 0 and trial_count >= max_trials:
                    break
                
                trial = trial_dir.name
                input_imu_file = trial_dir / "Input" / "imu_data.csv"
                
                if not input_imu_file.exists():
                    print(f"    ⚠️  IMU file not found: {trial}")
                    continue
                
                print(f"    🔄 Trial: {trial}")
                total += 1
                trial_count += 1
                
                # Create output trial directory structure
                output_trial_dir = output_root / subject / condition / trial
                
                if not dry_run:
                    # Copy all files except Input/imu_data.csv
                    for item in trial_dir.rglob('*'):
                        if item.is_file():
                            relative_path = item.relative_to(trial_dir)
                            # Skip the IMU data file (we'll create transformed version)
                            if relative_path == Path("Input/imu_data.csv"):
                                continue
                            dest_file = output_trial_dir / relative_path
                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dest_file)
                
                # Transform and save IMU data
                output_imu_file = output_trial_dir / "Input" / "imu_data.csv"
                if transform_trial_imu_data(input_imu_file, output_imu_file, rotation_matrices, dry_run):
                    successful += 1
    
    return successful, total


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Transform IMU data to OpenSim reference frame and create canonical dataset"
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to source dataset root (e.g., Final/Camargo)"
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Path to output dataset root (e.g., Canonical_Camargo)"
    )
    parser.add_argument(
        "--results-file",
        help="Path to optimization_results.json file"
    )
    parser.add_argument(
        "--results-dir",
        help="Path to results directory (will use combined_results.json if available)"
    )
    parser.add_argument(
        "--subjects",
        help="Comma-separated list of subjects to process (default: all)"
    )
    parser.add_argument(
        "--conditions",
        help="Comma-separated list of conditions to process (default: all)"
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=-1,
        help="Maximum trials per condition (default: -1 = all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save changes, just show what would be done"
    )
    
    args = parser.parse_args()
    
    # Parse subjects and conditions
    subjects = [s.strip() for s in args.subjects.split(',')] if args.subjects else None
    conditions = [c.strip() for c in args.conditions.split(',')] if args.conditions else None
    
    # Find rotation matrices
    rotation_matrices = None
    
    if args.results_file:
        results_path = Path(args.results_file)
        rotation_matrices = load_rotation_matrices(results_path)
    elif args.results_dir:
        results_dir = Path(args.results_dir)
        # Look for combined_results.json first, then individual optimization_results.json
        combined_file = results_dir / "combined_results.json"
        if combined_file.exists():
            print(f"📊 Found combined results file")
            rotation_matrices = load_rotation_matrices(combined_file)
        else:
            # Try to find optimization_results.json files
            results_files = list(results_dir.glob("**/optimization_results.json"))
            if results_files:
                results_path = results_files[0]
                rotation_matrices = load_rotation_matrices(results_path)
            else:
                print(f"❌ No results files found in {results_dir}")
                return
    else:
        print("❌ Must specify either --results-file or --results-dir")
        return
    
    if rotation_matrices is None or len(rotation_matrices) == 0:
        print("❌ Failed to load rotation matrices")
        return
    
    print(f"\n📊 Loaded {len(rotation_matrices)} segment rotation matrices:")
    for segment, matrix in rotation_matrices.items():
        print(f"   - {segment}: det = {np.linalg.det(matrix):.6f}")
    
    # Transform dataset
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"❌ Dataset root not found: {dataset_root}")
        return
    
    output_root = Path(args.output_root)
    
    print(f"\n🚀 Creating canonical dataset:")
    print(f"   Source: {dataset_root}")
    print(f"   Output: {output_root}")
    print(f"   Segments: {', '.join(rotation_matrices.keys())}")
    if args.dry_run:
        print(f"   🔍 DRY RUN MODE - No changes will be saved")
    print("=" * 60)
    
    successful, total = transform_dataset(
        dataset_root,
        output_root,
        rotation_matrices,
        subjects,
        conditions,
        args.max_trials,
        args.dry_run
    )
    
    print(f"\n✅ Transformation complete!")
    print(f"   Processed: {successful}/{total} trials successfully")
    if args.dry_run:
        print(f"   ⚠️  Dry run mode - no files were created")
    else:
        print(f"   Canonical dataset created at: {output_root}")


if __name__ == "__main__":
    main()

