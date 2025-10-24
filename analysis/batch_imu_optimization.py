#!/usr/bin/env python3
"""
Batch IMU orientation optimization for multiple subjects.

This script runs IMU orientation optimization across multiple subjects in a dataset,
saving individual rotation matrices for each subject. The results can then be used
with transform_imu_to_opensim_frame.py to create canonical datasets.

Usage:
    # Run on all subjects with specific condition/trial
    python batch_imu_optimization.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Camargo" \
        --output-root "results/camargo_batch" \
        --condition "levelground" --trial "LG_C0p0_S0p0_BT_1_10" \
        --segments "femur_r,tibia_r,pelvis"
    
    # Run on specific subjects
    python batch_imu_optimization.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Camargo" \
        --output-root "results/camargo_batch" \
        --subjects "AB06,AB07,AB08" \
        --condition "levelground" --trial "LG_C0p0_S0p0_BT_1_10" \
        --segments "all"
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def find_subjects(dataset_root: Path, subjects: Optional[List[str]] = None) -> List[str]:
    """
    Find subjects in the dataset.
    
    Args:
        dataset_root: Path to dataset root
        subjects: Optional list of specific subjects to process
    
    Returns:
        List of subject names
    """
    if subjects:
        # Validate that specified subjects exist
        existing_subjects = []
        for subject in subjects:
            subject_dir = dataset_root / subject
            if subject_dir.exists() and subject_dir.is_dir():
                existing_subjects.append(subject)
            else:
                print(f"⚠️  Subject directory not found: {subject}")
        return existing_subjects
    
    # Find all subjects
    all_subjects = []
    for item in dataset_root.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            all_subjects.append(item.name)
    
    return sorted(all_subjects)


def run_single_trial_optimization(
    dataset_root: Path,
    subject: str,
    condition: str,
    trial: str,
    output_dir: Path,
    segments: str,
    max_frames: int = 100000,
    gyro_in_degrees: bool = False,
    debug: bool = False
) -> bool:
    """
    Run IMU optimization for a single trial.
    
    Args:
        dataset_root: Path to dataset root
        subject: Subject ID
        condition: Condition name
        trial: Trial name
        output_dir: Output directory for this trial
        segments: Comma-separated segments or 'all'
        max_frames: Maximum frames to process
        gyro_in_degrees: Whether gyro data is in degrees
        debug: Enable debug mode
    
    Returns:
        True if successful, False otherwise
    """
    # Create trial-specific output directory
    trial_output_dir = output_dir / subject / condition / trial
    trial_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "imu_orientation_optimization.py"),
        "--dataset-root", str(dataset_root),
        "--subject", subject,
        "--condition", condition,
        "--trial", trial,
        "--output", str(trial_output_dir),
        "--segments", segments,
        "--max-frames", str(max_frames)
    ]
    
    if gyro_in_degrees:
        cmd.append("--gyro-in-degrees")
    
    if debug:
        cmd.append("--debug")
    
    print(f"🚀 Running optimization for {subject}/{condition}/{trial}...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {subject}/{condition}/{trial} optimization completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {subject}/{condition}/{trial} optimization failed with return code {e.returncode}")
        print(f"   Error output: {e.stderr}")
        return False


def collect_rotation_matrices(output_dir: Path, subjects: List[str]) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Collect rotation matrices from all trial results.
    
    Args:
        output_dir: Root output directory
        subjects: List of subjects processed
    
    Returns:
        Dict mapping subject -> condition -> trial -> segment -> rotation matrix
    """
    all_matrices = {}
    
    for subject in subjects:
        subject_dir = output_dir / subject
        if not subject_dir.exists():
            print(f"⚠️  Subject directory not found: {subject}")
            continue
        
        print(f"📊 Loading results for {subject}")
        subject_matrices = {}
        
        # Look for condition directories
        for condition_dir in subject_dir.iterdir():
            if not condition_dir.is_dir() or condition_dir.name.startswith('.'):
                continue
            
            condition = condition_dir.name
            condition_matrices = {}
            
            # Look for trial directories
            for trial_dir in condition_dir.iterdir():
                if not trial_dir.is_dir() or trial_dir.name.startswith('.'):
                    continue
                
                trial = trial_dir.name
                trial_matrices = {}
                
                # Look for combined_results.json (multi-segment)
                combined_file = trial_dir / "combined_results.json"
                if combined_file.exists():
                    try:
                        with open(combined_file, 'r') as f:
                            results = json.load(f)
                        
                        for segment_name, segment_data in results['segments'].items():
                            rotation_matrix = np.array(segment_data['optimal_rotation_matrix'])
                            trial_matrices[segment_name] = rotation_matrix
                        
                        print(f"   ✓ Loaded combined results for {subject}/{condition}/{trial}")
                        
                    except Exception as e:
                        print(f"   ❌ Error loading combined results for {subject}/{condition}/{trial}: {e}")
                        continue
                
                # Look for individual optimization_results.json files
                else:
                    results_files = list(trial_dir.glob("**/optimization_results.json"))
                    if results_files:
                        for results_file in results_files:
                            try:
                                with open(results_file, 'r') as f:
                                    results = json.load(f)
                                
                                # Extract segment name from path or results
                                segment_name = "femur_r"  # default
                                
                                # Try to get segment name from meta field first
                                if 'meta' in results and 'segment' in results['meta']:
                                    segment_name = results['meta']['segment']
                                elif 'configuration' in results and 'segment_name' in results['configuration']:
                                    segment_name = results['configuration']['segment_name']
                                else:
                                    # Try to extract from filename path
                                    parts = results_file.parts
                                    for part in parts:
                                        if part in ['femur_r', 'femur_l', 'tibia_r', 'tibia_l', 'pelvis', 'thigh_r', 'thigh_l', 'shank_r', 'shank_l']:
                                            segment_name = part
                                            break
                                
                                # Get rotation matrix from the new format
                                if 'result' in results and 'rotation_matrix' in results['result']:
                                    rotation_matrix = np.array(results['result']['rotation_matrix'])
                                elif 'optimization' in results and 'optimal_rotation_matrix' in results['optimization']:
                                    rotation_matrix = np.array(results['optimization']['optimal_rotation_matrix'])
                                else:
                                    print(f"   ❌ No rotation matrix found in {results_file}")
                                    continue
                                
                                trial_matrices[segment_name] = rotation_matrix
                                
                            except Exception as e:
                                print(f"   ❌ Error loading {results_file}: {e}")
                                continue
                        
                        if trial_matrices:
                            print(f"   ✓ Loaded {len(trial_matrices)} segments for {subject}/{condition}/{trial}")
                
                if trial_matrices:
                    condition_matrices[trial] = trial_matrices
            
            if condition_matrices:
                subject_matrices[condition] = condition_matrices
        
        if subject_matrices:
            all_matrices[subject] = subject_matrices
            total_trials = sum(len(condition_matrices) for condition_matrices in subject_matrices.values())
            print(f"   ✓ Loaded {total_trials} trials for {subject}")
    
    return all_matrices


def save_batch_results(
    output_dir: Path,
    all_matrices: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]],
    subjects: List[str]
) -> Path:
    """
    Save batch results in a format compatible with transform_imu_to_opensim_frame.py.
    
    Args:
        output_dir: Root output directory
        all_matrices: Dict mapping subject -> condition -> trial -> segment -> rotation matrix
        subjects: List of subjects processed
    
    Returns:
        Path to the saved batch results file
    """
    batch_results = {
        "metadata": {
            "dataset": "Camargo",
            "subjects": subjects,
            "total_subjects": len(subjects),
            "trials_per_subject": {}
        },
        "subjects": {}
    }
    
    # Organize by subject -> condition -> trial -> segment
    for subject, subject_matrices in all_matrices.items():
        batch_results["subjects"][subject] = {}
        total_trials = 0
        
        for condition, condition_matrices in subject_matrices.items():
            batch_results["subjects"][subject][condition] = {}
            
            for trial, trial_matrices in condition_matrices.items():
                batch_results["subjects"][subject][condition][trial] = {
                    "segments": {}
                }
                
                for segment_name, rotation_matrix in trial_matrices.items():
                    batch_results["subjects"][subject][condition][trial]["segments"][segment_name] = {
                        "optimal_rotation_matrix": rotation_matrix.tolist(),
                        "determinant": float(np.linalg.det(rotation_matrix))
                    }
                
                total_trials += 1
        
        batch_results["metadata"]["trials_per_subject"][subject] = total_trials
    
    # Save batch results
    batch_file = output_dir / "batch_results.json"
    with open(batch_file, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"📊 Saved batch results: {batch_file}")
    return batch_file


def find_trials_for_subject(dataset_root: Path, subject: str, conditions: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Find all condition/trial pairs for a subject.
    
    Args:
        dataset_root: Path to dataset root
        subject: Subject ID
        conditions: Optional list of conditions to filter by
    
    Returns:
        List of (condition, trial) tuples
    """
    subject_dir = dataset_root / subject
    if not subject_dir.exists():
        return []
    
    trials = []
    for condition_dir in subject_dir.iterdir():
        if not condition_dir.is_dir() or condition_dir.name.startswith('.') or condition_dir.name == 'opensim':
            continue
        
        condition = condition_dir.name
        
        # Filter by conditions if specified
        if conditions and condition not in conditions:
            continue
        
        for trial_dir in condition_dir.iterdir():
            if not trial_dir.is_dir() or trial_dir.name.startswith('.'):
                continue
            
            trial = trial_dir.name
            # Check if IMU data exists
            imu_file = trial_dir / "Input" / "imu_data.csv"
            if imu_file.exists():
                trials.append((condition, trial))
    
    return trials


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch IMU orientation optimization")
    parser.add_argument("--dataset-root", required=True, help="Path to dataset root")
    parser.add_argument("--output-root", required=True, help="Path to output root")
    parser.add_argument("--subjects", help="Comma-separated subjects (default: all)")
    parser.add_argument("--conditions", help="Comma-separated conditions (default: all)")
    parser.add_argument("--segments", default="all", help="Comma-separated segments or 'all'")
    parser.add_argument("--max-frames", type=int, default=100000, help="Max frames (default: 100000)")
    parser.add_argument("--gyro-in-degrees", action="store_true", help="Use degrees for gyro data")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--max-trials-per-subject", type=int, default=-1, help="Max trials per subject (default: -1 = all)")
    
    args = parser.parse_args()
    
    # Parse subjects and conditions
    subjects = [s.strip() for s in args.subjects.split(',')] if args.subjects else None
    conditions = [c.strip() for c in args.conditions.split(',')] if args.conditions else None
    
    # Find subjects to process
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"❌ Dataset root not found: {dataset_root}")
        return
    
    subjects_to_process = find_subjects(dataset_root, subjects)
    if not subjects_to_process:
        print("❌ No subjects found to process")
        return
    
    print(f"📋 Found {len(subjects_to_process)} subjects: {subjects_to_process}")
    print(f"📊 Segments: {args.segments}")
    if conditions:
        print(f"📊 Conditions: {conditions}")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each subject and all their trials
    successful_trials = []
    failed_trials = []
    
    for i, subject in enumerate(subjects_to_process, 1):
        print(f"\n[{i}/{len(subjects_to_process)}] Processing {subject}")
        
        # Find all trials for this subject
        trials = find_trials_for_subject(dataset_root, subject, conditions)
        if not trials:
            print(f"   ⚠️  No trials found for {subject}")
            continue
        
        # Limit trials if specified
        if args.max_trials_per_subject > 0:
            trials = trials[:args.max_trials_per_subject]
        
        print(f"   📊 Found {len(trials)} trials for {subject}")
        
        # Process each trial
        for j, (condition, trial) in enumerate(trials, 1):
            print(f"   [{j}/{len(trials)}] Processing {subject}/{condition}/{trial}")
            
            success = run_single_trial_optimization(
                dataset_root=dataset_root,
                subject=subject,
                condition=condition,
                trial=trial,
                output_dir=output_dir,
                segments=args.segments,
                max_frames=args.max_frames,
                gyro_in_degrees=args.gyro_in_degrees,
                debug=args.debug
            )
            
            if success:
                successful_trials.append((subject, condition, trial))
            else:
                failed_trials.append((subject, condition, trial))
    
    print(f"\n📊 Batch processing complete!")
    print(f"   ✅ Successful: {len(successful_trials)} trials")
    print(f"   ❌ Failed: {len(failed_trials)} trials")
    
    if failed_trials:
        print(f"   Failed trials: {failed_trials[:10]}{'...' if len(failed_trials) > 10 else ''}")
    
    if successful_trials:
        # Collect rotation matrices
        print(f"\n📊 Collecting rotation matrices...")
        all_matrices = collect_rotation_matrices(output_dir, subjects_to_process)
        
        if all_matrices:
            # Save batch results
            batch_file = save_batch_results(output_dir, all_matrices, subjects_to_process)
            
            print(f"\n✅ Batch results saved!")
            print(f"   📁 Output directory: {output_dir}")
            print(f"   📊 Batch results file: {batch_file}")
            print(f"\n🚀 Next steps:")
            print(f"   Use transform_batch_results.py with --batch-results {batch_file}")
        else:
            print("❌ No rotation matrices found to save")
    else:
        print("❌ No successful optimizations to save")


if __name__ == "__main__":
    main()
