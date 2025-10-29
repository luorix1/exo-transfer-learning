#!/usr/bin/env python3
"""
Complete workflow to create Canonical_MetaMobility dataset.

This script runs the full pipeline:
1. Batch IMU orientation optimization across all subjects
2. Transform IMU data using subject-specific rotation matrices
3. Create Canonical_MetaMobility dataset

Usage:
    # Run complete workflow
    python create_canonical_memo.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/MetaMobility" \
        --output-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Canonical_MetaMobility" \
        --conditions "0mps,1p0mps,1p2mps" \
        --segments "femur_r,tibia_r,pelvis"
    
    # Run on specific subjects
    python create_canonical_memo.py \
        --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/MetaMobility" \
        --output-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Canonical_MetaMobility" \
        --subjects "AB01,AB02,AB03" --conditions "1p2mps"
"""

import argparse
import subprocess
import sys
import shutil
import json
import os
from pathlib import Path
from tqdm import tqdm


def run_batch_optimization(
    dataset_root: Path,
    output_dir: Path,
    subjects: str,
    conditions: str,
    segments: str,
    max_frames: int = 100000,
    max_trials_per_subject: int = -1,
    debug: bool = False
) -> bool:
    """Run batch IMU optimization across specified trials for subjects."""
    print("🚀 Step 1: Running batch IMU optimization...")
    print("=" * 60)
    
    # Create a temporary results directory for optimization
    temp_results_dir = output_dir / "temp_optimization"
    temp_results_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        "analysis/batch_imu_optimization.py",
        "--dataset-root", str(dataset_root),
        "--output-root", str(temp_results_dir),
        "--segments", segments,
        "--max-frames", str(max_frames),
        "--max-trials-per-subject", str(max_trials_per_subject)
    ]
    
    if subjects:
        cmd.extend(["--subjects", subjects])
    
    if conditions:
        cmd.extend(["--conditions", conditions])
    
    if debug:
        cmd.append("--debug")
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Batch optimization completed successfully!")
        
        # Move results to final location
        final_results_dir = output_dir / "batch_results"
        if final_results_dir.exists():
            shutil.rmtree(final_results_dir)
        shutil.move(str(temp_results_dir), str(final_results_dir))
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Batch optimization failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False


def create_canonical_dataset(
    dataset_root: Path,
    output_root: Path,
    batch_results_file: Path,
    subjects: str,
    dry_run: bool = False
) -> bool:
    """Create canonical dataset by copying structure and transforming IMU data."""
    print("\n🚀 Step 2: Creating canonical dataset...")
    print("=" * 60)
    
    # Load batch results
    with open(batch_results_file, 'r') as f:
        batch_results = json.load(f)
    
    print(f"🔍 Debug: Batch results keys: {list(batch_results.keys())}")
    if 'subjects' in batch_results:
        print(f"🔍 Debug: Subjects in batch results: {list(batch_results['subjects'].keys())}")
    
    # Get list of subjects to process
    if subjects:
        subject_list = [s.strip() for s in subjects.split(',')]
    else:
        # Look for subjects in the 'subjects' key of batch results
        if 'subjects' in batch_results:
            subject_list = list(batch_results['subjects'].keys())
        else:
            subject_list = list(batch_results.keys())
    
    print(f"📋 Processing {len(subject_list)} subjects: {subject_list}")
    
    # Count total trials for progress bar
    total_trials = 0
    for subject in subject_list:
        if 'subjects' in batch_results:
            subject_data = batch_results['subjects'].get(subject)
        else:
            subject_data = batch_results.get(subject)
            
        if subject_data:
            subject_dir = dataset_root / subject
            if subject_dir.exists():
                for condition in subject_dir.iterdir():
                    if condition.is_dir():
                        for trial in condition.iterdir():
                            if trial.is_dir():
                                total_trials += 1
    
    print(f"📊 Total trials to process: {total_trials}")
    
    # Create progress bar for subjects
    subject_pbar = tqdm(subject_list, desc="Processing subjects", unit="subject")
    
    for subject in subject_pbar:
        # Look for subject in the correct location
        if 'subjects' in batch_results:
            subject_data = batch_results['subjects'].get(subject)
        else:
            subject_data = batch_results.get(subject)
            
        if not subject_data:
            print(f"⚠️  Subject {subject} not found in batch results, skipping")
            continue
            
        subject_pbar.set_description(f"Processing subject: {subject}")
        
        # Copy .osim files for this subject (only once per subject)
        copy_osim_files(dataset_root, output_root, subject, dry_run)
        
        # Find all trials for this subject in the dataset
        subject_dir = dataset_root / subject
        if not subject_dir.exists():
            subject_pbar.write(f"⚠️  Subject directory not found: {subject_dir}, skipping")
            continue
            
        # Get conditions for this subject
        conditions = [c for c in subject_dir.iterdir() if c.is_dir()]
        condition_pbar = tqdm(conditions, desc=f"  {subject} conditions", unit="condition", leave=False)
        
        for condition in condition_pbar:
            condition_pbar.set_description(f"  {subject}/{condition.name}")
            
            # Get trials for this condition
            trials = [t for t in condition.iterdir() if t.is_dir()]
            trial_pbar = tqdm(trials, desc=f"    {condition.name} trials", unit="trial", leave=False)
            
            for trial in trial_pbar:
                trial_pbar.set_description(f"    {trial.name}")
                
                # Create output directory structure
                output_trial_dir = output_root / subject / condition.name / trial.name
                output_trial_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy all files except Input/imu_data.csv
                for item in trial.iterdir():
                    if item.is_file():
                        # Copy file
                        if not dry_run:
                            shutil.copy2(item, output_trial_dir / item.name)
                    elif item.is_dir():
                        # Copy directory
                        output_item_dir = output_trial_dir / item.name
                        if not dry_run:
                            shutil.copytree(item, output_item_dir, dirs_exist_ok=True)
                        
                        # If this is the Input directory, transform the IMU data
                        if item.name == "Input":
                            success = transform_trial_imu_data(
                                trial, output_trial_dir, batch_results, subject, 
                                condition.name, trial.name, dry_run
                            )
                            if success:
                                trial_pbar.set_postfix({"status": "✓ transformed"})
                            else:
                                trial_pbar.set_postfix({"status": "⚠️ failed"})
                
                # Save optimization results for this trial
                save_trial_optimization_results(
                    output_trial_dir, batch_results, subject, 
                    condition.name, trial.name, dry_run
                )
            
            # Close nested progress bars
            trial_pbar.close()
        condition_pbar.close()
    
    # Close main progress bar
    subject_pbar.close()
    
    print("✅ Canonical dataset creation completed!")
    return True


def copy_osim_files(dataset_root: Path, output_root: Path, subject: str, dry_run: bool = False):
    """Copy .osim files to opensim folder for the subject."""
    try:
        # Create opensim directory for this subject
        subject_opensim_dir = output_root / subject / "opensim"
        if not dry_run:
            subject_opensim_dir.mkdir(parents=True, exist_ok=True)
        
        # Look for .osim files in the original dataset
        source_subject_dir = dataset_root / subject
        if not source_subject_dir.exists():
            print(f"    ⚠️  Source subject directory not found: {source_subject_dir}")
            return False
        
        # Search for .osim files in the subject directory and subdirectories
        osim_files = []
        for root, dirs, files in os.walk(source_subject_dir):
            for file in files:
                if file.endswith('.osim'):
                    osim_files.append(Path(root) / file)
        
        if not osim_files:
            print(f"    ⚠️  No .osim files found for subject {subject}")
            return False
        
        # Copy the first .osim file found (assuming there's one main model file)
        source_osim = osim_files[0]
        target_osim = subject_opensim_dir / f"{subject}.osim"
        
        if not dry_run:
            shutil.copy2(source_osim, target_osim)
            print(f"    ✓ Copied .osim file: {source_osim} -> {target_osim}")
        else:
            print(f"    🔍 Would copy .osim file: {source_osim} -> {target_osim}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error copying .osim files: {e}")
        return False


def save_trial_optimization_results(
    output_trial_dir: Path,
    batch_results: dict,
    subject: str,
    condition: str,
    trial: str,
    dry_run: bool = False
) -> bool:
    """Save optimization results for a specific trial."""
    try:
        # Get optimization results for this trial
        if 'subjects' in batch_results:
            subject_data = batch_results['subjects'].get(subject)
        else:
            subject_data = batch_results.get(subject)
            
        if not subject_data or condition not in subject_data or trial not in subject_data[condition]:
            return False
            
        trial_results = subject_data[condition][trial]
        
        # Create optimization results directory
        optimization_dir = output_trial_dir / "optimization"
        if not dry_run:
            optimization_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trial-specific results
        trial_results_file = optimization_dir / "trial_results.json"
        if not dry_run:
            import json
            with open(trial_results_file, 'w') as f:
                json.dump(trial_results, f, indent=2)
            print(f"    💾 Saved optimization results: {trial_results_file}")
        else:
            print(f"    🔍 Would save optimization results to: {trial_results_file}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error saving optimization results: {e}")
        return False


def transform_trial_imu_data(
    source_trial_dir: Path,
    output_trial_dir: Path,
    batch_results: dict,
    subject: str,
    condition: str,
    trial: str,
    dry_run: bool = False
) -> bool:
    """Transform IMU data for a specific trial."""
    try:
        # Get rotation matrices for this trial
        if 'subjects' in batch_results:
            subject_data = batch_results['subjects'].get(subject)
        else:
            subject_data = batch_results.get(subject)
            
        if not subject_data:
            print(f"    ⚠️  No rotation matrices found for {subject}")
            return False
            
        if condition not in subject_data:
            print(f"    ⚠️  No rotation matrices found for {subject}/{condition}")
            return False
            
        if trial not in subject_data[condition]:
            print(f"    ⚠️  No rotation matrices found for {subject}/{condition}/{trial}")
            return False
            
        trial_results = subject_data[condition][trial]
        
        # Load original IMU data
        imu_file = source_trial_dir / "Input" / "imu_data.csv"
        if not imu_file.exists():
            print(f"    ⚠️  IMU data file not found: {imu_file}")
            return False
            
        import pandas as pd
        import numpy as np
        
        # Try to read CSV with different parameters to handle potential data issues
        try:
            df = pd.read_csv(imu_file)
        except Exception as e:
            print(f"    ⚠️  Error reading CSV with default parameters: {e}")
            # Try with different parameters
            try:
                df = pd.read_csv(imu_file, dtype=str)
                # Convert numeric columns back to float
                for col in df.columns:
                    if col != 'time':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"    ✓ Successfully loaded CSV with string dtype workaround")
            except Exception as e2:
                print(f"    ❌ Failed to read CSV even with workaround: {e2}")
                return False
        
        # Transform IMU data for each segment
        segments_data = trial_results.get("segments", {})
        print(f"    🔍 Found {len(segments_data)} segments: {list(segments_data.keys())}")
        for segment, segment_data in segments_data.items():
            rotation_matrix = segment_data.get("optimal_rotation_matrix")
            if not rotation_matrix:
                print(f"    ⚠️  No rotation matrix found for {segment}")
                continue
            print(f"    🔍 Processing segment {segment} with rotation matrix shape: {np.array(rotation_matrix).shape}")
            
            # Map segments to MetaMobility column names
            if segment == "pelvis":
                gyro_cols = ["pelvis_gyro_x", "pelvis_gyro_y", "pelvis_gyro_z"]
                accel_cols = ["pelvis_accel_x", "pelvis_accel_y", "pelvis_accel_z"]
            elif segment == "femur_r":
                gyro_cols = ["femur_r_gyro_x", "femur_r_gyro_y", "femur_r_gyro_z"]
                accel_cols = ["femur_r_accel_x", "femur_r_accel_y", "femur_r_accel_z"]
            elif segment == "tibia_r":
                gyro_cols = ["tibia_r_gyro_x", "tibia_r_gyro_y", "tibia_r_gyro_z"]
                accel_cols = ["tibia_r_accel_x", "tibia_r_accel_y", "tibia_r_accel_z"]
            elif segment == "femur_l":
                gyro_cols = ["femur_l_gyro_x", "femur_l_gyro_y", "femur_l_gyro_z"]
                accel_cols = ["femur_l_accel_x", "femur_l_accel_y", "femur_l_accel_z"]
            elif segment == "tibia_l":
                gyro_cols = ["tibia_l_gyro_x", "tibia_l_gyro_y", "tibia_l_gyro_z"]
                accel_cols = ["tibia_l_accel_x", "tibia_l_accel_y", "tibia_l_accel_z"]
            else:
                print(f"    ⚠️  Unknown segment: {segment}")
                continue
                
            # Check if all required columns exist
            missing_gyro_cols = [col for col in gyro_cols if col not in df.columns]
            missing_accel_cols = [col for col in accel_cols if col not in df.columns]
            
            if missing_gyro_cols:
                print(f"    ⚠️  Missing gyro columns for {segment}: {missing_gyro_cols}")
                continue
                
            if missing_accel_cols:
                print(f"    ⚠️  Missing accel columns for {segment}: {missing_accel_cols}")
                continue
                
            # Extract and transform gyro data
            gyro_data = df[gyro_cols].values
            R = np.array(rotation_matrix, dtype=np.float64)
            transformed_gyro = gyro_data @ R.T
            
            # Extract and transform accel data
            accel_data = df[accel_cols].values
            transformed_accel = accel_data @ R.T
            
            # Update dataframe
            for i, col in enumerate(gyro_cols):
                df[col] = transformed_gyro[:, i]
                
            for i, col in enumerate(accel_cols):
                df[col] = transformed_accel[:, i]
                
            print(f"    ✓ Transformed {segment} IMU data (gyro + accel)")
        
        # Save transformed IMU data
        if not dry_run:
            output_imu_file = output_trial_dir / "Input" / "imu_data.csv"
            output_imu_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_imu_file, index=False)
            print(f"    💾 Saved transformed IMU data: {output_imu_file}")
        else:
            print(f"    🔍 Would save transformed IMU data to: {output_trial_dir / 'Input' / 'imu_data.csv'}")
            
        return True
        
    except Exception as e:
        import traceback
        print(f"    ❌ Error transforming IMU data: {e}")
        print(f"    🔍 Traceback: {traceback.format_exc()}")
        return False


def copy_femur_r_optimization_assets(
    output_root: Path,
    dry_run: bool = False
) -> None:
    """Copy femur_r optimization artifacts from batch_results to each trial folder.

    Expected source layout (created by the optimization step):
      output_root/batch_results/subjects/<subject>/<condition>/<trial>/segments/femur_r/

    Target layout in canonical dataset:
      output_root/<subject>/<condition>/<trial>/femur_r/
    """
    batch_results_dir = output_root / "batch_results"
    if not batch_results_dir.exists():
        print(f"⚠️  Batch results directory not found: {batch_results_dir}")
        return

    # Determine whether results are nested under a top-level 'subjects' key
    subjects_root = batch_results_dir / "subjects"
    if not subjects_root.exists():
        subjects_root = batch_results_dir

    # Walk subjects/conditions/trials
    for subject_dir in subjects_root.iterdir():
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name

        for condition_dir in subject_dir.iterdir():
            if not condition_dir.is_dir():
                continue
            condition = condition_dir.name

            for trial_dir in condition_dir.iterdir():
                if not trial_dir.is_dir():
                    continue
                trial = trial_dir.name

                femur_r_src = trial_dir / "segments" / "femur_r"
                if not femur_r_src.exists():
                    # Some pipelines may store per-segment outputs directly under the trial
                    alt_src = trial_dir / "femur_r"
                    femur_r_src = alt_src if alt_src.exists() else None

                if femur_r_src is None or not femur_r_src.exists():
                    # Nothing to copy for this trial
                    continue

                target_dir = output_root / subject / condition / trial / "femur_r"
                if dry_run:
                    print(f"🔍 Would copy femur_r artifacts: {femur_r_src} -> {target_dir}")
                    continue

                try:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    # Copy all files and directories from source into target (merge)
                    for item in femur_r_src.iterdir():
                        src_path = item
                        dst_path = target_dir / item.name
                        if src_path.is_dir():
                            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src_path, dst_path)
                    print(f"    ✓ Copied femur_r optimization assets: {femur_r_src} -> {target_dir}")
                except Exception as e:
                    print(f"    ❌ Error copying femur_r optimization assets for {subject}/{condition}/{trial}: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create Canonical_MetaMobility dataset")
    parser.add_argument("--dataset-root", required=True, help="Path to Final/MetaMobility dataset")
    parser.add_argument("--output-root", required=True, help="Path to output Canonical_MetaMobility dataset")
    parser.add_argument("--subjects", help="Comma-separated subjects (default: all)")
    parser.add_argument("--conditions", help="Comma-separated conditions (default: all)")
    parser.add_argument("--segments", default="femur_r,tibia_r,pelvis", help="Comma-separated segments")
    parser.add_argument("--max-frames", type=int, default=100000, help="Max frames (default: 100000)")
    parser.add_argument("--max-trials-per-subject", type=int, default=-1, help="Max trials per subject (default: -1 = all)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--dry-run", action="store_true", help="Don't save changes, just show what would be done")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip optimization, use existing results")
    parser.add_argument("--results-dir", help="Path to existing results directory (if skipping optimization)")
    
    args = parser.parse_args()
    
    # Validate inputs
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"❌ Dataset root not found: {dataset_root}")
        return
    
    output_root = Path(args.output_root)
    
    # Determine results directory
    if args.skip_optimization:
        if not args.results_dir:
            print("❌ Must specify --results-dir when skipping optimization")
            return
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"❌ Results directory not found: {results_dir}")
            return
    else:
        # Use output-root for results when not skipping optimization
        results_dir = output_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📋 Creating Canonical_MetaMobility dataset")
    print(f"   Source: {dataset_root}")
    print(f"   Output: {output_root}")
    print(f"   Results: {results_dir}")
    print(f"   Segments: {args.segments}")
    if args.subjects:
        print(f"   Subjects: {args.subjects}")
    if args.conditions:
        print(f"   Conditions: {args.conditions}")
    if args.max_trials_per_subject > 0:
        print(f"   Max trials per subject: {args.max_trials_per_subject}")
    if args.dry_run:
        print(f"   🔍 DRY RUN MODE - No changes will be saved")
    print("=" * 60)
    
    # Step 1: Run batch optimization (unless skipped)
    if not args.skip_optimization:
        success = run_batch_optimization(
            dataset_root=dataset_root,
            output_dir=results_dir,
            subjects=args.subjects,
            conditions=args.conditions,
            segments=args.segments,
            max_frames=args.max_frames,
            max_trials_per_subject=args.max_trials_per_subject,
            debug=args.debug
        )
        
        if not success:
            print("❌ Batch optimization failed, stopping workflow")
            return
    else:
        print("⏭️  Skipping optimization step (using existing results)")
    
    # Step 2: Create canonical dataset
    if args.skip_optimization:
        batch_results_file = results_dir / "batch_results.json"
    else:
        batch_results_file = results_dir / "batch_results" / "batch_results.json"
    
    if not batch_results_file.exists():
        print(f"❌ Batch results file not found: {batch_results_file}")
        return
    
    success = create_canonical_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        batch_results_file=batch_results_file,
        subjects=args.subjects,
        dry_run=args.dry_run
    )
    
    if not success:
        print("❌ Canonical dataset creation failed")
        return
    
    # Step 3: Copy batch results to output directory
    print("\n🚀 Step 3: Copying batch results...")
    print("=" * 60)
    
    if not args.skip_optimization:
        batch_results_source = results_dir / "batch_results"
        batch_results_target = output_root / "batch_results"
    else:
        batch_results_source = results_dir
        batch_results_target = output_root / "batch_results"
    
    if batch_results_source.exists():
        if not args.dry_run:
            if batch_results_target.exists():
                shutil.rmtree(batch_results_target)
            shutil.copytree(batch_results_source, batch_results_target)
            print(f"✅ Copied batch results: {batch_results_source} -> {batch_results_target}")
        else:
            print(f"🔍 Would copy batch results: {batch_results_source} -> {batch_results_target}")
    else:
        print(f"⚠️  Batch results source not found: {batch_results_source}")
    
    # Step 4: Copy femur_r optimization assets into each corresponding trial folder
    print("\n🚀 Step 4: Copying femur_r optimization assets to canonical dataset...")
    copy_femur_r_optimization_assets(output_root=output_root, dry_run=args.dry_run)

    print(f"\n🎉 Canonical_MetaMobility dataset creation complete!")
    print(f"   📁 Output dataset: {output_root}")
    print(f"   📊 Batch results: {output_root / 'batch_results'}")
    if args.dry_run:
        print(f"   ⚠️  Dry run mode - no files were created")


if __name__ == "__main__":
    main()
