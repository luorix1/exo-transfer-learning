#!/usr/bin/env python3
"""
Create synthetic Camargo IMU data using precomputed IMU orientations and OpenSim.

Overview
- Uses previously computed IMU mounting orientations (rotation matrices) from a batch
  optimization run (e.g., results/camargo_batch/batch_results.json).
- For each subject/condition/trial in the Camargo dataset, attaches virtual IMUs to
  the pelvis/femur_r/tibia_r (configurable) at each body origin using the optimized
  rotation, and computes clean gyro signals (angular velocity) in the IMU frame
  from the provided OpenSim states/motion file.

Requirements
- OpenSim Python API must be available on the target machine.

Inputs
- --dataset-root: Path to the original Camargo dataset root (subjects/conditions/trials).
- --output-root: Path to output Canonical dataset where synthetic Input/imu_data.csv
  will be written. This script will mirror the dataset structure if needed.
- --results-dir: Path to batch_results directory containing batch_results.json with
  per-trial rotation matrices for segments.
- --model-root: Directory containing subject-specific .osim models (e.g., Canonical
  dataset subject/opensim/<SUBJECT>.osim). If not provided, the script attempts to
  find a .osim within the subject tree under dataset-root.
- --states-glob: Glob pattern to locate the OpenSim states/motion file for each trial
  (e.g., "*states.sto" or "*kinematics.mot"). The first match will be used.

Notes
- The script only generates gyro signals (rad/s). If you want accel, extend the
  IMU computation accordingly.
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np


def try_import_opensim():
    try:
        import opensim as osim  # type: ignore
        return osim
    except Exception as e:
        print("ERROR: OpenSim Python API is not available. Install OpenSim >= 4.3 and ensure PYTHONPATH is set.")
        print(f"Detail: {e}")
        sys.exit(2)


def load_batch_results(results_dir: Path) -> Dict:
    """Load batch_results.json from the results directory (supports nested subjects)."""
    # Support both results_dir/batch_results.json and results_dir/batch_results/batch_results.json
    candidates = [
        results_dir / "batch_results.json",
        results_dir / "batch_results" / "batch_results.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c, "r") as f:
                return json.load(f)
    raise FileNotFoundError(f"batch_results.json not found under {results_dir}")


def get_model_path_for_trial(
    subject: str,
    trial_dir: Path,
    model_root: Optional[Path],
    dataset_root: Path,
) -> Optional[Path]:
    """Find the .osim model for this specific subject/trial.

    Camargo stores models in subject/date subfolders, so we search in this order:
      1) If model_root is provided: any <model_root>/**/<subject>*.osim (subject-prefixed)
      2) Inside the trial directory and its ancestors: trial_dir, condition dir,
         date dir (if present), and subject dir (recursive search for *.osim)
      3) Fallback: any *.osim under dataset_root/<subject>/**
    """
    # 1) model_root with subject prefix
    if model_root is not None and model_root.exists():
        hits = list(model_root.rglob(f"{subject}*.osim"))
        if hits:
            return hits[0]
        # also allow any *.osim under subject folder name if present
        subj_dirs = [p for p in model_root.rglob(subject) if p.is_dir()]
        for sd in subj_dirs:
            cand = list(sd.rglob("*.osim"))
            if cand:
                return cand[0]

    # 2) trial dir -> parent (condition) -> parent (possibly date) -> subject
    search_roots: List[Path] = []
    cur = trial_dir
    # add trial_dir and its first two ancestors
    for _ in range(4):
        if cur is None or not cur.exists():
            break
        search_roots.append(cur)
        if cur == cur.parent:
            break
        cur = cur.parent

    # ensure subject root is included
    subj_root = dataset_root / subject
    if subj_root.exists():
        search_roots.append(subj_root)

    for root in search_roots:
        hits = list(root.rglob("*.osim"))
        if hits:
            # Prefer files with subject name in them
            subject_hits = [h for h in hits if subject in h.name]
            if subject_hits:
                return subject_hits[0]
            return hits[0]

    # 3) fallback: anywhere under subject dir
    if subj_root.exists():
        hits = list(subj_root.rglob("*.osim"))
        if hits:
            return hits[0]
    return None


def find_states_file(trial_dir: Path, states_glob: str) -> Optional[Path]:
    """Find an OpenSim states or motion file using recursive patterns.

    Args:
      states_glob: comma-separated patterns, supports '**' (e.g.,
        "*states.sto,*.sto,*kinematics.mot,*ik.mot,*inverse_kinematics.mot")
    """
    patterns = [p.strip() for p in states_glob.split(',') if p.strip()]
    search_roots = [trial_dir, trial_dir.parent, trial_dir.parent.parent if trial_dir.parent != trial_dir.parent.parent else None]
    search_roots = [p for p in search_roots if p and p.exists()]

    for root in search_roots:
        for pat in patterns:
            # Use rglob for recursive search
            hits = list(root.rglob(pat))
            if hits:
                # Prefer files under analysis or results folders, then any
                preferred = [h for h in hits if any(k in h.as_posix().lower() for k in ["analysis", "results", "ik", "states"])]
                return preferred[0] if preferred else hits[0]
    return None


def add_sensor_frame(osim, model, parent_frame, name: str, R_sensor_in_parent: np.ndarray):
    """Create a PhysicalOffsetFrame attached to parent_frame with the given rotation."""
    # Convert numpy rotation (3x3) to SimTK Rotation and build a Transform
    R = osim.Rotation(osim.Mat33(*R_sensor_in_parent.flatten(order='F')))
    t = osim.Vec3(0.0, 0.0, 0.0)  # at body origin
    X_SP = osim.Transform(R, t)
    # In OpenSim 4.x, ctor signature is (name, parent_frame, Transform)
    sensor_frame = osim.PhysicalOffsetFrame(name, parent_frame, X_SP)
    model.addComponent(sensor_frame)
    return sensor_frame


def compute_gyro_from_states(osim, model, sensor_frame, parent_body, states_storage_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Compute angular velocity of the parent body expressed in the sensor frame.
    
    Simplified approach: Just read the body's angular velocity in its own frame,
    then apply the sensor rotation to express it in sensor coordinates.

    Args:
        osim: OpenSim module
        model: OpenSim Model
        sensor_frame: PhysicalOffsetFrame representing the IMU
        parent_body: The body the sensor is attached to
        states_storage_path: Path to states file

    Returns
      times: shape (N,)
      gyro:  shape (N, 3) [rad/s]
    """
    # Load states trajectory - just read the table directly
    try:
        table = osim.TimeSeriesTable(str(states_storage_path))
    except Exception:
        storage = osim.Storage(str(states_storage_path))
        table = osim.TableUtilities().convertStorageToTable(storage)

    # Get time and data dimensions
    n = table.getNumRows()
    print(f"        🔍 Table has {n} time points")
    times = np.zeros(n, dtype=float)
    gyro = np.zeros((n, 3), dtype=float)
    
    # Get column labels
    labels = list(table.getColumnLabels())
    
    # Find coordinate columns (simple names, no paths)
    coord_to_col = {}
    speed_to_col = {}
    for label in labels:
        if label == 'time':
            continue
        if label.endswith('_u'):
            coord_name = label[:-2]
            speed_to_col[coord_name] = label
        elif not ('moment' in label.lower() or 'force' in label.lower()):
            coord_to_col[label] = label
    
    print(f"        🔍 Found {len(coord_to_col)} coordinates, {len(speed_to_col)} speeds")
    
    # Get the sensor's rotation matrix relative to the parent body
    # This is fixed and defined when we created the sensor frame
    state = model.initSystem()
    sensor_R_in_body = sensor_frame.getOffsetTransform().R().asMat33()
    R_sensor_body = np.array([
        [sensor_R_in_body.get(0,0), sensor_R_in_body.get(0,1), sensor_R_in_body.get(0,2)],
        [sensor_R_in_body.get(1,0), sensor_R_in_body.get(1,1), sensor_R_in_body.get(1,2)],
        [sensor_R_in_body.get(2,0), sensor_R_in_body.get(2,1), sensor_R_in_body.get(2,2)]
    ])
    
    print(f"        🔍 Sensor rotation in body frame:\n{R_sensor_body}")
    
    # For each time point, set coordinates/speeds and get body angular velocity
    for i in range(n):
        times[i] = table.getIndependentColumn()[i]
        
        # Set all coordinate values and speeds
        for coord_name, col_label in coord_to_col.items():
            try:
                q = table.getDependentColumn(col_label)[i]
                coord = model.getCoordinateSet().get(coord_name)
                coord.setValue(state, q, False)
            except:
                pass
        
        for coord_name, speed_col in speed_to_col.items():
            try:
                u = table.getDependentColumn(speed_col)[i]
                coord = model.getCoordinateSet().get(coord_name)
                coord.setSpeedValue(state, u)
            except:
                pass
        
        # Realize to get kinematics
        model.assemble(state)
        model.realizeVelocity(state)
        
        # Get body's angular velocity in ground frame
        w_ground = parent_body.getAngularVelocityInGround(state)
        
        # Get body's orientation in ground
        R_body_ground = parent_body.getTransformInGround(state).R().asMat33()
        R_GB = np.array([
            [R_body_ground.get(0,0), R_body_ground.get(0,1), R_body_ground.get(0,2)],
            [R_body_ground.get(1,0), R_body_ground.get(1,1), R_body_ground.get(1,2)],
            [R_body_ground.get(2,0), R_body_ground.get(2,1), R_body_ground.get(2,2)]
        ])
        
        # Angular velocity in body frame: w_body = R_BG @ w_ground
        R_BG = R_GB.T  # Transpose to get body-to-ground inverse
        w_ground_vec = np.array([w_ground.get(0), w_ground.get(1), w_ground.get(2)])
        w_body = R_BG @ w_ground_vec
        
        # Angular velocity in sensor frame: w_sensor = R_sensor_body @ w_body
        w_sensor = R_sensor_body @ w_body
        
        if i == 0:
            print(f"        🔍 Sample 0: w_ground={w_ground_vec}, w_body={w_body}, w_sensor={w_sensor}")
        
        gyro[i, :] = w_sensor
    
    print(f"        ✓ Computed gyro data: shape={gyro.shape}, range=[{gyro.min():.3f}, {gyro.max():.3f}]")
    return times, gyro


def write_imu_csv(output_file: Path, time: np.ndarray, segment_to_gyro: Dict[str, np.ndarray]):
    import pandas as pd
    data = {
        'time': time,
    }
    # Concatenate segment gyros as columns: <segment>_gyro_{x,y,z}
    for seg, g in segment_to_gyro.items():
        data[f"{seg}_gyro_x"] = g[:, 0]
        data[f"{seg}_gyro_y"] = g[:, 1]
        data[f"{seg}_gyro_z"] = g[:, 2]
    df = pd.DataFrame(data)
    print(f"    📊 Writing CSV with {len(df)} rows, {len(df.columns)} columns: {list(df.columns)}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    # Verify file was written
    if output_file.exists():
        file_size = output_file.stat().st_size
        print(f"    ✓ File written: {output_file} ({file_size} bytes)")
    else:
        print(f"    ❌ File was not created: {output_file}")


def copy_trial_structure(source_trial_dir: Path, output_trial_dir: Path, dry_run: bool = False):
    """Copy the trial folder structure (excluding Input/imu_data.csv which will be generated)."""
    import shutil
    for item in source_trial_dir.iterdir():
        if item.is_file():
            # Copy file
            if not dry_run:
                output_trial_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, output_trial_dir / item.name)
        elif item.is_dir():
            # Copy directory
            output_item_dir = output_trial_dir / item.name
            if item.name == "Input":
                # Create Input directory but don't copy imu_data.csv (we'll generate it)
                if not dry_run:
                    output_item_dir.mkdir(parents=True, exist_ok=True)
                    for sub_item in item.iterdir():
                        if sub_item.name != "imu_data.csv":
                            if sub_item.is_file():
                                shutil.copy2(sub_item, output_item_dir / sub_item.name)
                            elif sub_item.is_dir():
                                shutil.copytree(sub_item, output_item_dir / sub_item.name, dirs_exist_ok=True)
            else:
                # Copy other directories as-is
                if not dry_run:
                    shutil.copytree(item, output_item_dir, dirs_exist_ok=True)


def copy_osim_files(dataset_root: Path, output_root: Path, subject: str, dry_run: bool = False):
    """Copy .osim files to opensim folder for the subject."""
    import shutil
    try:
        # Create opensim directory for this subject
        subject_opensim_dir = output_root / subject / "opensim"
        if not dry_run:
            subject_opensim_dir.mkdir(parents=True, exist_ok=True)
        
        # Look for .osim files in the original dataset
        source_subject_dir = dataset_root / subject
        if not source_subject_dir.exists():
            return False
        
        # Search for .osim files in the subject directory and subdirectories
        osim_files = list(source_subject_dir.rglob("*.osim"))
        
        if not osim_files:
            return False
        
        # Copy the first .osim file found (assuming there's one main model file)
        source_osim = osim_files[0]
        target_osim = subject_opensim_dir / f"{subject}.osim"
        
        if not dry_run:
            shutil.copy2(source_osim, target_osim)
        
        return True
        
    except Exception as e:
        print(f"    ⚠️  Error copying .osim files: {e}")
        return False


def process_trial(
    osim,
    subject: str,
    condition: str,
    trial: str,
    dataset_root: Path,
    output_root: Path,
    batch_results: Dict,
    model_root: Optional[Path],
    states_glob: str,
    segments: List[str],
    dry_run: bool,
) -> bool:
    subject_dir = dataset_root / subject / condition / trial
    if not subject_dir.exists():
        print(f"    ⚠️  Missing trial directory: {subject_dir}")
        return False
    
    # Copy trial structure (all files except Input/imu_data.csv)
    output_trial_dir = output_root / subject / condition / trial
    copy_trial_structure(subject_dir, output_trial_dir, dry_run)

    # Locate states file
    states_file = find_states_file(subject_dir, states_glob)
    if states_file is None:
        print(f"    ⚠️  No states/motion file matching '{states_glob}' for {subject}/{condition}/{trial}")
        return False

    # Find model
    model_path = get_model_path_for_trial(subject, subject_dir, model_root, dataset_root)
    if model_path is None:
        print(f"    ⚠️  No .osim model found for subject {subject}")
        return False

    # Get per-trial rotation matrices from batch_results
    subj_data = batch_results.get('subjects', {}).get(subject, batch_results.get(subject))
    if subj_data is None or condition not in subj_data or trial not in subj_data[condition]:
        print(f"    ⚠️  No rotation matrices found in batch results for {subject}/{condition}/{trial}")
        return False
    trial_results = subj_data[condition][trial]

    # Load model
    model = osim.Model(str(model_path))

    # Attach sensor frames to requested segments
    seg_to_frame = {}
    seg_to_body = {}  # Keep track of parent bodies
    for seg in segments:
        seg_key = seg
        if seg_key not in trial_results.get('segments', {}):
            print(f"    ⚠️  Segment '{seg_key}' missing in batch results; skipping this segment")
            continue
        R_list = trial_results['segments'][seg_key].get('optimal_rotation_matrix')
        if R_list is None:
            print(f"    ⚠️  No rotation matrix for segment '{seg_key}'")
            continue
        R_np = np.array(R_list, dtype=float)
        if R_np.shape != (3, 3):
            print(f"    ⚠️  Rotation matrix for '{seg_key}' has wrong shape: {R_np.shape}")
            continue

        # Map segment key to body name in model
        # Assumes pelvis -> 'pelvis', femur_r -> 'femur_r', tibia_r -> 'tibia_r'
        body_name = seg_key
        try:
            parent_body = model.getBodySet().get(body_name)
        except Exception:
            print(f"    ⚠️  Body '{body_name}' not found in model; skipping segment '{seg_key}'")
            continue

        sensor_name = f"imu_{seg_key}"
        sensor_frame = add_sensor_frame(osim, model, parent_body, sensor_name, R_np)
        seg_to_frame[seg_key] = sensor_frame
        seg_to_body[seg_key] = parent_body

    if not seg_to_frame:
        print("    ⚠️  No valid segments to process for this trial")
        return False

    # Initialize system once after adding components
    model.finalizeConnections()
    model.initSystem()

    # Compute gyros for each segment
    print(f"    🔍 Computing gyro for {len(seg_to_frame)} segments: {list(seg_to_frame.keys())}")
    time_ref = None
    seg_to_gyro = {}
    for seg_key, sensor_frame in seg_to_frame.items():
        print(f"      → Processing segment '{seg_key}'...")
        parent_body = seg_to_body[seg_key]
        t, g = compute_gyro_from_states(osim, model, sensor_frame, parent_body, states_file)
        if time_ref is None:
            time_ref = t
        seg_to_gyro[seg_key] = g
        print(f"        ✓ Got {len(g)} samples for '{seg_key}'")
    
    # Align all gyro arrays to the same length (minimum length)
    if seg_to_gyro:
        min_len = min(len(g) for g in seg_to_gyro.values())
        print(f"    🔍 Aligning all segments to {min_len} samples")
        time_ref = time_ref[:min_len]
        for seg_key in seg_to_gyro:
            seg_to_gyro[seg_key] = seg_to_gyro[seg_key][:min_len]

    if time_ref is None or len(time_ref) == 0:
        print("    ⚠️  Failed to compute any gyro data")
        return False
    
    print(f"    ✓ Final data: {len(time_ref)} time points, {len(seg_to_gyro)} segments")

    # Write imu_data.csv under output_root mirror
    output_trial_dir = output_root / subject / condition / trial
    if dry_run:
        print(f"    🔍 DRY RUN: Would write {output_trial_dir / 'Input' / 'imu_data.csv'}")
        return True

    write_imu_csv(output_trial_dir / "Input" / "imu_data.csv", time_ref, seg_to_gyro)
    print(f"    💾 Wrote synthetic IMU: {output_trial_dir / 'Input' / 'imu_data.csv'}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create synthetic Camargo IMU data using OpenSim and optimized IMU orientations.")
    parser.add_argument("--dataset-root", required=True, help="Path to Camargo dataset root")
    parser.add_argument("--output-root", required=True, help="Path to output Canonical dataset root")
    parser.add_argument("--results-dir", required=True, help="Path to camargo batch results directory (contains batch_results.json)")
    parser.add_argument("--model-root", help="Path to subject models root (optional; script will search if not provided)")
    parser.add_argument("--states-glob", default="*states.sto,*.sto,*kinematics.mot,*ik.mot,*inverse_kinematics.mot",
                        help="Comma-separated recursive patterns to find states/motion (supports **). Examples: '*states.sto,*.sto,*kinematics.mot' ")
    parser.add_argument("--subjects", help="Comma-separated subject IDs to process (default: all found in results)")
    parser.add_argument("--conditions", help="Comma-separated conditions to process (default: all)")
    parser.add_argument("--segments", default="pelvis,femur_r,tibia_r", help="Comma-separated segments to synthesize (subset of batch results)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files, just print actions")
    parser.add_argument("--geometry-dir", help="Directory containing OpenSim Geometry (e.g., '/Applications/OpenSim 4.5/Geometry')")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    results_dir = Path(args.results_dir)
    model_root = Path(args.model_root) if args.model_root else None
    states_glob = args.states_glob
    segments = [s.strip() for s in args.segments.split(',') if s.strip()]

    if not dataset_root.exists():
        print(f"❌ Dataset root not found: {dataset_root}")
        sys.exit(1)
    if not results_dir.exists():
        print(f"❌ Results dir not found: {results_dir}")
        sys.exit(1)

    batch_results = load_batch_results(results_dir)

    # Configure geometry search path if provided
    if args.geometry_dir:
        geo = Path(args.geometry_dir)
        if geo.exists():
            os.environ.setdefault('OPENSIM_HOME', str(geo.parent))
            os.environ['OPENSIM_GEOMETRY_PATH'] = str(geo)
            try:
                # OpenSim 4.4+ utility
                if hasattr(osim, 'ModelVisualizer') and hasattr(osim.ModelVisualizer, 'addDirToGeometrySearchPaths'):
                    osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geo))
            except Exception:
                pass

    # Determine subject list
    if args.subjects:
        subject_list = [s.strip() for s in args.subjects.split(',') if s.strip()]
    else:
        subject_list = list(batch_results.get('subjects', batch_results).keys())

    # Quick inventory to help users pick filters correctly
    all_subjects = list(batch_results.get('subjects', batch_results).keys())
    if not all_subjects:
        print("⚠️  No subjects found in batch_results.json. Check --results-dir path.")
    else:
        print(f"Found {len(all_subjects)} subjects in batch results: {all_subjects[:10]}{' ...' if len(all_subjects)>10 else ''}")

    # Determine conditions per subject
    cond_filter = set([c.strip() for c in args.conditions.split(',') if c.strip()]) if args.conditions else None

    # OpenSim import
    osim = try_import_opensim()

    total = 0
    ok = 0
    copied_osim_subjects = set()  # Track which subjects we've already copied osim files for
    
    for subject in subject_list:
        subj_data = batch_results.get('subjects', {}).get(subject, batch_results.get(subject))
        if subj_data is None:
            print(f"⚠️  No entries for subject {subject} in batch results; skipping")
            continue
        
        # Copy .osim files for this subject (only once per subject)
        if subject not in copied_osim_subjects:
            copy_osim_files(dataset_root, output_root, subject, args.dry_run)
            copied_osim_subjects.add(subject)
        
        for condition, trials_dict in subj_data.items():
            if cond_filter and condition not in cond_filter:
                continue
            for trial in trials_dict.keys():
                total += 1
                print(f"Processing {subject}/{condition}/{trial} ...")
                if process_trial(
                    osim=osim,
                    subject=subject,
                    condition=condition,
                    trial=trial,
                    dataset_root=dataset_root,
                    output_root=output_root,
                    batch_results=batch_results,
                    model_root=model_root,
                    states_glob=states_glob,
                    segments=segments,
                    dry_run=args.dry_run,
                ):
                    ok += 1

    print(f"\nDone. Trials attempted: {total}, Successful: {ok}")
    if total == 0:
        # Show helpful hint of available conditions per first few subjects
        try:
            preview = []
            for s_idx, (sname, sdata) in enumerate((batch_results.get('subjects', batch_results)).items()):
                if s_idx >= 5:
                    break
                preview.append({sname: list(sdata.keys())})
            if preview:
                print("Hint: Available conditions by subject (preview):")
                for row in preview:
                    print("  ", row)
            if args.conditions:
                print("Check that --conditions exactly match the keys above (Linux is case-sensitive).")
        except Exception:
            pass


if __name__ == "__main__":
    main()


