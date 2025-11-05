#!/usr/bin/env python3
"""
Testing script for TCN-based joint moment prediction with heuristic coordinate frame transformation.

This script applies a pre-defined relative transform (rotation matrix) to convert inputs
from a different dataset's coordinate frame to match the training dataset's frame.
This is useful for testing models on datasets with different IMU mounting orientations.
"""

import os
import warnings
import logging
import sys
import gc

os.environ["MKL_VERBOSE"] = "0"
os.environ["MKL_DISABLE_FAST_MM"] = "1"
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "0"
os.environ["TORCH_BACKEND_DISABLE_NNPACK"] = "1"

import json
import torch

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.spatial.transform import Rotation as R
import wandb

from model.tcn import TCNModel
from data.dataloader import DataHandler
from trainer import Trainer
from loss import JointMomentLoss
from config.hyperparameters import DEFAULT_TCN_CONFIG


def load_model(model_path: str, config: dict, device: torch.device):
    """Load a trained model."""
    model = TCNModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_normalization_params(save_dir: str):
    """Load normalization parameters."""
    input_mean = np.load(os.path.join(save_dir, "input_mean.npy"))
    input_std = np.load(os.path.join(save_dir, "input_std.npy"))
    label_mean = np.load(os.path.join(save_dir, "label_mean.npy"))
    label_std = np.load(os.path.join(save_dir, "label_std.npy"))
    return input_mean, input_std, label_mean, label_std


def load_transform_matrix(transform_path: str = None, euler_angles: list = None, rotation_axis: str = 'z', rotation_angle: float = None) -> np.ndarray:
    """
    Load or create a rotation matrix for coordinate frame transformation.
    
    IMPORTANT: The returned matrix should transform FROM test dataset's IMU frame 
    TO training dataset's IMU frame. For example:
    - If testing on MetaMobility but model trained on Camargo
    - The matrix should be R_Cam_from_Meta (MetaMobility → Camargo)
    - This ensures test data matches the coordinate frame the model expects
    
    Args:
        transform_path: Path to .npy file containing a 3x3 rotation matrix
                       (should be R_test_to_train, e.g., memo_to_camargo.npy)
        euler_angles: List of [x, y, z] Euler angles in degrees (extrinsic rotations)
        rotation_axis: Single axis rotation ('x', 'y', or 'z')
        rotation_angle: Rotation angle in degrees (for single axis rotation)
    
    Returns:
        3x3 rotation matrix that transforms FROM test frame TO training frame
    """
    if transform_path:
        if os.path.exists(transform_path):
            transform = np.load(transform_path)
            if transform.shape != (3, 3):
                raise ValueError(f"Transform matrix must be 3x3, got shape {transform.shape}")
            print(f"Loaded transform matrix from {transform_path}")
            return transform
        else:
            raise FileNotFoundError(f"Transform matrix file not found: {transform_path}")
    
    elif euler_angles:
        # Create rotation from Euler angles (degrees)
        euler_rad = np.deg2rad(euler_angles)
        r = R.from_euler('xyz', euler_rad, degrees=False)
        transform = r.as_matrix()
        print(f"Created transform matrix from Euler angles: {euler_angles} degrees")
        return transform
    
    elif rotation_axis and rotation_angle is not None:
        # Create rotation around single axis
        angle_rad = np.deg2rad(rotation_angle)
        if rotation_axis.lower() == 'x':
            r = R.from_euler('x', angle_rad, degrees=False)
        elif rotation_axis.lower() == 'y':
            r = R.from_euler('y', angle_rad, degrees=False)
        elif rotation_axis.lower() == 'z':
            r = R.from_euler('z', angle_rad, degrees=False)
        else:
            raise ValueError(f"Invalid rotation axis: {rotation_axis}. Must be 'x', 'y', or 'z'")
        transform = r.as_matrix()
        print(f"Created transform matrix: {rotation_angle} degrees around {rotation_axis} axis")
        return transform
    
    else:
        # Default: identity matrix (no transformation)
        print("No transform specified, using identity matrix (no transformation)")
        return np.eye(3)


def apply_transform_to_gyro_data(gyro_data: np.ndarray, transform_matrix: np.ndarray, imu_segments: list) -> np.ndarray:
    """
    Apply rotation matrix to gyroscope data.
    
    IMPORTANT: The transform_matrix should convert FROM the test dataset's IMU frame 
    TO the training dataset's IMU frame. For example:
    - If model trained on Camargo and testing on MetaMobility
    - transform_matrix should be R_Cam_from_Meta (MetaMobility → Camargo)
    - This transforms test data to match what the model expects
    
    Args:
        gyro_data: Input gyro data in test dataset's coordinate frame, shape (N, channels)
                   - For single IMU: (N, 3) - [femur_x, femur_y, femur_z]
                   - For dual IMU: (N, 6) - [pelvis_x, pelvis_y, pelvis_z, femur_x, femur_y, femur_z]
        transform_matrix: 3x3 rotation matrix that transforms FROM test frame TO training frame
                          (e.g., R_Cam_from_Meta for MetaMobility → Camargo)
        imu_segments: List of IMU segments ['femur'] or ['pelvis', 'femur']
    
    Returns:
        Transformed gyro data in training dataset's coordinate frame, same shape as input
    """
    if len(imu_segments) == 1:
        # Single IMU: apply transform to all 3 channels
        # gyro_data shape: (N, 3) - each row is a 3D vector in test frame
        # transform_matrix: R_test_to_train (e.g., R_Cam_from_Meta)
        # For row vectors: v_train = v_test @ R_test_to_train.T
        transformed = gyro_data @ transform_matrix.T
        return transformed
    elif len(imu_segments) == 2:
        # Dual IMU: apply transform to each 3-channel group separately
        # gyro_data shape: (N, 6)
        # First 3 channels: pelvis, last 3 channels: femur
        # transform_matrix: R_test_to_train (applied to both segments)
        pelvis_gyro = gyro_data[:, :3]
        femur_gyro = gyro_data[:, 3:6]
        
        # Transform each segment's gyro data from test frame to training frame
        transformed_pelvis = pelvis_gyro @ transform_matrix.T
        transformed_femur = femur_gyro @ transform_matrix.T
        
        transformed = np.hstack((transformed_pelvis, transformed_femur))
        return transformed
    else:
        raise ValueError(f"Unsupported number of IMU segments: {len(imu_segments)}")


def detect_dataset_type(data_root: str) -> str:
    """Detect dataset type based on data_root path to determine sampling rate."""
    if 'Camargo' in data_root:
        return 'camargo'  # Higher sampling rate, needs downsampling
    elif 'Keaton' in data_root:
        return 'keaton'  # Higher sampling rate, needs downsampling
    elif 'Molinaro' in data_root:
        return 'molinaro'  # Higher sampling rate, needs downsampling
    elif 'MetaMobility' in data_root:
        return 'memo'     # Standard sampling rate
    else:
        return 'unknown'  # Default to no special handling


def butter_lowpass_zero_phase(data: np.ndarray, cutoff_hz: float = 6.0, fs_hz: float = 100.0, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter to match dataloader preprocessing."""
    if data is None or data.size == 0:
        return data
    # Design Butterworth
    nyq = 0.5 * fs_hz
    wn = cutoff_hz / nyq
    b, a = signal.butter(order, wn, btype='low', analog=False)
    # filtfilt along time axis (axis=0). Expect shape (N, 1)
    try:
        return signal.filtfilt(b, a, data.squeeze(), axis=0, method='pad', padlen=min(3 * max(len(a), len(b)), max(0, len(data) - 1))).reshape(-1, 1)
    except ValueError:
        # If sequence too short for padlen, fall back to lfilter twice
        y = signal.lfilter(b, a, data.squeeze(), axis=0)
        y = signal.lfilter(b, a, y[::-1], axis=0)[::-1]
        return y.reshape(-1, 1)


def predict_on_trial(
    model,
    trial_path: str,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    label_mean: np.ndarray,
    label_std: np.ndarray,
    window_size: int,
    device: torch.device,
    imu_segments: list,
    transform_matrix: np.ndarray,
    label_filter_hz: float = 6.0,
    normalize: bool = True,
    dataset_type: str = 'unknown',
):
    """
    Make predictions on a single trial with coordinate frame transformation.
    
    This function is identical to test.py's predict_on_trial except it applies
    a transform_matrix to the input gyro data before normalization.
    """
    # Load IMU data
    input_file_dir = os.path.join(trial_path, "Input")
    label_file_dir = os.path.join(trial_path, "Label")
    
    if not os.path.exists(input_file_dir) or not os.path.exists(label_file_dir):
        print(f"Missing directories in {trial_path}")
        return None, None, None
    
    # Find CSV files (ignore images)
    input_csv_files = sorted([f for f in os.listdir(input_file_dir) if f.lower().endswith('.csv')])
    label_csv_files = sorted([f for f in os.listdir(label_file_dir) if f.lower().endswith('.csv')])
    
    if not input_csv_files or not label_csv_files:
        print(f"Missing CSV files in {trial_path}")
        return None, None, None
    
    # Prefer joint_moment CSV for labels
    label_file_name = next((f for f in label_csv_files if 'joint_moment' in f.lower()), label_csv_files[0])
    
    imu_path = os.path.join(input_file_dir, input_csv_files[0])  # Use first CSV file in Input
    label_path = os.path.join(label_file_dir, label_file_name)

    # Load data
    try:
        imu_df = pd.read_csv(imu_path, sep=None, engine="python", on_bad_lines="skip")
        label_df = pd.read_csv(label_path, sep=None, engine="python", on_bad_lines="skip")
    except:
        imu_df = pd.read_csv(imu_path, sep=",", on_bad_lines="skip")
        label_df = pd.read_csv(label_path, sep=",", on_bad_lines="skip")

    # Extract gyroscope data based on configured IMU segments (matching dataloader)
    gyro_cols = [col for col in imu_df.columns if "gyro" in col.lower()]

    # Configure based on imu_segments parameter
    if len(imu_segments) == 1 and imu_segments[0].lower() in ["femur", "thigh"]:
        # Single femur/thigh IMU mode (3 channels)
        thigh_r_gyro = [
            col
            for col in gyro_cols
            if "thigh_r" in col.lower() or "femur_r" in col.lower()
        ]
        thigh_l_gyro = [
            col
            for col in gyro_cols
            if "thigh_l" in col.lower() or "femur_l" in col.lower()
        ]

        if not thigh_r_gyro or len(thigh_r_gyro) < 3:
            print(f"Required IMU segment 'femur/thigh' not found in {trial_path}")
            print(f"  Available gyro columns: {gyro_cols}")
            return None, None, None

        # Process right side data
        input_data_r = imu_df[thigh_r_gyro[:3]].values
        
        # Process left side data if available
        input_data_l = None
        if thigh_l_gyro and len(thigh_l_gyro) >= 3:
            input_data_l = imu_df[thigh_l_gyro[:3]].values
        
        # Stack left and right data (matching dataloader logic)
        if dataset_type == 'memo':
            input_data = input_data_r
        else:
            if input_data_l is not None and len(input_data_l) > 0:
                if np.random.randint(0, 2):
                    input_data = np.vstack((input_data_r, input_data_l))
                else:
                    input_data = np.vstack((input_data_l, input_data_r))
            else:
                input_data = input_data_r

    elif len(imu_segments) == 2:
        # Dual IMU mode
        seg1 = imu_segments[0].lower()
        seg2 = imu_segments[1].lower()

        pelvis_gyro = [col for col in gyro_cols if "pelvis" in col.lower()]
        thigh_r_gyro = [
            col
            for col in gyro_cols
            if "thigh_r" in col.lower() or "femur_r" in col.lower()
        ]
        thigh_l_gyro = [
            col
            for col in gyro_cols
            if "thigh_l" in col.lower() or "femur_l" in col.lower()
        ]

        if "pelvis" in [seg1, seg2] and (
            "femur" in [seg1, seg2] or "thigh" in [seg1, seg2]
        ):
            if not pelvis_gyro or len(pelvis_gyro) < 3:
                print(f"Required IMU segment 'pelvis' not found in {trial_path}")
                print(f"  Available gyro columns: {gyro_cols}")
                return None, None, None
            if not thigh_r_gyro or len(thigh_r_gyro) < 3:
                print(f"Required IMU segment 'femur/thigh' not found in {trial_path}")
                print(f"  Available gyro columns: {gyro_cols}")
                return None, None, None

            # Process right side data (pelvis + thigh_r)
            input_data_r = imu_df[pelvis_gyro[:3] + thigh_r_gyro[:3]].values
            
            # Process left side data if available (pelvis + thigh_l)
            input_data_l = None
            if thigh_l_gyro and len(thigh_l_gyro) >= 3:
                input_data_l = imu_df[pelvis_gyro[:3] + thigh_l_gyro[:3]].values
            
            # Stack left and right data (matching dataloader logic)
            if dataset_type == 'memo':
                input_data = input_data_r
            else:
                if input_data_l is not None and len(input_data_l) > 0:
                    if np.random.randint(0, 2):
                        input_data = np.vstack((input_data_r, input_data_l))
                    else:
                        input_data = np.vstack((input_data_l, input_data_r))
                else:
                    input_data = input_data_r
        else:
            print(f"Unsupported IMU segment configuration: {imu_segments}")
            return None, None, None
    else:
        print(f"Invalid number of IMU segments: {len(imu_segments)}")
        return None, None, None

    # Apply coordinate frame transformation BEFORE downsampling and normalization
    input_data = apply_transform_to_gyro_data(input_data, transform_matrix, imu_segments)
    
    # Apply downsampling for high-rate datasets (matching dataloader logic)
    if dataset_type in ['camargo', 'keaton', 'molinaro']:
        print(f"Applying downsampling (::2) for {dataset_type} dataset...")
        original_input_size = input_data.shape[0]
        input_data = input_data[::2]
        print(f"Downsampled input: {original_input_size} -> {input_data.shape[0]} samples")
    
    # Normalize input data if requested
    if normalize:
        input_data = (input_data - input_mean) / input_std

    # Create sliding windows - process in batches for memory efficiency
    predictions = []
    true_labels = []
    
    batch_size = 128  # Process windows in batches
    num_windows = len(input_data) - window_size + 1
    
    for batch_start in range(0, num_windows, batch_size):
        batch_end = min(batch_start + batch_size, num_windows)
        
        # Create batch of windows
        batch_windows = []
        for i in range(batch_start, batch_end):
            window = input_data[i : i + window_size]
            batch_windows.append(window.T)
        
        # Stack and convert to tensor
        batch_tensor = torch.FloatTensor(np.stack(batch_windows)).to(device)
        
        with torch.no_grad():
            batch_preds = model(batch_tensor)
            predictions.append(batch_preds.cpu().numpy())
            
            # Explicitly delete tensors to free memory
            del batch_tensor, batch_preds, batch_windows
    
    # Concatenate all batch predictions
    if predictions:
        predictions = np.concatenate(predictions, axis=0)

    # Get corresponding true labels (matching dataloader: use both left and right)
    hip_flexion_r_col = [
        col for col in label_df.columns if "hip_flexion_r_moment" in col.lower()
    ]
    hip_flexion_l_col = [
        col for col in label_df.columns if "hip_flexion_l_moment" in col.lower()
    ]

    true_labels = []
    if hip_flexion_r_col or hip_flexion_l_col:
        # Process right side data
        true_data_r = None
        if hip_flexion_r_col:
            true_data_r = label_df[hip_flexion_r_col[0]].values.reshape(-1, 1)
            
            # Apply downsampling to labels if needed (matching dataloader)
            if dataset_type in ['camargo', 'keaton', 'molinaro']:
                true_data_r = true_data_r[::2]
            
            # Apply the same low-pass filter as used in training
            true_data_r = butter_lowpass_zero_phase(true_data_r, cutoff_hz=label_filter_hz)
        
        # Process left side data
        true_data_l = None
        if hip_flexion_l_col:
            true_data_l = label_df[hip_flexion_l_col[0]].values.reshape(-1, 1)
            
            # Apply downsampling to labels if needed (matching dataloader)
            if dataset_type in ['camargo', 'keaton', 'molinaro']:
                true_data_l = true_data_l[::2]
            elif dataset_type == 'memo':
                # Apply sign flip for memo dataset (matching dataloader)
                true_data_l = -true_data_l
            
            # Apply the same low-pass filter as used in training
            true_data_l = butter_lowpass_zero_phase(true_data_l, cutoff_hz=label_filter_hz)
        
        # Stack left and right data (matching dataloader logic)
        if dataset_type == 'memo':
            if true_data_r is not None:
                true_data = true_data_r
            else:
                true_data = None
        else:
            if true_data_r is not None and true_data_l is not None:
                if np.random.randint(0, 2):
                    true_data = np.vstack((true_data_r, true_data_l))
                else:
                    true_data = np.vstack((true_data_l, true_data_r))
            elif true_data_r is not None:
                true_data = true_data_r
            elif true_data_l is not None:
                true_data = true_data_l
            else:
                true_data = None
        
        if true_data is not None:            
            # Make sure we don't go out of bounds
            for i in range(num_windows):
                label_idx = min(i + window_size - 1, len(true_data) - 1)
                true_labels.append(true_data[label_idx])
            true_labels = np.array(true_labels)
    else:
        # Fallback: try generic hip moment columns
        hip_moment_cols = [
            col
            for col in label_df.columns
            if "hip" in col.lower() and "moment" in col.lower()
        ]
        if hip_moment_cols:
            hip_r_col = [
                col
                for col in hip_moment_cols
                if "r" in col.lower() or "right" in col.lower()
            ]
            hip_l_col = [
                col
                for col in hip_moment_cols
                if "l" in col.lower() or "left" in col.lower()
            ]

            true_data_r = None
            if hip_r_col:
                true_data_r = label_df[hip_r_col[0]].values.reshape(-1, 1)
                if dataset_type in ['camargo', 'keaton', 'molinaro']:
                    true_data_r = true_data_r[::2]
                true_data_r = butter_lowpass_zero_phase(true_data_r, cutoff_hz=label_filter_hz)
            
            true_data_l = None
            if hip_l_col:
                true_data_l = label_df[hip_l_col[0]].values.reshape(-1, 1)
                if dataset_type in ['camargo', 'keaton', 'molinaro']:
                    true_data_l = true_data_l[::2]
                true_data_l = butter_lowpass_zero_phase(true_data_l, cutoff_hz=label_filter_hz)
            
            if dataset_type == 'memo':
                if true_data_r is not None:
                    true_data = true_data_r
                else:
                    true_data = None
            else:
                if true_data_r is not None and true_data_l is not None:
                    if np.random.randint(0, 2):
                        true_data = np.vstack((true_data_r, true_data_l))
                    else:
                        true_data = np.vstack((true_data_l, true_data_r))
                elif true_data_r is not None:
                    true_data = true_data_r
                elif true_data_l is not None:
                    true_data = true_data_l
                else:
                    true_data = None
            
            if true_data is not None:
                for i in range(num_windows):
                    label_idx = min(i + window_size - 1, len(true_data) - 1)
                    true_labels.append(true_data[label_idx])
                true_labels = np.array(true_labels)

    if len(predictions) == 0 or len(true_labels) == 0:
        return None, None, None

    # Denormalize predictions using training dataset stats (only if normalization was used)
    if normalize:
        predictions_denorm = predictions * label_std + label_mean
        true_labels_denorm = true_labels
    else:
        predictions_denorm = predictions
        true_labels_denorm = true_labels

    return predictions_denorm, true_labels_denorm, input_data


# The rest of the file (evaluate_model and main) is identical to test.py
# but with transform_matrix parameter added. I'll copy the evaluate_model and main functions:

def evaluate_model(
    model_path: str,
    data_root: str,
    save_dir: str,
    subjects: list,
    conditions: list,
    window_size: int,
    device: torch.device,
    imu_segments: list,
    transform_matrix: np.ndarray,
    label_filter_hz: float = 6.0,
    normalize: bool = True,
    args=None,
):
    """
    Evaluate model on test subjects with coordinate frame transformation.
    
    This function is identical to test.py's evaluate_model except it applies
    a transform_matrix to the input gyro data.
    """

    # Detect dataset type for informational purposes
    dataset_type = detect_dataset_type(data_root)
    print(f"Detected dataset type: {dataset_type}")

    # Initialize wandb (same pattern as train.py)
    if not args.no_wandb:
        if args.wandb_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb_run_name = f"tcn_evaluation_heuristic_{timestamp}"
        else:
            wandb_run_name = args.wandb_name
        
        experiment_config = {
            'model_path': model_path,
            'data_root': data_root,
            'subjects': subjects,
            'conditions': conditions,
            'window_size': window_size,
            'imu_segments': imu_segments,
            'label_filter_hz': label_filter_hz,
            'normalize': normalize,
            'dataset_type': dataset_type,
            'transform_matrix': transform_matrix.tolist(),
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            'pytorch_version': torch.__version__,
        }
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=experiment_config,
            tags=args.wandb_tags + ['evaluation', 'tcn', 'joint_moment', 'heuristic_transform'],
            notes=f"TCN model evaluation with heuristic transform. Test: {len(subjects)} subjects, Conditions: {conditions}"
        )
        wandb_run = wandb.run
        print(f"Wandb run initialized: {wandb_run.url}")
    else:
        wandb_run = None
        print("Wandb logging disabled")

    # Load normalization parameters
    input_mean, input_std, label_mean, label_std = load_normalization_params(save_dir)

    # Try to load saved config, otherwise use default with provided parameters
    config_path = os.path.join(save_dir, "config.json")
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        if window_size is not None and window_size != config.get("window_size"):
            print(f"Overriding window_size: {config.get('window_size')} -> {window_size}")
            config["window_size"] = window_size
    else:
        print(f"Config file not found at {config_path}, using default config with provided parameters")
        if imu_segments is None:
            imu_segments = ["pelvis", "femur"]
            print(f"Using default imu_segments: {imu_segments}")
        if len(imu_segments) == 1 and imu_segments[0].lower() in ["femur", "thigh"]:
            input_size = 3
        else:
            input_size = 6
        if window_size is None:
            window_size = 100
            print(f"Using default window_size: {window_size}")
        config = DEFAULT_TCN_CONFIG.copy()
        config["input_size"] = input_size
        config["output_size"] = 1
        config["window_size"] = window_size

    # Load model
    model = load_model(model_path, config, device)

    # Get imu_segments from config or use provided value
    config_imu_segments = config.get(
        "imu_segments", imu_segments if imu_segments else ["pelvis", "femur"]
    )
    
    # Get label_filter_hz and normalize from config
    config_label_filter_hz = config.get("label_filter_hz", 6.0)
    config_normalize = config.get("normalize", True)

    all_predictions = []
    all_true_labels = []
    trial_names = []
    trial_predictions = []
    trial_true_labels = []

    # Evaluate on each subject and condition
    for subject in subjects:
        subject_path = os.path.join(data_root, subject)
        if not os.path.exists(subject_path):
            print(f"Subject {subject} not found")
            continue

        for condition in conditions:
            condition_path = os.path.join(subject_path, condition)
            if not os.path.exists(condition_path):
                print(f"Condition {condition} not found for subject {subject}")
                continue

            for trial in os.listdir(condition_path):
                trial_path = os.path.join(condition_path, trial)
                if not os.path.isdir(trial_path):
                    continue

                print(f"Processing {subject}/{condition}/{trial}")

                pred, true, input_data = predict_on_trial(
                    model,
                    trial_path,
                    input_mean,
                    input_std,
                    label_mean,
                    label_std,
                    config["window_size"],
                    device,
                    config_imu_segments,
                    transform_matrix,
                    label_filter_hz,
                    normalize,
                    dataset_type,
                )

                if pred is not None:
                    all_predictions.append(pred)
                    all_true_labels.append(true)
                    trial_names.append(f"{subject}/{condition}/{trial}")
                    trial_predictions.append(pred)
                    trial_true_labels.append(true)
                
                gc.collect()

    if not all_predictions:
        print("No valid predictions made")
        return

    # Concatenate all predictions and labels
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    
    # Remove any NaN values before calculating metrics
    valid_mask = ~(np.isnan(all_predictions) | np.isnan(all_true_labels))
    valid_mask = valid_mask.flatten()
    
    all_predictions_clean = all_predictions[valid_mask]
    all_true_labels_clean = all_true_labels[valid_mask]
    
    print(f"Total samples: {len(all_predictions)}, Valid samples: {len(all_predictions_clean)}, NaN samples removed: {len(all_predictions) - len(all_predictions_clean)}")

    # Calculate metrics
    mse = np.mean((all_predictions_clean - all_true_labels_clean) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_predictions_clean - all_true_labels_clean))
    
    ss_res = np.sum((all_true_labels_clean - all_predictions_clean) ** 2)
    ss_tot = np.sum((all_true_labels_clean - np.mean(all_true_labels_clean)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    gt_mean_abs = np.mean(np.abs(all_true_labels_clean))
    pred_mean_abs = np.mean(np.abs(all_predictions_clean))
    scale_ratio = pred_mean_abs / gt_mean_abs if gt_mean_abs > 0 else 0

    print(f"\nEvaluation Results:")
    print(f"RMSE: {rmse:.4f} N-m/kg")
    print(f"MAE: {mae:.4f} N-m/kg")
    print(f"R² Score: {r2_score:.4f}")
    print(f"Valid samples: {len(all_predictions_clean)}")
    print(f"\nScale Analysis:")
    print(f"Ground Truth Mean |Value|: {gt_mean_abs:.4f} N-m/kg")
    print(f"Predicted Mean |Value|: {pred_mean_abs:.4f} N-m/kg")
    print(f"Scale Ratio (Pred/GT): {scale_ratio:.4f}")
    if abs(scale_ratio - 1.0) > 0.5:
        print(f"⚠️  WARNING: Large scale mismatch detected! Ratio should be ~1.0")
    else:
        print(f"✓ Scale appears reasonable")

    # Log metrics to wandb
    if wandb_run is not None:
        wandb.log({
            'evaluation/rmse': rmse,
            'evaluation/mae': mae,
            'evaluation/r2_score': r2_score,
            'evaluation/mse': mse,
            'evaluation/valid_samples': len(all_predictions_clean),
            'evaluation/total_samples': len(all_predictions),
            'evaluation/gt_mean_abs': gt_mean_abs,
            'evaluation/pred_mean_abs': pred_mean_abs,
            'evaluation/scale_ratio': scale_ratio,
        })

    # Offset and range diagnostics
    gt_mean = float(np.mean(all_true_labels_clean))
    pred_mean = float(np.mean(all_predictions_clean))
    offset = pred_mean - gt_mean
    gt_min, gt_max = float(np.min(all_true_labels_clean)), float(np.max(all_true_labels_clean))
    pred_min, pred_max = float(np.min(all_predictions_clean)), float(np.max(all_predictions_clean))
    gt_span = gt_max - gt_min
    pred_span = pred_max - pred_min

    try:
        a, b = np.polyfit(all_true_labels_clean.flatten(), all_predictions_clean.flatten(), 1)
    except Exception:
        a, b = np.nan, np.nan

    print("\nOffset/Range Diagnostics:")
    print(f"Mean (GT): {gt_mean:.4f}  |  Mean (Pred): {pred_mean:.4f}  |  Offset (Pred-GT): {offset:+.4f} N-m/kg")
    print(f"Range (GT): [{gt_min:.4f}, {gt_max:.4f}]  span={gt_span:.4f}")
    print(f"Range (Pred): [{pred_min:.4f}, {pred_max:.4f}]  span={pred_span:.4f}")
    print(f"Linear fit: Pred ≈ {a:.3f} * GT + {b:.3f}")

    # Plot 1: Scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    true_flat = all_true_labels_clean.flatten()
    pred_flat = all_predictions_clean.flatten()
    ax.scatter(true_flat, pred_flat, alpha=0.3, s=10, color='blue', edgecolors='none')
    min_val = min(true_flat.min(), pred_flat.min())
    max_val = max(true_flat.max(), pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y=x (Perfect Prediction)")
    ax.text(0.05, 0.95, f'R² = {r2_score:.4f}\nRMSE = {rmse:.4f} N-m/kg\nMAE = {mae:.4f} N-m/kg', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel("True Hip Moment (N-m/kg)", fontsize=12)
    ax.set_ylabel("Predicted Hip Moment (N-m/kg)", fontsize=12)
    ax.set_title("Hip Moment Prediction (Heuristic Transform)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    scatter_plot_path = os.path.join(save_dir, "evaluation_scatter_heuristic.png")
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches="tight")
    if wandb_run is not None:
        wandb.log({"evaluation/scatter_plot_heuristic": wandb.Image(scatter_plot_path)})
    plt.close()

    # Plot 2: Time series plots for sample trials
    num_sample_trials = min(10, len(trial_names))
    if num_sample_trials > 0:
        print(f"\nGenerating time series plots for {num_sample_trials} sample trials...")
        for i in range(num_sample_trials):
            if i >= len(trial_predictions):
                break
            pred_trial = trial_predictions[i]
            true_trial = trial_true_labels[i]
            trial_name = trial_names[i]
            print(f"  Processing trial {i + 1}/{num_sample_trials}: {trial_name}")

            fig, ax = plt.subplots(1, 1, figsize=(14, 6))
            time_axis = np.arange(len(pred_trial))
            ax.plot(time_axis, true_trial.flatten(), "b-", label="Ground Truth", linewidth=2, alpha=0.8)
            ax.plot(time_axis, pred_trial.flatten(), "r--", label="Prediction", linewidth=2, alpha=0.8)
            
            trial_valid_mask = ~(np.isnan(pred_trial) | np.isnan(true_trial))
            if np.any(trial_valid_mask):
                trial_pred_clean = pred_trial[trial_valid_mask]
                trial_true_clean = true_trial[trial_valid_mask]
                trial_rmse = np.sqrt(np.mean((trial_pred_clean - trial_true_clean) ** 2))
                trial_mae = np.mean(np.abs(trial_pred_clean - trial_true_clean))
                ss_res = np.sum((trial_true_clean - trial_pred_clean) ** 2)
                ss_tot = np.sum((trial_true_clean - np.mean(trial_true_clean)) ** 2)
                trial_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                ax.text(0.02, 0.98, f'Trial RMSE: {trial_rmse:.4f} N-m/kg\nTrial MAE: {trial_mae:.4f} N-m/kg\nTrial R²: {trial_r2:.4f}', 
                        transform=ax.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            ax.set_xlabel("Sample Index", fontsize=12)
            ax.set_ylabel("Hip Flexion Moment (N-m/kg)", fontsize=12)
            ax.set_title(f"Time Series (Heuristic Transform): {trial_name}", fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            safe_trial_name = trial_name.replace("/", "_").replace("\\", "_")
            timeseries_plot_path = os.path.join(save_dir, f"timeseries_heuristic_{safe_trial_name}.png")
            plt.savefig(timeseries_plot_path, dpi=300, bbox_inches="tight")
            if wandb_run is not None:
                wandb.log({f"evaluation/timeseries_heuristic_{safe_trial_name}": wandb.Image(timeseries_plot_path)})
            plt.close()
            print(f"  Saved time series plot: timeseries_heuristic_{safe_trial_name}.png")

    # Save detailed results
    print("Creating results DataFrame...")
    results_df = pd.DataFrame({
        "true_hip_moment": true_flat,
        "pred_hip_moment": pred_flat,
    })
    print("Saving results to CSV...")
    results_df.to_csv(os.path.join(save_dir, "evaluation_results_heuristic.csv"), index=False)
    print("CSV saved successfully.")
    print(f"\nResults saved to {save_dir}")
    print(f"  - evaluation_scatter_heuristic.png: Scatter plot of all predictions")
    print(f"  - timeseries_heuristic_*.png: Time series plots for {num_sample_trials} sample trial(s)")
    print(f"  - evaluation_results_heuristic.csv: Detailed prediction results")
    
    plt.close('all')
    gc.collect()
    
    if wandb_run is not None:
        wandb.finish()
    
    print("Evaluation completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Test TCN model for joint moment prediction with heuristic coordinate frame transform"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Canonical",
        help="Path to Canonical dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory containing normalization parameters",
    )
    parser.add_argument(
        "--test_subjects",
        nargs="+",
        default=["BT11", "BT12", "BT13", "BT14", "BT15"],
        help="Test subjects",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["levelground"], help="Conditions to test on"
    )
    parser.add_argument(
        "--imu_segments",
        nargs="+",
        default=None,
        help='IMU segments to use (optional, will use config.json if available): ["femur"] for single thigh IMU (3 channels), ["pelvis", "femur"] for dual (6 channels)',
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Window size for temporal sequences (optional, will use config.json if available)",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable input normalization (use raw IMU data)",
    )
    parser.add_argument(
        "--transform_matrix",
        type=str,
        default=None,
        help="Path to .npy file containing 3x3 rotation matrix for coordinate frame transformation",
    )
    parser.add_argument(
        "--euler_angles",
        nargs=3,
        type=float,
        default=None,
        help="Euler angles in degrees [x, y, z] for coordinate frame transformation (extrinsic rotations)",
    )
    parser.add_argument(
        "--rotation_axis",
        type=str,
        choices=['x', 'y', 'z'],
        default=None,
        help="Single axis rotation (use with --rotation_angle)",
    )
    parser.add_argument(
        "--rotation_angle",
        type=float,
        default=None,
        help="Rotation angle in degrees for single axis rotation (use with --rotation_axis)",
    )
    parser.add_argument(
        "--wandb_project", type=str, default='transfer-learning',
        help="Wandb project name for logging"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="Wandb entity name"
    )
    parser.add_argument(
        "--wandb_name", type=str, default=None,
        help="Wandb run name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--wandb_tags", nargs="+", default=[],
        help="Wandb tags for the experiment"
    )
    parser.add_argument(
        "--no_wandb", action='store_true',
        help="Disable wandb logging"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if model and normalization files exist
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return

    norm_files = ["input_mean.npy", "input_std.npy", "label_mean.npy", "label_std.npy"]
    for file in norm_files:
        if not os.path.exists(os.path.join(args.save_dir, file)):
            print(f"Normalization file not found: {os.path.join(args.save_dir, file)}")
            return

    # Load or create transform matrix
    transform_matrix = load_transform_matrix(
        transform_path=args.transform_matrix,
        euler_angles=args.euler_angles,
        rotation_axis=args.rotation_axis,
        rotation_angle=args.rotation_angle
    )
    
    print(f"\nTransform matrix:")
    print(transform_matrix)
    print(f"Determinant: {np.linalg.det(transform_matrix):.6f} (should be ~1.0 for rotation matrix)")

    # Load config to get label_filter_hz and normalize
    config_path = os.path.join(args.save_dir, "config.json")
    label_filter_hz = 6.0
    normalize = True
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            label_filter_hz = config.get("label_filter_hz", 6.0)
            normalize = config.get("normalize", True)
            print(f"Using label filter: {label_filter_hz} Hz")
            print(f"Using input normalization: {normalize}")
    
    if args.no_normalize:
        normalize = False
        print("Input normalization disabled by --no_normalize flag")

    # Evaluate model
    evaluate_model(
        model_path=args.model_path,
        data_root=args.data_root,
        save_dir=args.save_dir,
        subjects=args.test_subjects,
        conditions=args.conditions,
        window_size=args.window_size,
        device=device,
        imu_segments=args.imu_segments,
        transform_matrix=transform_matrix,
        label_filter_hz=label_filter_hz,
        normalize=normalize,
        args=args,
    )
    
    plt.close('all')
    gc.collect()
    print("Script completed successfully!")

    sys.exit(0)

if __name__ == "__main__":
    main()

