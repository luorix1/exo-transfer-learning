#!/usr/bin/env python3
"""
Testing script for TCN-based joint moment prediction from IMU data.
"""

import os
os.environ["MKL_VERBOSE"] = "0"
os.environ["MKL_DISABLE_FAST_MM"] = "1"
import json
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
    input_mean = np.load(os.path.join(save_dir, 'input_mean.npy'))
    input_std = np.load(os.path.join(save_dir, 'input_std.npy'))
    label_mean = np.load(os.path.join(save_dir, 'label_mean.npy'))
    label_std = np.load(os.path.join(save_dir, 'label_std.npy'))
    return input_mean, input_std, label_mean, label_std


def predict_on_trial(model, trial_path: str, input_mean: np.ndarray, input_std: np.ndarray,
                    label_mean: np.ndarray, label_std: np.ndarray, window_size: int, device: torch.device,
                    imu_segments: list):
    """Make predictions on a single trial."""
    # Load IMU data
    imu_path = os.path.join(trial_path, 'Input', 'imu_data.csv')
    label_path = os.path.join(trial_path, 'Label', 'joint_moment.csv')
    
    if not os.path.exists(imu_path) or not os.path.exists(label_path):
        print(f"Missing data files in {trial_path}")
        return None, None, None
    
    # Load data
    try:
        imu_df = pd.read_csv(imu_path, sep=None, engine='python', on_bad_lines='skip')
        label_df = pd.read_csv(label_path, sep=None, engine='python', on_bad_lines='skip')
    except:
        imu_df = pd.read_csv(imu_path, sep=',', on_bad_lines='skip')
        label_df = pd.read_csv(label_path, sep=',', on_bad_lines='skip')
    
    # Extract gyroscope data based on configured IMU segments
    gyro_cols = [col for col in imu_df.columns if 'gyro' in col.lower()]
    
    # Configure based on imu_segments parameter
    if len(imu_segments) == 1 and imu_segments[0].lower() in ['femur', 'thigh']:
        # Single femur/thigh IMU mode (3 channels)
        thigh_r_gyro = [col for col in gyro_cols if 'thigh_r' in col.lower() or 'femur_r' in col.lower()]
        
        if not thigh_r_gyro or len(thigh_r_gyro) < 3:
            print(f"Required IMU segment 'femur/thigh' not found in {trial_path}")
            print(f"  Available gyro columns: {gyro_cols}")
            return None, None, None
        
        input_cols = thigh_r_gyro[:3]
        input_data = imu_df[input_cols].values
    
    elif len(imu_segments) == 2:
        # Dual IMU mode
        seg1 = imu_segments[0].lower()
        seg2 = imu_segments[1].lower()
        
        pelvis_gyro = [col for col in gyro_cols if 'pelvis' in col.lower()]
        thigh_r_gyro = [col for col in gyro_cols if 'thigh_r' in col.lower() or 'femur_r' in col.lower()]
        
        if 'pelvis' in [seg1, seg2] and ('femur' in [seg1, seg2] or 'thigh' in [seg1, seg2]):
            if not pelvis_gyro or len(pelvis_gyro) < 3:
                print(f"Required IMU segment 'pelvis' not found in {trial_path}")
                print(f"  Available gyro columns: {gyro_cols}")
                return None, None, None
            if not thigh_r_gyro or len(thigh_r_gyro) < 3:
                print(f"Required IMU segment 'femur/thigh' not found in {trial_path}")
                print(f"  Available gyro columns: {gyro_cols}")
                return None, None, None
            
            input_cols = pelvis_gyro[:3] + thigh_r_gyro[:3]
            input_data = imu_df[input_cols].values
        else:
            print(f"Unsupported IMU segment configuration: {imu_segments}")
            return None, None, None
    else:
        print(f"Invalid number of IMU segments: {len(imu_segments)}")
        return None, None, None
    
    # Normalize input data
    input_data = (input_data - input_mean) / input_std
    
    # Create sliding windows
    predictions = []
    true_labels = []
    
    for i in range(len(input_data) - window_size + 1):
        window = input_data[i:i + window_size]
        window_tensor = torch.FloatTensor(window.T).unsqueeze(0).to(device)  # Add batch dimension
        
        with torch.no_grad():
            pred = model(window_tensor)
            predictions.append(pred.cpu().numpy())
    
    # Get corresponding true labels (unilateral: use right hip flexion moment)
    # Look for hip_flexion_r_moment first
    hip_flexion_r_col = [col for col in label_df.columns 
                        if 'hip_flexion_r_moment' in col.lower()]
    
    if hip_flexion_r_col:
        true_data = label_df[hip_flexion_r_col[0]].values.reshape(-1, 1)
        # Get labels corresponding to the last time point of each window
        for i in range(len(input_data) - window_size + 1):
            true_labels.append(true_data[i + window_size - 1])
    else:
        # Fallback: try generic hip moment columns
        hip_moment_cols = [col for col in label_df.columns 
                          if 'hip' in col.lower() and 'moment' in col.lower()]
        if hip_moment_cols:
            # Use right hip moment for unilateral model
            hip_r_col = [col for col in hip_moment_cols 
                        if 'r' in col.lower() or 'right' in col.lower()]
            
            if hip_r_col:
                true_data = label_df[hip_r_col[0]].values.reshape(-1, 1)
                for i in range(len(input_data) - window_size + 1):
                    true_labels.append(true_data[i + window_size - 1])
            elif len(hip_moment_cols) == 1:
                # Fallback: use single moment column
                true_data = label_df[hip_moment_cols[0]].values.reshape(-1, 1)
                for i in range(len(input_data) - window_size + 1):
                    true_labels.append(true_data[i + window_size - 1])
    
    if not predictions or not true_labels:
        return None, None, None
    
    predictions = np.array(predictions).squeeze()
    true_labels = np.array(true_labels)
    
    # Denormalize predictions
    predictions_denorm = predictions * label_std + label_mean
    true_labels_denorm = true_labels
    
    return predictions_denorm, true_labels_denorm, input_data


def evaluate_model(model_path: str, data_root: str, save_dir: str, subjects: list, 
                   conditions: list, window_size: int, device: torch.device, imu_segments: list):
    """Evaluate model on test subjects."""
    
    # Load normalization parameters
    input_mean, input_std, label_mean, label_std = load_normalization_params(save_dir)
    
    # Try to load saved config, otherwise use default with provided parameters
    config_path = os.path.join(save_dir, 'config.json')
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Override window_size if provided
        if window_size is not None and window_size != config.get('window_size'):
            print(f"Overriding window_size: {config.get('window_size')} -> {window_size}")
            config['window_size'] = window_size
    else:
        print(f"Config file not found at {config_path}, using default config with provided parameters")
        
        # Use provided imu_segments or default
        if imu_segments is None:
            imu_segments = ['pelvis', 'femur']
            print(f"Using default imu_segments: {imu_segments}")
        
        # Auto-adjust input_size based on IMU segments
        if len(imu_segments) == 1 and imu_segments[0].lower() in ['femur', 'thigh']:
            input_size = 3  # Single IMU: 3 gyro channels
        else:
            input_size = 6  # Dual IMU: 6 gyro channels
        
        # Use provided window_size or default
        if window_size is None:
            window_size = 100
            print(f"Using default window_size: {window_size}")
        
        # Load model configuration from default config file
        config = DEFAULT_TCN_CONFIG.copy()
        config['input_size'] = input_size
        config['output_size'] = 1  # Unilateral training: single hip moment output
        config['window_size'] = window_size
    
    # Load model
    model = load_model(model_path, config, device)
    
    # Get imu_segments from config or use provided value
    config_imu_segments = config.get('imu_segments', imu_segments if imu_segments else ['pelvis', 'femur'])
    
    all_predictions = []
    all_true_labels = []
    trial_names = []
    
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
                    model, trial_path, input_mean, input_std, 
                    label_mean, label_std, config['window_size'], device, config_imu_segments
                )
                
                if pred is not None:
                    all_predictions.append(pred)
                    all_true_labels.append(true)
                    trial_names.append(f"{subject}/{condition}/{trial}")
    
    if not all_predictions:
        print("No valid predictions made")
        return
    
    # Concatenate all predictions and labels
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    
    # Calculate metrics
    mse = np.mean((all_predictions - all_true_labels) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_predictions - all_true_labels))
    
    print(f"\nEvaluation Results:")
    print(f"RMSE: {rmse:.4f} N-m/kg")
    print(f"MAE: {mae:.4f} N-m/kg")
    print(f"Number of samples: {len(all_predictions)}")
    
    # Plot predictions vs ground truth (unilateral: single hip moment)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    # Flatten arrays for plotting
    true_flat = all_true_labels.flatten()
    pred_flat = all_predictions.flatten()
    
    ax.scatter(true_flat, pred_flat, alpha=0.5)
    ax.plot([true_flat.min(), true_flat.max()], 
            [true_flat.min(), true_flat.max()], 'r--', label='Perfect Prediction')
    ax.set_xlabel('True Hip Moment (N-m/kg)')
    ax.set_ylabel('Predicted Hip Moment (N-m/kg)')
    ax.set_title('Hip Moment Prediction (Unilateral Model)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'true_hip_moment': true_flat,
        'pred_hip_moment': pred_flat,
    })
    
    results_df.to_csv(os.path.join(save_dir, 'evaluation_results.csv'), index=False)
    print(f"Results saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Test TCN model for joint moment prediction')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, 
                       default='/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Canonical',
                       help='Path to Canonical dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory containing normalization parameters')
    parser.add_argument('--test_subjects', nargs='+', 
                       default=['BT11', 'BT12', 'BT13', 'BT14', 'BT15'],
                       help='Test subjects')
    parser.add_argument('--conditions', nargs='+', default=['levelground'],
                       help='Conditions to test on')
    parser.add_argument('--imu_segments', nargs='+', default=None,
                       help='IMU segments to use (optional, will use config.json if available): ["femur"] for single thigh IMU (3 channels), ["pelvis", "femur"] for dual (6 channels)')
    parser.add_argument('--window_size', type=int, default=None,
                       help='Window size for temporal sequences (optional, will use config.json if available)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model and normalization files exist
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return
    
    norm_files = ['input_mean.npy', 'input_std.npy', 'label_mean.npy', 'label_std.npy']
    for file in norm_files:
        if not os.path.exists(os.path.join(args.save_dir, file)):
            print(f"Normalization file not found: {os.path.join(args.save_dir, file)}")
            return
    
    # Evaluate model
    evaluate_model(
        model_path=args.model_path,
        data_root=args.data_root,
        save_dir=args.save_dir,
        subjects=args.test_subjects,
        conditions=args.conditions,
        window_size=args.window_size,
        device=device,
        imu_segments=args.imu_segments
    )


if __name__ == '__main__':
    main()
