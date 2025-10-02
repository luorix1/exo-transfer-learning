#!/usr/bin/env python3
"""
Testing script for TCN-based joint moment prediction from IMU data.
"""

import os
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
                    label_mean: np.ndarray, label_std: np.ndarray, window_size: int, device: torch.device):
    """Make predictions on a single trial."""
    # Load IMU data
    imu_path = os.path.join(trial_path, 'Input', 'imu_data.csv')
    label_path = os.path.join(trial_path, 'Label', 'joint_moment.csv')
    
    if not os.path.exists(imu_path) or not os.path.exists(label_path):
        print(f"Missing data files in {trial_path}")
        return None, None, None
    
    # Load data
    imu_df = pd.read_csv(imu_path)
    label_df = pd.read_csv(label_path)
    
    # Extract gyroscope data
    gyro_cols = [col for col in imu_df.columns if 'gyro' in col.lower()]
    if len(gyro_cols) < 6:
        print(f"Insufficient gyro channels in {trial_path}")
        return None, None, None
    
    # Get pelvis and thigh gyro data
    pelvis_gyro = [col for col in gyro_cols if 'pelvis' in col.lower()]
    thigh_r_gyro = [col for col in gyro_cols if 'thigh_r' in col.lower() or 'femur_r' in col.lower()]
    thigh_l_gyro = [col for col in gyro_cols if 'thigh_l' in col.lower() or 'femur_l' in col.lower()]
    
    if not (pelvis_gyro and thigh_r_gyro and thigh_l_gyro):
        print(f"Missing required gyro channels in {trial_path}")
        return None, None, None
    
    # Combine pelvis and thigh gyro data
    input_cols = pelvis_gyro + thigh_r_gyro
    input_data = imu_df[input_cols].values
    
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
    
    # Get corresponding true labels
    hip_moment_cols = [col for col in label_df.columns if 'hip' in col.lower() and 'moment' in col.lower()]
    if len(hip_moment_cols) >= 2:
        hip_r_col = [col for col in hip_moment_cols if 'r' in col.lower() or 'right' in col.lower()]
        hip_l_col = [col for col in hip_moment_cols if 'l' in col.lower() or 'left' in col.lower()]
        
        if hip_r_col and hip_l_col:
            true_data = label_df[hip_r_col + hip_l_col].values
            # Get labels corresponding to the last time point of each window
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
                   conditions: list, window_size: int, device: torch.device):
    """Evaluate model on test subjects."""
    
    # Load normalization parameters
    input_mean, input_std, label_mean, label_std = load_normalization_params(save_dir)
    
    # Load model configuration (you might want to save this during training)
    config = {
        'input_size': 6,
        'output_size': 2,
        'num_channels': [64, 64, 32, 32],
        'kernel_size': 2,
        'number_of_layers': 2,
        'dropout': 0.2,
        'dilations': [1, 2, 4, 8],
        'window_size': window_size,
    }
    
    # Load model
    model = load_model(model_path, config, device)
    
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
                    label_mean, label_std, window_size, device
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
    
    # Plot predictions vs ground truth
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, joint in enumerate(['Right Hip', 'Left Hip']):
        axes[i].scatter(all_true_labels[:, i], all_predictions[:, i], alpha=0.5)
        axes[i].plot([all_true_labels[:, i].min(), all_true_labels[:, i].max()], 
                    [all_true_labels[:, i].min(), all_true_labels[:, i].max()], 'r--')
        axes[i].set_xlabel(f'True {joint} Moment (N-m/kg)')
        axes[i].set_ylabel(f'Predicted {joint} Moment (N-m/kg)')
        axes[i].set_title(f'{joint} Moment Prediction')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'trial': trial_names * len(all_predictions) if len(trial_names) == 1 else [name for name in trial_names for _ in range(len(all_predictions) // len(trial_names))],
        'true_right_hip': all_true_labels[:, 0],
        'pred_right_hip': all_predictions[:, 0],
        'true_left_hip': all_true_labels[:, 1],
        'pred_left_hip': all_predictions[:, 1],
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
    parser.add_argument('--window_size', type=int, default=100,
                       help='Window size for temporal sequences')
    
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
        device=device
    )


if __name__ == '__main__':
    main()
