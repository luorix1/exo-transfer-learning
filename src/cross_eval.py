#!/usr/bin/env python3
"""
Cross-dataset evaluation for TCN-based joint moment prediction.

This script evaluates a trained TCN model across ALL subjects in a target dataset
for specified conditions, computes per-subject metrics and overall RMSE/R² per condition,
and generates trend plots of metrics across subjects for each condition.
"""

import os
import warnings
warnings.filterwarnings('ignore', message='.*Could not initialize NNPACK.*')
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

import torch
import wandb

from model.tcn import TCNModel
from data.dataloader import DataHandler
from config.hyperparameters import DEFAULT_TCN_CONFIG


def load_model(model_path: str, config: dict, device: torch.device):
    model = TCNModel(config).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_normalization_params(save_dir: str):
    input_mean = np.load(os.path.join(save_dir, 'input_mean.npy'))
    input_std = np.load(os.path.join(save_dir, 'input_std.npy'))
    label_mean = np.load(os.path.join(save_dir, 'label_mean.npy'))
    label_std = np.load(os.path.join(save_dir, 'label_std.npy'))
    return input_mean, input_std, label_mean, label_std


def detect_dataset_type(data_root: str) -> str:
    if 'Camargo' in data_root:
        return 'camargo'
    if 'Keaton' in data_root:
        return 'keaton'
    if 'Molinaro' in data_root:
        return 'molinaro'
    if 'MetaMobility' in data_root:
        return 'memo'
    return 'unknown'


def butter_lowpass_zero_phase(data: np.ndarray, cutoff_hz: float = 6.0, fs_hz: float = 100.0, order: int = 4) -> np.ndarray:
    if data is None or data.size == 0:
        return data
    nyq = 0.5 * fs_hz
    wn = cutoff_hz / nyq
    b, a = signal.butter(order, wn, btype='low', analog=False)
    try:
        return signal.filtfilt(b, a, data.squeeze(), axis=0, method='pad', padlen=min(3 * max(len(a), len(b)), max(0, len(data) - 1))).reshape(-1, 1)
    except ValueError:
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
    imu_segments: List[str],
    label_filter_hz: float = 6.0,
    normalize: bool = True,
    dataset_type: str = 'unknown',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    imu_path = os.path.join(trial_path, 'Input', 'imu_data.csv')
    label_path = os.path.join(trial_path, 'Label', 'joint_moment.csv')
    if not (os.path.exists(imu_path) and os.path.exists(label_path)):
        return None, None, None

    try:
        imu_df = pd.read_csv(imu_path, sep=None, engine='python', on_bad_lines='skip')
        label_df = pd.read_csv(label_path, sep=None, engine='python', on_bad_lines='skip')
    except Exception:
        imu_df = pd.read_csv(imu_path, sep=',', on_bad_lines='skip')
        label_df = pd.read_csv(label_path, sep=',', on_bad_lines='skip')

    gyro_cols = [col for col in imu_df.columns if 'gyro' in col.lower()]

    if len(imu_segments) == 1 and imu_segments[0].lower() in ['femur', 'thigh']:
        thigh_r_gyro = [c for c in gyro_cols if 'thigh_r' in c.lower() or 'femur_r' in c.lower()]
        thigh_l_gyro = [c for c in gyro_cols if 'thigh_l' in c.lower() or 'femur_l' in c.lower()]
        if not thigh_r_gyro or len(thigh_r_gyro) < 3:
            return None, None, None
        input_data_r = imu_df[thigh_r_gyro[:3]].values
        input_data_l = imu_df[thigh_l_gyro[:3]].values if thigh_l_gyro and len(thigh_l_gyro) >= 3 else None
        if dataset_type == 'memo':
            input_data = input_data_r
        else:
            if input_data_l is not None and len(input_data_l) > 0:
                input_data = np.vstack((input_data_r, input_data_l)) if np.random.randint(0, 2) else np.vstack((input_data_l, input_data_r))
            else:
                input_data = input_data_r
    elif len(imu_segments) == 2:
        pelvis_gyro = [c for c in gyro_cols if 'pelvis' in c.lower()]
        thigh_r_gyro = [c for c in gyro_cols if 'thigh_r' in c.lower() or 'femur_r' in c.lower()]
        thigh_l_gyro = [c for c in gyro_cols if 'thigh_l' in c.lower() or 'femur_l' in c.lower()]
        if not pelvis_gyro or len(pelvis_gyro) < 3 or not thigh_r_gyro or len(thigh_r_gyro) < 3:
            return None, None, None
        input_data_r = imu_df[pelvis_gyro[:3] + thigh_r_gyro[:3]].values
        input_data_l = imu_df[pelvis_gyro[:3] + thigh_l_gyro[:3]].values if thigh_l_gyro and len(thigh_l_gyro) >= 3 else None
        if dataset_type == 'memo':
            input_data = input_data_r
        else:
            if input_data_l is not None and len(input_data_l) > 0:
                input_data = np.vstack((input_data_r, input_data_l)) if np.random.randint(0, 2) else np.vstack((input_data_l, input_data_r))
            else:
                input_data = input_data_r
    else:
        return None, None, None

    if dataset_type in ['camargo', 'keaton', 'molinaro']:
        input_data = input_data[::2]

    if normalize:
        input_data = (input_data - input_mean) / input_std

    predictions: List[np.ndarray] = []
    true_labels: List[np.ndarray] = []
    batch_size = 128
    num_windows = len(input_data) - window_size + 1
    if num_windows <= 0:
        return None, None, None

    for batch_start in range(0, num_windows, batch_size):
        batch_end = min(batch_start + batch_size, num_windows)
        batch_windows = []
        for i in range(batch_start, batch_end):
            window = input_data[i: i + window_size]
            batch_windows.append(window.T)
        batch_tensor = torch.FloatTensor(np.stack(batch_windows)).to(device)
        with torch.no_grad():
            batch_preds = model(batch_tensor)
            predictions.append(batch_preds.cpu().numpy())
        del batch_tensor

    if predictions:
        predictions = np.concatenate(predictions, axis=0)

    # Labels
    hip_flexion_r_col = [c for c in label_df.columns if 'hip_flexion_r_moment' in c.lower()]
    hip_flexion_l_col = [c for c in label_df.columns if 'hip_flexion_l_moment' in c.lower()]

    true_data = None
    true_data_r = label_df[hip_flexion_r_col[0]].values.reshape(-1, 1) if hip_flexion_r_col else None
    true_data_l = label_df[hip_flexion_l_col[0]].values.reshape(-1, 1) if hip_flexion_l_col else None

    if dataset_type in ['camargo', 'keaton', 'molinaro']:
        if true_data_r is not None:
            true_data_r = true_data_r[::2]
        if true_data_l is not None:
            true_data_l = true_data_l[::2]
    elif dataset_type == 'memo':
        if true_data_l is not None:
            true_data_l = -true_data_l  # sign flip for left on MetaMobility

    if true_data_r is not None:
        true_data_r = butter_lowpass_zero_phase(true_data_r, cutoff_hz=label_filter_hz)
    if true_data_l is not None:
        true_data_l = butter_lowpass_zero_phase(true_data_l, cutoff_hz=label_filter_hz)

    if dataset_type == 'memo':
        true_data = true_data_r
    else:
        if true_data_r is not None and true_data_l is not None:
            true_data = np.vstack((true_data_r, true_data_l)) if np.random.randint(0, 2) else np.vstack((true_data_l, true_data_r))
        elif true_data_r is not None:
            true_data = true_data_r
        elif true_data_l is not None:
            true_data = true_data_l

    if true_data is None:
        return None, None, None

    for i in range(num_windows):
        idx = min(i + window_size - 1, len(true_data) - 1)
        true_labels.append(true_data[idx])

    if not predictions or not true_labels:
        return None, None, None

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    if normalize:
        predictions_denorm = predictions * label_std + label_mean
        true_labels_denorm = true_labels
    else:
        predictions_denorm = predictions
        true_labels_denorm = true_labels

    return predictions_denorm, true_labels_denorm, input_data


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    y_pred_c = y_pred[mask]
    y_true_c = y_true[mask]
    if y_pred_c.size == 0:
        return float('nan'), float('nan')
    mse = np.mean((y_pred_c - y_true_c) ** 2)
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((y_true_c - y_pred_c) ** 2)
    ss_tot = np.sum((y_true_c - np.mean(y_true_c)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else float('nan')
    return rmse, r2


def discover_subjects(data_root: str) -> List[str]:
    subs = []
    for name in sorted(os.listdir(data_root)):
        if name.startswith('.'):
            continue
        p = os.path.join(data_root, name)
        if os.path.isdir(p):
            subs.append(name)
    return subs


def plot_trends(per_subject_metrics: Dict[str, Dict[str, Tuple[float, float]]], conditions: List[str], save_dir: str) -> None:
    subjects = sorted(per_subject_metrics.keys())
    for cond in conditions:
        rmse_vals = []
        r2_vals = []
        sub_list = []
        for s in subjects:
            if cond in per_subject_metrics[s]:
                rmse, r2 = per_subject_metrics[s][cond]
                rmse_vals.append(rmse)
                r2_vals.append(r2)
                sub_list.append(s)
        if not sub_list:
            continue
        # RMSE trend
        plt.figure(figsize=(12, 4))
        plt.plot(range(len(sub_list)), rmse_vals, 'o-', label=f'RMSE - {cond}')
        plt.xticks(range(len(sub_list)), sub_list, rotation=45, ha='right')
        plt.ylabel('RMSE (N-m/kg)')
        plt.title(f'RMSE across subjects - {cond}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(save_dir, f'rmse_trend_{cond}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        # R2 trend
        plt.figure(figsize=(12, 4))
        plt.plot(range(len(sub_list)), r2_vals, 's--', color='green', label=f'R² - {cond}')
        plt.xticks(range(len(sub_list)), sub_list, rotation=45, ha='right')
        plt.ylabel('R²')
        plt.title(f'R² across subjects - {cond}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(save_dir, f'r2_trend_{cond}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Cross-dataset evaluation over all subjects and per condition')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--data_root', type=str, required=True, help='Path to target dataset root (Canonical format)')
    parser.add_argument('--save_dir', type=str, default='./cross_eval', help='Directory to save outputs')
    parser.add_argument('--conditions', nargs='+', default=['levelground'], help='Conditions to test on')
    parser.add_argument('--imu_segments', nargs='+', default=None, help='IMU segments to use (if None, use config)')
    parser.add_argument('--window_size', type=int, default=None, help='Override window size (optional)')
    parser.add_argument('--no_normalize', action='store_true', help='Disable input normalization')
    parser.add_argument('--wandb_project', type=str, default='transfer-learning', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name (auto)')
    parser.add_argument('--wandb_tags', nargs='+', default=[], help='Wandb tags')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config from training save_dir if present near model_path
    model_dir = Path(args.model_path).parent
    config_path = model_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = DEFAULT_TCN_CONFIG.copy()

    # Override runtime options
    if args.window_size is not None:
        config['window_size'] = args.window_size
    if args.imu_segments is not None:
        config['imu_segments'] = args.imu_segments
    config['normalize'] = not args.no_normalize

    # Determine input size from imu_segments
    imu_segments = config.get('imu_segments', ['pelvis', 'femur']) if args.imu_segments is None else args.imu_segments
    if len(imu_segments) == 1 and imu_segments[0].lower() in ['femur', 'thigh']:
        input_size = 3
    else:
        input_size = 6
    config['input_size'] = input_size
    config['output_size'] = 1

    # Load normalization parameters (use model_dir by default)
    try:
        input_mean, input_std, label_mean, label_std = load_normalization_params(str(model_dir))
    except Exception:
        # fallback to user-provided save_dir
        input_mean, input_std, label_mean, label_std = load_normalization_params(args.save_dir)

    # Build model and load weights
    model = load_model(args.model_path, config, device)

    # W&B init
    if not args.no_wandb:
        if args.wandb_name is None:
            from datetime import datetime
            args.wandb_name = f"cross_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                **config,
                'data_root': args.data_root,
                'conditions': args.conditions,
                'imu_segments': imu_segments,
                'normalize': config['normalize'],
            },
            tags=args.wandb_tags + ['cross-eval', 'tcn'],
        )
        run = wandb.run
    else:
        run = None

    dataset_type = detect_dataset_type(args.data_root)

    # Discover all subjects in target dataset
    subjects = discover_subjects(args.data_root)
    if not subjects:
        print('No subjects found in data_root')
        return

    window_size = int(config.get('window_size', 100))
    normalize = bool(config.get('normalize', True))

    per_subject_metrics: Dict[str, Dict[str, Tuple[float, float]]] = {}
    per_condition_overall: Dict[str, Tuple[float, float]] = {}

    # CSV rows collector
    csv_rows: List[Dict] = []

    for subject in subjects:
        per_subject_metrics[subject] = {}
        for condition in args.conditions:
            subject_path = os.path.join(args.data_root, subject)
            condition_path = os.path.join(subject_path, condition)
            if not os.path.isdir(condition_path):
                continue

            all_preds = []
            all_trues = []

            for trial in sorted(os.listdir(condition_path)):
                trial_path = os.path.join(condition_path, trial)
                if not os.path.isdir(trial_path):
                    continue
                pred, true, _ = predict_on_trial(
                    model,
                    trial_path,
                    input_mean,
                    input_std,
                    label_mean,
                    label_std,
                    window_size,
                    device,
                    imu_segments,
                    config.get('label_filter_hz', 6.0),
                    normalize,
                    dataset_type,
                )
                if pred is None or true is None:
                    continue
                all_preds.append(pred)
                all_trues.append(true)

            if not all_preds:
                continue

            all_preds = np.concatenate(all_preds, axis=0).flatten()
            all_trues = np.concatenate(all_trues, axis=0).flatten()

            rmse, r2 = compute_metrics(all_preds, all_trues)
            per_subject_metrics[subject][condition] = (rmse, r2)
            csv_rows.append({'subject': subject, 'condition': condition, 'rmse': rmse, 'r2': r2})

    # Compute overall per-condition
    for condition in args.conditions:
        cond_rmses = []
        cond_r2s = []
        for s, cond_dict in per_subject_metrics.items():
            if condition in cond_dict:
                rmse, r2 = cond_dict[condition]
                if not (np.isnan(rmse) or np.isnan(r2)):
                    cond_rmses.append(rmse)
                    cond_r2s.append(r2)
        if cond_rmses:
            per_condition_overall[condition] = (float(np.mean(cond_rmses)), float(np.mean(cond_r2s)))

    # Save CSV
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        df.to_csv(os.path.join(args.save_dir, 'cross_eval_per_subject.csv'), index=False)

    # Save per-condition summary
    if per_condition_overall:
        summary_rows = [{'condition': c, 'rmse_mean': v[0], 'r2_mean': v[1]} for c, v in per_condition_overall.items()]
        pd.DataFrame(summary_rows).to_csv(os.path.join(args.save_dir, 'cross_eval_summary.csv'), index=False)

    # Plot trends
    plot_trends(per_subject_metrics, args.conditions, args.save_dir)

    # Log to W&B
    if run is not None:
        if csv_rows:
            wandb.log({'cross_eval/per_subject_table': wandb.Table(dataframe=pd.DataFrame(csv_rows))})
        for cond, (rmse_m, r2_m) in per_condition_overall.items():
            wandb.log({f'cross_eval/{cond}/rmse_mean': rmse_m, f'cross_eval/{cond}/r2_mean': r2_m})
        # Upload images
        for cond in args.conditions:
            rmse_path = os.path.join(args.save_dir, f'rmse_trend_{cond}.png')
            r2_path = os.path.join(args.save_dir, f'r2_trend_{cond}.png')
            if os.path.exists(rmse_path):
                wandb.log({f'cross_eval/{cond}/rmse_trend': wandb.Image(rmse_path)})
            if os.path.exists(r2_path):
                wandb.log({f'cross_eval/{cond}/r2_trend': wandb.Image(r2_path)})
        wandb.finish()

    # Print summary
    print('\nPer-condition summary (mean across subjects):')
    for cond, (rmse_m, r2_m) in per_condition_overall.items():
        print(f"  {cond:<16} RMSE={rmse_m:.4f} N-m/kg   R²={r2_m:.4f}")
    print(f"Results saved to {args.save_dir}")


if __name__ == '__main__':
    main()
