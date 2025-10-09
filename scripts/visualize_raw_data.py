#!/usr/bin/env python3
"""
Visualize raw hip joint moments directly from a trial's Label/joint_moment.csv.

- No filtering
- No normalization/denormalization
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Safe non-interactive backend
import matplotlib.pyplot as plt
from scipy import signal


def find_label_file(trial_path: str) -> str:
    label_dir = os.path.join(trial_path, "Label")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    # Prefer canonical name
    canonical = os.path.join(label_dir, "joint_moment.csv")
    if os.path.exists(canonical):
        return canonical

    # Fallback to first CSV file
    for name in sorted(os.listdir(label_dir)):
        if name.lower().endswith('.csv'):
            return os.path.join(label_dir, name)

    raise FileNotFoundError(f"No CSV files found in {label_dir}")


def load_label_df(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='skip')
    except Exception:
        return pd.read_csv(csv_path, sep=',', on_bad_lines='skip')


def pick_columns(label_df: pd.DataFrame):
    cols_lower = [c.lower() for c in label_df.columns]

    # Find all three axes for right and left hip moments
    r_flexion = None
    r_adduction = None
    r_rotation = None
    l_flexion = None
    l_adduction = None
    l_rotation = None
    
    for i, c in enumerate(cols_lower):
        # Right hip moments
        if r_flexion is None and 'hip_flexion_r_moment' in c:
            r_flexion = label_df.columns[i]
        if r_adduction is None and 'hip_adduction_r_moment' in c:
            r_adduction = label_df.columns[i]
        if r_rotation is None and 'hip_rotation_r_moment' in c:
            r_rotation = label_df.columns[i]
        
        # Left hip moments
        if l_flexion is None and 'hip_flexion_l_moment' in c:
            l_flexion = label_df.columns[i]
        if l_adduction is None and 'hip_adduction_l_moment' in c:
            l_adduction = label_df.columns[i]
        if l_rotation is None and 'hip_rotation_l_moment' in c:
            l_rotation = label_df.columns[i]

    return {
        'r_flexion': r_flexion,
        'r_adduction': r_adduction,
        'r_rotation': r_rotation,
        'l_flexion': l_flexion,
        'l_adduction': l_adduction,
        'l_rotation': l_rotation,
    }


def butter_lowpass_zero_phase(data: np.ndarray, cutoff_hz: float = 6.0, fs_hz: float = 100.0, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter (same as in dataloader.py)."""
    if data is None or data.size == 0:
        return data
    
    # Design Butterworth filter
    nyq = 0.5 * fs_hz
    wn = float(cutoff_hz) / nyq
    b, a = signal.butter(order, wn, btype='low', analog=False)
    
    # Apply filtfilt for zero-phase filtering
    try:
        return signal.filtfilt(b, a, data.squeeze(), axis=0, method='pad', 
                              padlen=min(3 * max(len(a), len(b)), max(0, len(data) - 1))).reshape(-1, 1)
    except ValueError:
        # If sequence too short for padlen, fall back to lfilter twice
        y = signal.lfilter(b, a, data.squeeze(), axis=0)
        y = signal.lfilter(b, a, y[::-1], axis=0)[::-1]
        return y.reshape(-1, 1)


def plot_raw(label_df: pd.DataFrame, columns: dict, title: str, save_path: str = None, 
             label_filter_hz: float = 6.0, show_smoothed: bool = True):
    num_samples = len(label_df)
    x = np.arange(num_samples)  # sample index; dataset is typically 100 Hz

    # Create 3x1 subplots for flexion, adduction, rotation
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    moment_types = ['flexion', 'adduction', 'rotation']
    colors_r = ['b', 'g', 'c']
    colors_l = ['r', 'm', 'orange']
    
    for idx, moment_type in enumerate(moment_types):
        ax = axes[idx]
        stats_lines = []
        
        # Plot right side
        r_col = columns[f'r_{moment_type}']
        if r_col is not None:
            y_r = label_df[r_col].values.astype(float)
            ax.plot(x, y_r, color=colors_r[idx], linewidth=1.0, alpha=0.6, label=f"Right {moment_type.capitalize()} (Raw)")
            
            # Apply smoothing if requested
            if show_smoothed:
                y_r_smooth = butter_lowpass_zero_phase(y_r.reshape(-1, 1), cutoff_hz=label_filter_hz).flatten()
                ax.plot(x, y_r_smooth, color=colors_r[idx], linewidth=2.0, alpha=0.9, 
                       label=f"Right {moment_type.capitalize()} (Smoothed {label_filter_hz}Hz)")
                stats_lines.append(f"R Raw: mean={np.nanmean(y_r):.4f}, std={np.nanstd(y_r):.4f}")
                stats_lines.append(f"R Smooth: mean={np.nanmean(y_r_smooth):.4f}, std={np.nanstd(y_r_smooth):.4f}")
            else:
                stats_lines.append(f"R: mean={np.nanmean(y_r):.4f}, std={np.nanstd(y_r):.4f}, range=[{np.nanmin(y_r):.4f}, {np.nanmax(y_r):.4f}]")
        
        # Plot left side
        l_col = columns[f'l_{moment_type}']
        if l_col is not None:
            y_l = label_df[l_col].values.astype(float)
            ax.plot(x, y_l, color=colors_l[idx], linestyle=':', linewidth=1.0, alpha=0.6, 
                   label=f"Left {moment_type.capitalize()} (Raw)")
            
            # Apply smoothing if requested
            if show_smoothed:
                y_l_smooth = butter_lowpass_zero_phase(y_l.reshape(-1, 1), cutoff_hz=label_filter_hz).flatten()
                ax.plot(x, y_l_smooth, color=colors_l[idx], linestyle='--', linewidth=2.0, alpha=0.9,
                       label=f"Left {moment_type.capitalize()} (Smoothed {label_filter_hz}Hz)")
                stats_lines.append(f"L Raw: mean={np.nanmean(y_l):.4f}, std={np.nanstd(y_l):.4f}")
                stats_lines.append(f"L Smooth: mean={np.nanmean(y_l_smooth):.4f}, std={np.nanstd(y_l_smooth):.4f}")
            else:
                stats_lines.append(f"L: mean={np.nanmean(y_l):.4f}, std={np.nanstd(y_l):.4f}, range=[{np.nanmin(y_l):.4f}, {np.nanmax(y_l):.4f}]")
        
        ax.set_title(f"Hip {moment_type.capitalize()} Moment", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Moment (unit as-is in CSV)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        if stats_lines:
            ax.text(0.01, 0.99, "\n".join(stats_lines), transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize raw hip joint moments from Label/joint_moment.csv")
    parser.add_argument("--data_root", type=str, required=True, help="Root of Canonical dataset (e.g., Canonical_Camargo)")
    parser.add_argument("--subject", type=str, required=True, help="Subject folder name")
    parser.add_argument("--condition", type=str, required=True, help="Condition folder name")
    parser.add_argument("--trial", type=str, required=True, help="Trial folder name")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure (PNG)")
    parser.add_argument("--label_filter_hz", type=float, default=6.0, help="Low-pass cutoff frequency (Hz) for smoothing (default: 6.0)")
    parser.add_argument("--no_smooth", action="store_true", help="Disable smoothed overlay (show only raw data)")

    args = parser.parse_args()

    trial_path = os.path.join(args.data_root, args.subject, args.condition, args.trial)
    if not os.path.isdir(trial_path):
        raise FileNotFoundError(f"Trial path not found: {trial_path}")

    csv_path = find_label_file(trial_path)
    print(f"Reading: {csv_path}")
    df = load_label_df(csv_path)

    columns = pick_columns(df)
    
    # Check if we found any hip moment columns
    has_data = any(v is not None for v in columns.values())
    if not has_data:
        raise ValueError("Could not find hip moment columns in the label CSV.")
    
    # Print found columns
    print("Found columns:")
    for key, val in columns.items():
        if val is not None:
            print(f"  {key}: {val}")

    show_smoothed = not args.no_smooth
    if show_smoothed:
        title = f"Raw + Smoothed Hip Joint Moments (All Axes): {args.subject}/{args.condition}/{args.trial}"
        filename_suffix = "raw_smooth_3axis"
    else:
        title = f"Raw Hip Joint Moments (All Axes): {args.subject}/{args.condition}/{args.trial}"
        filename_suffix = "raw_3axis"
    
    save_path = args.save
    if save_path is None:
        # default save next to script root if not provided
        safe_name = f"{filename_suffix}_{args.subject}_{args.condition}_{args.trial}.png".replace('/', '_').replace('\\', '_')
        save_path = os.path.join(os.getcwd(), safe_name)

    plot_raw(df, columns, title, save_path, args.label_filter_hz, show_smoothed)


if __name__ == "__main__":
    main()


