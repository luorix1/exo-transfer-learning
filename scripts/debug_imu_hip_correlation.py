#!/usr/bin/env python3
"""
Debug script to plot thigh IMU z-axis gyro values alongside corresponding hip moments.
Shows raw IMU data (no sign flipping) vs smoothed hip moments for correlation analysis.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Safe non-interactive backend
import matplotlib.pyplot as plt
from scipy import signal


def find_input_file(trial_path: str) -> str:
    """Find IMU input file in the Input directory."""
    input_dir = os.path.join(trial_path, "Input")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Look for IMU files
    for name in sorted(os.listdir(input_dir)):
        if 'imu' in name.lower() and name.lower().endswith('.csv'):
            return os.path.join(input_dir, name)

    raise FileNotFoundError(f"No IMU CSV files found in {input_dir}")


def find_label_file(trial_path: str) -> str:
    """Find label file in the Label directory."""
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


def load_csv_file(csv_path: str) -> pd.DataFrame:
    """Load CSV file with flexible delimiter handling."""
    try:
        return pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='skip')
    except Exception:
        return pd.read_csv(csv_path, sep=',', on_bad_lines='skip')


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


def extract_imu_gyro_data(imu_df: pd.DataFrame):
    """Extract thigh IMU z-axis gyro data for both sides."""
    gyro_cols = [col for col in imu_df.columns if 'gyro' in col.lower()]
    
    # Find thigh gyro columns
    thigh_r_gyro = [col for col in gyro_cols if 'thigh_r' in col.lower() or 'femur_r' in col.lower()]
    thigh_l_gyro = [col for col in gyro_cols if 'thigh_l' in col.lower() or 'femur_l' in col.lower()]
    
    # Extract z-axis (index 2) for each side
    thigh_r_z_gyro = None
    thigh_l_z_gyro = None
    
    if thigh_r_gyro and len(thigh_r_gyro) >= 3:
        thigh_r_z_gyro = imu_df[thigh_r_gyro[2]].values  # Z-axis is index 2
    
    if thigh_l_gyro and len(thigh_l_gyro) >= 3:
        thigh_l_z_gyro = imu_df[thigh_l_gyro[2]].values  # Z-axis is index 2
    
    return thigh_r_z_gyro, thigh_l_z_gyro


def extract_hip_moment_data(label_df: pd.DataFrame):
    """Extract hip flexion moment data for both sides."""
    # Look for hip flexion moments
    hip_flexion_r_col = [col for col in label_df.columns 
                        if 'hip_flexion_r_moment' in col.lower()]
    hip_flexion_l_col = [col for col in label_df.columns 
                        if 'hip_flexion_l_moment' in col.lower()]
    
    hip_r_moment = None
    hip_l_moment = None
    
    if hip_flexion_r_col:
        hip_r_moment = label_df[hip_flexion_r_col[0]].values.astype(float)
    
    if hip_flexion_l_col:
        hip_l_moment = label_df[hip_flexion_l_col[0]].values.astype(float)
    
    return hip_r_moment, hip_l_moment


def plot_imu_hip_correlation(imu_r_z, imu_l_z, hip_r_moment, hip_l_moment, 
                           title: str, save_path: str = None, label_filter_hz: float = 6.0):
    """Plot thigh IMU z-axis gyro vs hip flexion moment for both sides."""
    
    # Create 2x1 subplots for right and left sides
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Right side plot
    ax_r = axes[0]
    if imu_r_z is not None and hip_r_moment is not None:
        # Ensure same length
        min_len = min(len(imu_r_z), len(hip_r_moment))
        imu_r_z = imu_r_z[:min_len]
        hip_r_moment = hip_r_moment[:min_len]
        
        x = np.arange(min_len)
        
        # Plot raw IMU z-axis gyro
        ax_r.plot(x, imu_r_z, 'b-', linewidth=1.0, alpha=0.7, label='Thigh R Z Gyro (Raw)')
        
        # Plot raw and smoothed hip moment
        ax_r.plot(x, hip_r_moment, 'r:', linewidth=1.0, alpha=0.6, label='Hip R Flexion Moment (Raw)')
        
        # Apply smoothing to hip moment
        hip_r_smooth = butter_lowpass_zero_phase(hip_r_moment.reshape(-1, 1), cutoff_hz=label_filter_hz).flatten()
        ax_r.plot(x, hip_r_smooth, 'r-', linewidth=2.0, alpha=0.9, label=f'Hip R Flexion Moment (Smoothed {label_filter_hz}Hz)')
        
        # Calculate correlation
        corr_raw = np.corrcoef(imu_r_z, hip_r_moment)[0, 1]
        corr_smooth = np.corrcoef(imu_r_z, hip_r_smooth)[0, 1]
        
        ax_r.set_title(f'Right Side: Thigh Z Gyro vs Hip Flexion Moment\n'
                      f'Correlation (Raw): {corr_raw:.3f}, Correlation (Smoothed): {corr_smooth:.3f}', 
                      fontsize=12, fontweight='bold')
        ax_r.set_ylabel('Amplitude')
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(loc='upper right')
        
        # Add statistics
        stats_text = (f"IMU Z: mean={np.nanmean(imu_r_z):.4f}, std={np.nanstd(imu_r_z):.4f}\n"
                     f"Hip Raw: mean={np.nanmean(hip_r_moment):.4f}, std={np.nanstd(hip_r_moment):.4f}\n"
                     f"Hip Smooth: mean={np.nanmean(hip_r_smooth):.4f}, std={np.nanstd(hip_r_smooth):.4f}")
        ax_r.text(0.01, 0.99, stats_text, transform=ax_r.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax_r.set_title('Right Side: No data available', fontsize=12, fontweight='bold')
        ax_r.text(0.5, 0.5, 'No right side data found', transform=ax_r.transAxes, 
                 ha='center', va='center', fontsize=14)
    
    # Left side plot
    ax_l = axes[1]
    if imu_l_z is not None and hip_l_moment is not None:
        # Ensure same length
        min_len = min(len(imu_l_z), len(hip_l_moment))
        imu_l_z = imu_l_z[:min_len]
        hip_l_moment = hip_l_moment[:min_len]
        
        x = np.arange(min_len)
        
        # Plot raw IMU z-axis gyro (NO sign flipping for debugging)
        ax_l.plot(x, imu_l_z, 'g-', linewidth=1.0, alpha=0.7, label='Thigh L Z Gyro (Raw)')
        
        # Plot raw and smoothed hip moment
        ax_l.plot(x, hip_l_moment, 'm:', linewidth=1.0, alpha=0.6, label='Hip L Flexion Moment (Raw)')
        
        # Apply smoothing to hip moment
        hip_l_smooth = butter_lowpass_zero_phase(hip_l_moment.reshape(-1, 1), cutoff_hz=label_filter_hz).flatten()
        ax_l.plot(x, hip_l_smooth, 'm-', linewidth=2.0, alpha=0.9, label=f'Hip L Flexion Moment (Smoothed {label_filter_hz}Hz)')
        
        # Calculate correlation
        corr_raw = np.corrcoef(imu_l_z, hip_l_moment)[0, 1]
        corr_smooth = np.corrcoef(imu_l_z, hip_l_smooth)[0, 1]
        
        ax_l.set_title(f'Left Side: Thigh Z Gyro vs Hip Flexion Moment\n'
                      f'Correlation (Raw): {corr_raw:.3f}, Correlation (Smoothed): {corr_smooth:.3f}', 
                      fontsize=12, fontweight='bold')
        ax_l.set_xlabel('Sample Index')
        ax_l.set_ylabel('Amplitude')
        ax_l.grid(True, alpha=0.3)
        ax_l.legend(loc='upper right')
        
        # Add statistics
        stats_text = (f"IMU Z: mean={np.nanmean(imu_l_z):.4f}, std={np.nanstd(imu_l_z):.4f}\n"
                     f"Hip Raw: mean={np.nanmean(hip_l_moment):.4f}, std={np.nanstd(hip_l_moment):.4f}\n"
                     f"Hip Smooth: mean={np.nanmean(hip_l_smooth):.4f}, std={np.nanstd(hip_l_smooth):.4f}")
        ax_l.text(0.01, 0.99, stats_text, transform=ax_l.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax_l.set_title('Left Side: No data available', fontsize=12, fontweight='bold')
        ax_l.text(0.5, 0.5, 'No left side data found', transform=ax_l.transAxes, 
                 ha='center', va='center', fontsize=14)
        ax_l.set_xlabel('Sample Index')
    
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
    parser = argparse.ArgumentParser(description="Debug IMU-Hip moment correlation analysis")
    parser.add_argument("--data_root", type=str, required=True, help="Root of Canonical dataset (e.g., Canonical_Camargo)")
    parser.add_argument("--subject", type=str, required=True, help="Subject folder name")
    parser.add_argument("--condition", type=str, required=True, help="Condition folder name")
    parser.add_argument("--trial", type=str, required=True, help="Trial folder name")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure (PNG)")
    parser.add_argument("--label_filter_hz", type=float, default=6.0, help="Low-pass cutoff frequency (Hz) for smoothing (default: 6.0)")

    args = parser.parse_args()

    trial_path = os.path.join(args.data_root, args.subject, args.condition, args.trial)
    if not os.path.isdir(trial_path):
        raise FileNotFoundError(f"Trial path not found: {trial_path}")

    # Load IMU data
    imu_path = find_input_file(trial_path)
    print(f"Reading IMU data: {imu_path}")
    imu_df = load_csv_file(imu_path)
    
    # Load label data
    label_path = find_label_file(trial_path)
    print(f"Reading label data: {label_path}")
    label_df = load_csv_file(label_path)

    # Extract data
    thigh_r_z_gyro, thigh_l_z_gyro = extract_imu_gyro_data(imu_df)
    hip_r_moment, hip_l_moment = extract_hip_moment_data(label_df)
    
    # Print found data
    print("Found data:")
    print(f"  Right thigh Z gyro: {'Yes' if thigh_r_z_gyro is not None else 'No'}")
    print(f"  Left thigh Z gyro: {'Yes' if thigh_l_z_gyro is not None else 'No'}")
    print(f"  Right hip moment: {'Yes' if hip_r_moment is not None else 'No'}")
    print(f"  Left hip moment: {'Yes' if hip_l_moment is not None else 'No'}")
    
    if thigh_r_z_gyro is not None:
        print(f"  Right thigh Z gyro shape: {thigh_r_z_gyro.shape}")
    if thigh_l_z_gyro is not None:
        print(f"  Left thigh Z gyro shape: {thigh_l_z_gyro.shape}")
    if hip_r_moment is not None:
        print(f"  Right hip moment shape: {hip_r_moment.shape}")
    if hip_l_moment is not None:
        print(f"  Left hip moment shape: {hip_l_moment.shape}")

    title = f"IMU-Hip Correlation Debug: {args.subject}/{args.condition}/{args.trial}"
    save_path = args.save
    if save_path is None:
        # default save next to script root if not provided
        safe_name = f"imu_hip_correlation_{args.subject}_{args.condition}_{args.trial}.png".replace('/', '_').replace('\\', '_')
        save_path = os.path.join(os.getcwd(), safe_name)

    plot_imu_hip_correlation(thigh_r_z_gyro, thigh_l_z_gyro, hip_r_moment, hip_l_moment, 
                           title, save_path, args.label_filter_hz)


if __name__ == "__main__":
    main()
