#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_trial_paths(data_root: str, subject: str, condition: str, trial: str) -> tuple[str, str]:
    trial_dir = os.path.join(data_root, subject, condition, trial)
    imu_csv = os.path.join(trial_dir, 'Input', 'imu_data.csv')
    label_csv = os.path.join(trial_dir, 'Label', 'joint_moment.csv')
    if not os.path.isfile(imu_csv) or not os.path.isfile(label_csv):
        raise FileNotFoundError(f"Missing files under {trial_dir}:\n  {imu_csv}\n  {label_csv}")
    return imu_csv, label_csv


def read_csv_flexible(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
    except Exception:
        return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description='Plot right hip flexion moment vs right thigh gyro (x,y,z) for Canonical_Molinaro')
    parser.add_argument('--data_root', required=True, help='Path to Canonical_Molinaro root')
    parser.add_argument('--subject', required=True, help='Subject ID (e.g., AB21)')
    parser.add_argument('--condition', required=True, help='Condition (e.g., levelground, stair, ramp)')
    parser.add_argument('--trial', required=True, help='Trial folder name (e.g., levelground_0.0_0.75_01)')
    parser.add_argument('--save', default=None, help='Optional output png path')

    args = parser.parse_args()

    imu_csv, label_csv = find_trial_paths(args.data_root, args.subject, args.condition, args.trial)

    imu_df = read_csv_flexible(imu_csv)
    label_df = read_csv_flexible(label_csv)

    if 'time' not in imu_df.columns or 'time' not in label_df.columns:
        raise ValueError('Both imu_data.csv and joint_moment.csv must contain a "time" column')

    needed_imu = ['thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z']
    missing_imu = [c for c in needed_imu if c not in imu_df.columns]
    if missing_imu:
        raise ValueError(f"Missing IMU columns: {missing_imu} in {imu_csv}")

    needed_label = ['hip_flexion_r_moment']
    missing_lab = [c for c in needed_label if c not in label_df.columns]
    if missing_lab:
        raise ValueError(f"Missing label columns: {missing_lab} in {label_csv}")

    merged = pd.merge(imu_df[['time'] + needed_imu],
                      label_df[['time'] + needed_label],
                      on='time', how='inner')

    # Use only rows without NaN in any of the selected columns
    cols = ['time'] + needed_imu + needed_label
    clean = merged.dropna(subset=cols).reset_index(drop=True)
    if clean.empty:
        raise ValueError('No overlapping non-NaN samples after sync on time')

    t = clean['time'].to_numpy()
    gxr = clean['thigh_r_gyro_x'].to_numpy()
    gyr = clean['thigh_r_gyro_y'].to_numpy()
    gzr = clean['thigh_r_gyro_z'].to_numpy()
    hip_flex_r = clean['hip_flexion_r_moment'].to_numpy()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, gxr, 'b-', linewidth=1.0, label='thigh_r_gyro_x')
    axes[0].set_ylabel('Gyro X (rad/s)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    axes[1].plot(t, gyr, 'g-', linewidth=1.0, label='thigh_r_gyro_y')
    axes[1].set_ylabel('Gyro Y (rad/s)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    axes[2].plot(t, gzr, 'r-', linewidth=1.0, label='thigh_r_gyro_z')
    axes[2].set_ylabel('Gyro Z (rad/s)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    axes[3].plot(t, hip_flex_r, 'm-', linewidth=1.2, label='hip_flexion_r_moment')
    axes[3].set_ylabel('Moment (N·m/kg)')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')

    fig.suptitle(f"{args.subject}/{args.condition}/{args.trial}: Right Hip Flexion vs Right Thigh Gyro")
    plt.tight_layout()

    out_path = args.save
    if out_path is None:
        safe = f"plot_rhip_vs_rthigh_{args.subject}_{args.condition}_{args.trial}.png".replace('/', '_')
        out_path = os.path.join(os.getcwd(), safe)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()


