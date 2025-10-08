#!/usr/bin/env python3
"""
Domain check utility: compare hip moment ranges/statistics between two datasets.

Uses the actual DataHandler and LoadData classes to ensure we're comparing
exactly what the training/testing pipeline sees.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))
from data.dataloader import DataHandler
from config.hyperparameters import DEFAULT_TCN_CONFIG


def collect_stats_with_dataloader(dataset_root: str, 
                                 subjects: List[str], 
                                 conditions: List[str],
                                 imu_segments: List[str],
                                 verbose: bool = False):
    """Use DataHandler to load data exactly like training/testing does."""
    
    # Create a minimal config for dataloader
    config = DEFAULT_TCN_CONFIG.copy()
    config.update({
        'data_root': dataset_root,
        'window_size': 100,  # Default window size
        'batch_size': 32,
        'number_of_workers': 0,
        'validation_split': 0.2,
        'dataset_proportion': 1.0,
        'transfer_learning': False,
        'imu_segments': imu_segments,
        'input_size': 6 if len(imu_segments) == 2 else 3,
        'output_size': 1
    })
    
    if verbose:
        print(f"Loading dataset: {dataset_root}")
        print(f"Subjects: {subjects}")
        print(f"Conditions: {conditions}")
        print(f"IMU segments: {imu_segments}")
    
    # Initialize DataHandler
    data_handler = DataHandler(
        data_root=dataset_root,
        hyperparam_config=config,
        pretrained_model_path=None
    )
    
    # Load data using the same logic as training
    try:
        data_handler.load_data(
            train_data_partition=subjects,
            train_data_condition=conditions,
            test_data_partition=[]  # We only need training data for stats
        )
        
        # Extract labels from the loaded data
        labels = data_handler.train_data.label
        
        if len(labels) == 0:
            return pd.DataFrame(), {"min": np.nan, "max": np.nan, "span": np.nan, 
                                 "mean": np.nan, "std": np.nan, "mean_abs": np.nan,
                                 "num_samples": 0, "num_trials": 0}
        
        # Calculate statistics
        labels_flat = labels.flatten()
        labels_clean = labels_flat[~np.isnan(labels_flat)]
        
        if len(labels_clean) == 0:
            return pd.DataFrame(), {"min": np.nan, "max": np.nan, "span": np.nan, 
                                 "mean": np.nan, "std": np.nan, "mean_abs": np.nan,
                                 "num_samples": 0, "num_trials": 0}
        
        overall = {
            "min": float(np.min(labels_clean)),
            "max": float(np.max(labels_clean)),
            "span": float(np.max(labels_clean) - np.min(labels_clean)),
            "mean": float(np.mean(labels_clean)),
            "std": float(np.std(labels_clean)),
            "mean_abs": float(np.mean(np.abs(labels_clean))),
            "num_samples": int(len(labels_clean)),
            "num_trials": len(data_handler.train_data.input_list),
        }
        
        # Create summary DataFrame
        records = []
        for i, (input_data, label_data) in enumerate(zip(data_handler.train_data.input_list, 
                                                         data_handler.train_data.label_list)):
            if len(label_data) > 0:
                label_flat = label_data.flatten()
                label_clean = label_flat[~np.isnan(label_flat)]
                if len(label_clean) > 0:
                    records.append({
                        "trial_idx": i,
                        "min": float(np.min(label_clean)),
                        "max": float(np.max(label_clean)),
                        "span": float(np.max(label_clean) - np.min(label_clean)),
                        "mean": float(np.mean(label_clean)),
                        "std": float(np.std(label_clean)),
                        "mean_abs": float(np.mean(np.abs(label_clean))),
                        "samples": int(len(label_clean))
                    })
        
        summary_df = pd.DataFrame.from_records(records)
        return summary_df, overall
        
    except Exception as e:
        if verbose:
            print(f"Error loading {dataset_root}: {e}")
        return pd.DataFrame(), {"min": np.nan, "max": np.nan, "span": np.nan, 
                               "mean": np.nan, "std": np.nan, "mean_abs": np.nan,
                               "num_samples": 0, "num_trials": 0}


def print_comparison(name_a: str, overall_a: Dict, name_b: str, overall_b: Dict):
    def fmt(d, k):
        v = d.get(k, np.nan)
        return f"{v:.4f}" if isinstance(v, (float, int)) and not np.isnan(v) else str(v)

    print("\n=== Domain Check: Hip Flexion Moment (N-m/kg) ===")
    print(f"Dataset A ({name_a}) overall:")
    print(f"  min={fmt(overall_a, 'min')}  max={fmt(overall_a, 'max')}  span={fmt(overall_a, 'span')}")
    print(f"  mean={fmt(overall_a, 'mean')}  std={fmt(overall_a, 'std')}  mean|.|={fmt(overall_a, 'mean_abs')}")
    print(f"  trials={overall_a.get('num_trials', 0)}  samples={overall_a.get('num_samples', 0)}")

    print(f"\nDataset B ({name_b}) overall:")
    print(f"  min={fmt(overall_b, 'min')}  max={fmt(overall_b, 'max')}  span={fmt(overall_b, 'span')}")
    print(f"  mean={fmt(overall_b, 'mean')}  std={fmt(overall_b, 'std')}  mean|.|={fmt(overall_b, 'mean_abs')}")
    print(f"  trials={overall_b.get('num_trials', 0)}  samples={overall_b.get('num_samples', 0)}")

    if all(not np.isnan(v) for v in [overall_a.get('mean_abs', np.nan), overall_b.get('mean_abs', np.nan)]):
        ratio = overall_b['mean_abs'] / overall_a['mean_abs'] if overall_a['mean_abs'] != 0 else np.nan
        print(f"\nScale ratio (B/A by mean|.|): {ratio:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare hip moment ranges across two datasets using DataHandler")
    parser.add_argument("--dataset_a", required=True, help="Path to dataset A root (Canonical-style)")
    parser.add_argument("--dataset_b", required=True, help="Path to dataset B root (Canonical-style)")
    parser.add_argument("--name_a", default="A", help="Label/name for dataset A")
    parser.add_argument("--name_b", default="B", help="Label/name for dataset B")
    parser.add_argument("--subjects", default=None, help="Comma-separated subject IDs to include (optional)")
    parser.add_argument("--conditions", default=None, help="Comma-separated conditions to include (optional)")
    parser.add_argument("--imu_segments", nargs="+", default=["pelvis", "femur"], 
                       help="IMU segments to use (default: ['pelvis', 'femur'])")
    parser.add_argument("--out", default="./domain_check_out", help="Output directory for CSV summaries")
    parser.add_argument("--verbose", action="store_true", help="Print progress while scanning")
    args = parser.parse_args()

    root_a = Path(args.dataset_a)
    root_b = Path(args.dataset_b)
    if not root_a.exists() or not root_b.exists():
        raise FileNotFoundError("One or both dataset roots do not exist")

    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None
    conditions = [c.strip() for c in args.conditions.split(",")] if args.conditions else None

    # Use DataHandler to load data exactly like training/testing
    summary_a, overall_a = collect_stats_with_dataloader(
        str(root_a), subjects, conditions, args.imu_segments, verbose=args.verbose
    )
    summary_b, overall_b = collect_stats_with_dataloader(
        str(root_b), subjects, conditions, args.imu_segments, verbose=args.verbose
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_a.to_csv(out_dir / f"summary_{args.name_a}.csv", index=False)
    summary_b.to_csv(out_dir / f"summary_{args.name_b}.csv", index=False)

    print_comparison(args.name_a, overall_a, args.name_b, overall_b)
    print(f"\nSaved per-trial summaries to: {out_dir}")


if __name__ == "__main__":
    main()