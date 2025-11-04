#!/usr/bin/env python3
"""
Script to calculate and compare statistics between Canonical_Camargo and Canonical_MetaMobility datasets.
This helps understand distribution differences that may affect transfer learning.
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    # Seaborn is optional, matplotlib will work fine without it
    pass

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataloader import LoadData
from config.hyperparameters import DEFAULT_TCN_CONFIG

# Set style for better-looking plots
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def detect_dataset_type(data_root: str) -> str:
    """Detect dataset type based on data_root path."""
    if 'Canonical_Camargo' in data_root or 'Camargo' in data_root:
        return 'camargo'
    elif 'Canonical_MetaMobility' in data_root or 'MetaMobility' in data_root or 'memo' in data_root.lower():
        return 'memo'
    else:
        return 'unknown'


def get_all_subjects_and_conditions(data_root: str):
    """Get all subjects and conditions available in the dataset."""
    subjects = []
    conditions = {}
    
    if not os.path.exists(data_root):
        print(f"Warning: Data root does not exist: {data_root}")
        return subjects, conditions
    
    print(f"\nScanning directory structure in: {data_root}")
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            subjects.append(item)
            conditions[item] = []
            
            # Debug: print what's in each subject directory
            subject_contents = os.listdir(item_path)
            print(f"  Subject {item}: {len(subject_contents)} items")
            
            for condition_item in subject_contents:
                condition_path = os.path.join(item_path, condition_item)
                if os.path.isdir(condition_path) and not condition_item.startswith('.'):
                    # Check if it has Input and Label directories
                    input_dir = os.path.join(condition_path, 'Input')
                    label_dir = os.path.join(condition_path, 'Label')
                    if os.path.exists(input_dir) and os.path.exists(label_dir):
                        conditions[item].append(condition_item)
                        print(f"    Found condition: {condition_item}")
                    else:
                        # Check what's actually in this directory
                        if os.path.exists(input_dir) or os.path.exists(label_dir):
                            print(f"    Partial condition {condition_item}: has Input={os.path.exists(input_dir)}, Label={os.path.exists(label_dir)}")
    
    print(f"\nFound {len(subjects)} subjects")
    all_conditions = set()
    for subj, conds in conditions.items():
        if conds:
            all_conditions.update(conds)
    print(f"Found {len(all_conditions)} unique conditions across all subjects: {sorted(all_conditions)}")
    
    return sorted(subjects), conditions


def compute_dataset_statistics(
    data_root: str,
    subjects: list = None,
    conditions: list = None,
    imu_segments: list = ['pelvis', 'femur'],
    label_filter_hz: float = 6.0,
    dataset_type: str = 'unknown'
):
    """
    Compute statistics for a dataset.
    
    Args:
        data_root: Root directory of the dataset
        subjects: List of subject IDs to include (None = all)
        conditions: List of conditions to include (None = all)
        imu_segments: IMU segments to use
        label_filter_hz: Low-pass filter frequency for labels
        dataset_type: Type of dataset (affects downsampling)
    """
    print(f"\n{'='*80}")
    print(f"Computing statistics for: {data_root}")
    print(f"{'='*80}")
    
    # Get all available subjects and conditions if not specified
    all_subjects, all_conditions_dict = get_all_subjects_and_conditions(data_root)
    
    if subjects is None:
        subjects = all_subjects
        print(f"Using all subjects: {subjects}")
    else:
        # Filter to only subjects that exist
        subjects = [s for s in subjects if s in all_subjects]
        if not subjects:
            print(f"Warning: None of the specified subjects found in {data_root}")
            return None
    
    # Collect unique conditions across all subjects
    if conditions is None:
        all_condition_sets = [set(all_conditions_dict.get(s, [])) for s in subjects]
        if all_condition_sets:
            # Filter out empty sets
            non_empty_sets = [s for s in all_condition_sets if s]
            if non_empty_sets:
                conditions = sorted(list(set.union(*non_empty_sets)))
                print(f"Using all conditions found: {conditions}")
            else:
                print(f"Warning: No conditions found for any of the specified subjects")
                print(f"  Available subjects and their conditions:")
                for subj in subjects:
                    conds = all_conditions_dict.get(subj, [])
                    print(f"    {subj}: {conds if conds else 'None'}")
                return None
        else:
            print(f"Warning: No conditions found")
            return None
    else:
        print(f"Using specified conditions: {conditions}")
    
    # Prepare config
    config = DEFAULT_TCN_CONFIG.copy()
    config['imu_segments'] = imu_segments
    config['label_filter_hz'] = label_filter_hz
    config['normalize'] = False  # Don't normalize - we want raw statistics
    
    # Auto-adjust input_size based on IMU segments
    if len(imu_segments) == 1 and imu_segments[0].lower() in ['femur', 'thigh']:
        input_size = 3
    else:
        input_size = 6
    config['input_size'] = input_size
    
    try:
        # Load data using LoadData (this handles all preprocessing including filtering)
        dataset = LoadData(
            root=data_root,
            partitions=subjects,
            conditions=conditions,
            window_size=config['window_size'],
            data_type="statistics_computation",
            dataset_proportion=1.0,
            input_mean=None,  # Don't use any normalization
            input_std=None,
            label_mean=None,
            label_std=None,
            normalize=False,  # Compute statistics on raw data
            imu_segments=imu_segments,
            augment=False,
            label_filter_hz=label_filter_hz,
            dataset_type=dataset_type,
            subject_info_df=None,
            param_mean=None,
            param_std=None,
        )
        
        # Get statistics (already computed by LoadData)
        stats = {
            'input_mean': dataset.input_mean.copy(),
            'input_std': dataset.input_std.copy(),
            'label_mean': dataset.label_mean.copy(),
            'label_std': dataset.label_std.copy(),
            'num_samples': len(dataset.input),
            'input_shape': dataset.input.shape,
            'label_shape': dataset.label.shape,
        }
        
        # Additional statistics
        stats['input_min'] = np.min(dataset.input, axis=0)
        stats['input_max'] = np.max(dataset.input, axis=0)
        stats['label_min'] = np.min(dataset.label, axis=0)
        stats['label_max'] = np.max(dataset.label, axis=0)
        stats['input_range'] = stats['input_max'] - stats['input_min']
        stats['label_range'] = stats['label_max'] - stats['label_min']
        
        # Store actual data for visualization (sample if too large to avoid memory issues)
        max_samples_for_viz = 100000
        if len(dataset.input) > max_samples_for_viz:
            sample_indices = np.random.choice(len(dataset.input), max_samples_for_viz, replace=False)
            stats['input_data'] = dataset.input[sample_indices].copy()
            stats['label_data'] = dataset.label[sample_indices].copy()
            print(f"  Sampled {max_samples_for_viz:,} samples from {len(dataset.input):,} total for visualization")
        else:
            stats['input_data'] = dataset.input.copy()
            stats['label_data'] = dataset.label.copy()
        
        print(f"\n✓ Statistics computed successfully")
        print(f"  Number of samples: {stats['num_samples']:,}")
        print(f"  Input shape: {stats['input_shape']}")
        print(f"  Label shape: {stats['label_shape']}")
        
        return stats
        
    except Exception as e:
        print(f"Error computing statistics: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_statistics(stats: dict, dataset_name: str):
    """Print statistics in a formatted way."""
    print(f"\n{'='*80}")
    print(f"{dataset_name} Statistics")
    print(f"{'='*80}")
    
    if stats is None:
        print("No statistics available")
        return
    
    print(f"\nDataset Overview:")
    print(f"  Total samples: {stats['num_samples']:,}")
    print(f"  Input shape: {stats['input_shape']}")
    print(f"  Label shape: {stats['label_shape']}")
    
    print(f"\nInput Statistics (IMU Gyroscope):")
    input_dim = stats['input_mean'].shape[0]
    for i in range(input_dim):
        print(f"  Channel {i+1}:")
        print(f"    Mean: {stats['input_mean'][i]:8.6f}")
        print(f"    Std:  {stats['input_std'][i]:8.6f}")
        print(f"    Min:  {stats['input_min'][i]:8.6f}")
        print(f"    Max:  {stats['input_max'][i]:8.6f}")
        print(f"    Range: {stats['input_range'][i]:8.6f}")
    
    print(f"\nLabel Statistics (Hip Flexion Moment):")
    label_dim = stats['label_mean'].shape[0]
    for i in range(label_dim):
        print(f"  Channel {i+1}:")
        print(f"    Mean: {stats['label_mean'][i]:8.6f} N-m/kg")
        print(f"    Std:  {stats['label_std'][i]:8.6f} N-m/kg")
        print(f"    Min:  {stats['label_min'][i]:8.6f} N-m/kg")
        print(f"    Max:  {stats['label_max'][i]:8.6f} N-m/kg")
        print(f"    Range: {stats['label_range'][i]:8.6f} N-m/kg")


def compare_statistics(stats1: dict, name1: str, stats2: dict, name2: str):
    """Compare two sets of statistics."""
    print(f"\n{'='*80}")
    print(f"Comparison: {name1} vs {name2}")
    print(f"{'='*80}")
    
    if stats1 is None or stats2 is None:
        print("Cannot compare: one or both statistics are missing")
        return
    
    print(f"\nSample Count Comparison:")
    print(f"  {name1}: {stats1['num_samples']:,} samples")
    print(f"  {name2}: {stats2['num_samples']:,} samples")
    ratio = stats1['num_samples'] / stats2['num_samples'] if stats2['num_samples'] > 0 else 0
    print(f"  Ratio ({name1}/{name2}): {ratio:.3f}x")
    
    # Input statistics comparison
    input_dim = min(stats1['input_mean'].shape[0], stats2['input_mean'].shape[0])
    print(f"\nInput Statistics Comparison (IMU Gyroscope):")
    print(f"{'Channel':<10} {'Metric':<15} {name1:<25} {name2:<25} {'Difference':<15} {'Ratio':<10}")
    print("-" * 100)
    
    for i in range(input_dim):
        print(f"\nChannel {i+1}:")
        metrics = [
            ('Mean', stats1['input_mean'][i], stats2['input_mean'][i]),
            ('Std', stats1['input_std'][i], stats2['input_std'][i]),
            ('Min', stats1['input_min'][i], stats2['input_min'][i]),
            ('Max', stats1['input_max'][i], stats2['input_max'][i]),
            ('Range', stats1['input_range'][i], stats2['input_range'][i]),
        ]
        
        for metric, val1, val2 in metrics:
            diff = val1 - val2
            ratio_val = val1 / val2 if abs(val2) > 1e-10 else np.nan
            print(f"{'':<10} {metric:<15} {val1:>24.6f} {val2:>24.6f} {diff:>14.6f} {ratio_val:>9.3f}")
    
    # Label statistics comparison
    label_dim = min(stats1['label_mean'].shape[0], stats2['label_mean'].shape[0])
    print(f"\nLabel Statistics Comparison (Hip Flexion Moment):")
    print(f"{'Channel':<10} {'Metric':<15} {name1:<25} {name2:<25} {'Difference':<15} {'Ratio':<10}")
    print("-" * 100)
    
    for i in range(label_dim):
        print(f"\nChannel {i+1}:")
        metrics = [
            ('Mean', stats1['label_mean'][i], stats2['label_mean'][i]),
            ('Std', stats1['label_std'][i], stats2['label_std'][i]),
            ('Min', stats1['label_min'][i], stats2['label_min'][i]),
            ('Max', stats1['label_max'][i], stats2['label_max'][i]),
            ('Range', stats1['label_range'][i], stats2['label_range'][i]),
        ]
        
        for metric, val1, val2 in metrics:
            diff = val1 - val2
            ratio_val = val1 / val2 if abs(val2) > 1e-10 else np.nan
            print(f"{'':<10} {metric:<15} {val1:>24.6f} {val2:>24.6f} {diff:>14.6f} {ratio_val:>9.3f}")
    
    # Normalization compatibility check
    print(f"\n{'='*80}")
    print("Normalization Compatibility Analysis")
    print(f"{'='*80}")
    print("\nWhen training on one dataset and testing on another, the model expects")
    print("inputs normalized using the training dataset's statistics.")
    print("\nThis comparison shows how different the distributions are, which affects")
    print("how well normalization transfer will work.\n")
    
    # Check if using same normalization would be problematic
    mean_diff_input = np.abs(stats1['input_mean'] - stats2['input_mean']).max()
    std_diff_input = np.abs(stats1['input_std'] - stats2['input_std']).max()
    mean_diff_label = np.abs(stats1['label_mean'] - stats2['label_mean']).max()
    std_diff_label = np.abs(stats1['label_std'] - stats2['label_std']).max()
    
    print(f"Input Mean Difference (max): {mean_diff_input:.6f}")
    print(f"Input Std Difference (max):  {std_diff_input:.6f}")
    print(f"Label Mean Difference (max): {mean_diff_label:.6f} N-m/kg")
    print(f"Label Std Difference (max):  {std_diff_label:.6f} N-m/kg")
    
    # Relative differences
    mean_rel_diff_input = (mean_diff_input / (np.abs(stats1['input_mean']).max() + 1e-10)) * 100
    std_rel_diff_input = (std_diff_input / (stats1['input_std'].max() + 1e-10)) * 100
    mean_rel_diff_label = (mean_diff_label / (np.abs(stats1['label_mean']).max() + 1e-10)) * 100
    std_rel_diff_label = (std_diff_label / (stats1['label_std'].max() + 1e-10)) * 100
    
    print(f"\nRelative Differences (as % of {name1} max values):")
    print(f"  Input Mean: {mean_rel_diff_input:.2f}%")
    print(f"  Input Std:  {std_rel_diff_input:.2f}%")
    print(f"  Label Mean: {mean_rel_diff_label:.2f}%")
    print(f"  Label Std:  {std_rel_diff_label:.2f}%")


def create_visualizations(
    stats1: dict, name1: str,
    stats2: dict, name2: str,
    output_dir: str
):
    """Create comprehensive visualizations comparing two datasets."""
    if stats1 is None or stats2 is None:
        print("Cannot create visualizations: one or both statistics are missing")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Creating visualizations...")
    print(f"{'='*80}")
    
    # 1. Comparison bar charts for mean and std
    create_statistics_comparison_plots(stats1, name1, stats2, name2, output_dir)
    
    # 2. Distribution overlays
    create_distribution_plots(stats1, name1, stats2, name2, output_dir)
    
    # 3. Box plots
    create_box_plots(stats1, name1, stats2, name2, output_dir)
    
    # 4. Summary comparison plot
    create_summary_comparison(stats1, name1, stats2, name2, output_dir)
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


def create_statistics_comparison_plots(stats1: dict, name1: str, stats2: dict, name2: str, output_dir: str):
    """Create bar charts comparing mean and std for inputs and labels."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    input_dim = min(stats1['input_mean'].shape[0], stats2['input_mean'].shape[0])
    label_dim = min(stats1['label_mean'].shape[0], stats2['label_mean'].shape[0])
    
    channels_input = [f'Ch{i+1}' for i in range(input_dim)]
    channels_label = [f'Ch{i+1}' for i in range(label_dim)]
    
    x_pos_input = np.arange(len(channels_input))
    x_pos_label = np.arange(len(channels_label))
    width = 0.35
    
    # Input Mean Comparison
    ax = axes[0, 0]
    bars1 = ax.bar(x_pos_input - width/2, stats1['input_mean'], width, 
                   label=name1, alpha=0.8, color='#3498db')
    bars2 = ax.bar(x_pos_input + width/2, stats2['input_mean'], width, 
                   label=name2, alpha=0.8, color='#e74c3c')
    ax.set_xlabel('Input Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Value', fontsize=12, fontweight='bold')
    ax.set_title('Input Mean Comparison (IMU Gyroscope)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos_input)
    ax.set_xticklabels(channels_input)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Input Std Comparison
    ax = axes[0, 1]
    bars1 = ax.bar(x_pos_input - width/2, stats1['input_std'], width, 
                   label=name1, alpha=0.8, color='#3498db')
    bars2 = ax.bar(x_pos_input + width/2, stats2['input_std'], width, 
                   label=name2, alpha=0.8, color='#e74c3c')
    ax.set_xlabel('Input Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax.set_title('Input Std Comparison (IMU Gyroscope)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos_input)
    ax.set_xticklabels(channels_input)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Label Mean Comparison
    ax = axes[1, 0]
    bars1 = ax.bar(x_pos_label - width/2, stats1['label_mean'], width, 
                   label=name1, alpha=0.8, color='#3498db')
    bars2 = ax.bar(x_pos_label + width/2, stats2['label_mean'], width, 
                   label=name2, alpha=0.8, color='#e74c3c')
    ax.set_xlabel('Label Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Value (N-m/kg)', fontsize=12, fontweight='bold')
    ax.set_title('Label Mean Comparison (Hip Flexion Moment)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos_label)
    ax.set_xticklabels(channels_label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Label Std Comparison
    ax = axes[1, 1]
    bars1 = ax.bar(x_pos_label - width/2, stats1['label_std'], width, 
                   label=name1, alpha=0.8, color='#3498db')
    bars2 = ax.bar(x_pos_label + width/2, stats2['label_std'], width, 
                   label=name2, alpha=0.8, color='#e74c3c')
    ax.set_xlabel('Label Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standard Deviation (N-m/kg)', fontsize=12, fontweight='bold')
    ax.set_title('Label Std Comparison (Hip Flexion Moment)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos_label)
    ax.set_xticklabels(channels_label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: statistics_comparison.png")


def create_distribution_plots(stats1: dict, name1: str, stats2: dict, name2: str, output_dir: str):
    """Create distribution overlay plots using histograms."""
    input_dim = min(stats1['input_mean'].shape[0], stats2['input_mean'].shape[0])
    label_dim = min(stats1['label_mean'].shape[0], stats2['label_mean'].shape[0])
    
    # Input distributions
    fig, axes = plt.subplots(input_dim, 1, figsize=(12, 4 * input_dim))
    if input_dim == 1:
        axes = [axes]
    
    for i in range(input_dim):
        ax = axes[i]
        data1 = stats1['input_data'][:, i]
        data2 = stats2['input_data'][:, i]
        
        ax.hist(data1, bins=50, alpha=0.6, label=name1, color='#3498db', density=True)
        ax.hist(data2, bins=50, alpha=0.6, label=name2, color='#e74c3c', density=True)
        ax.axvline(stats1['input_mean'][i], color='#2980b9', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(stats2['input_mean'][i], color='#c0392b', linestyle='--', linewidth=2, alpha=0.8)
        ax.set_xlabel(f'Input Channel {i+1} Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'Input Channel {i+1} Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'input_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: input_distributions.png")
    
    # Label distributions
    fig, axes = plt.subplots(label_dim, 1, figsize=(12, 6 * label_dim))
    if label_dim == 1:
        axes = [axes]
    
    for i in range(label_dim):
        ax = axes[i]
        data1 = stats1['label_data'][:, i]
        data2 = stats2['label_data'][:, i]
        
        ax.hist(data1, bins=50, alpha=0.6, label=name1, color='#3498db', density=True)
        ax.hist(data2, bins=50, alpha=0.6, label=name2, color='#e74c3c', density=True)
        ax.axvline(stats1['label_mean'][i], color='#2980b9', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(stats2['label_mean'][i], color='#c0392b', linestyle='--', linewidth=2, alpha=0.8)
        ax.set_xlabel(f'Label Channel {i+1} Value (N-m/kg)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'Label Channel {i+1} Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: label_distributions.png")


def create_box_plots(stats1: dict, name1: str, stats2: dict, name2: str, output_dir: str):
    """Create box plots comparing ranges."""
    input_dim = min(stats1['input_mean'].shape[0], stats2['input_mean'].shape[0])
    label_dim = min(stats1['label_mean'].shape[0], stats2['label_mean'].shape[0])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Input box plots
    ax = axes[0]
    data_to_plot = []
    labels_plot = []
    for i in range(input_dim):
        data_to_plot.append(stats1['input_data'][:, i])
        data_to_plot.append(stats2['input_data'][:, i])
        labels_plot.append(f'{name1}\nCh{i+1}')
        labels_plot.append(f'{name2}\nCh{i+1}')
    
    bp = ax.boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
    colors = ['#3498db', '#e74c3c'] * input_dim
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Input Data Box Plots', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Label box plots
    ax = axes[1]
    data_to_plot = []
    labels_plot = []
    for i in range(label_dim):
        data_to_plot.append(stats1['label_data'][:, i])
        data_to_plot.append(stats2['label_data'][:, i])
        labels_plot.append(f'{name1}\nCh{i+1}')
        labels_plot.append(f'{name2}\nCh{i+1}')
    
    bp = ax.boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
    colors = ['#3498db', '#e74c3c'] * label_dim
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Value (N-m/kg)', fontsize=12, fontweight='bold')
    ax.set_title('Label Data Box Plots', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'box_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: box_plots.png")


def create_summary_comparison(stats1: dict, name1: str, stats2: dict, name2: str, output_dir: str):
    """Create a comprehensive summary comparison plot."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Sample count comparison
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar([name1, name2], [stats1['num_samples'], stats2['num_samples']], 
                   color=['#3498db', '#e74c3c'], alpha=0.8)
    ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax.set_title('Sample Count Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=10)
    
    # 2. Input mean differences
    ax = fig.add_subplot(gs[0, 1])
    input_dim = min(stats1['input_mean'].shape[0], stats2['input_mean'].shape[0])
    mean_diff = np.abs(stats1['input_mean'][:input_dim] - stats2['input_mean'][:input_dim])
    channels = [f'Ch{i+1}' for i in range(input_dim)]
    bars = ax.bar(channels, mean_diff, color='#9b59b6', alpha=0.8)
    ax.set_ylabel('Absolute Difference', fontsize=11, fontweight='bold')
    ax.set_title('Input Mean Differences', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Input std differences
    ax = fig.add_subplot(gs[0, 2])
    std_diff = np.abs(stats1['input_std'][:input_dim] - stats2['input_std'][:input_dim])
    bars = ax.bar(channels, std_diff, color='#16a085', alpha=0.8)
    ax.set_ylabel('Absolute Difference', fontsize=11, fontweight='bold')
    ax.set_title('Input Std Differences', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Label mean differences
    ax = fig.add_subplot(gs[1, 1])
    label_dim = min(stats1['label_mean'].shape[0], stats2['label_mean'].shape[0])
    mean_diff = np.abs(stats1['label_mean'][:label_dim] - stats2['label_mean'][:label_dim])
    channels = [f'Ch{i+1}' for i in range(label_dim)]
    bars = ax.bar(channels, mean_diff, color='#e67e22', alpha=0.8)
    ax.set_ylabel('Absolute Difference (N-m/kg)', fontsize=11, fontweight='bold')
    ax.set_title('Label Mean Differences', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Label std differences
    ax = fig.add_subplot(gs[1, 2])
    std_diff = np.abs(stats1['label_std'][:label_dim] - stats2['label_std'][:label_dim])
    bars = ax.bar(channels, std_diff, color='#27ae60', alpha=0.8)
    ax.set_ylabel('Absolute Difference (N-m/kg)', fontsize=11, fontweight='bold')
    ax.set_title('Label Std Differences', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Input range comparison
    ax = fig.add_subplot(gs[2, 0])
    ranges1 = stats1['input_range'][:input_dim]
    ranges2 = stats2['input_range'][:input_dim]
    x = np.arange(input_dim)
    width = 0.35
    ax.bar(x - width/2, ranges1, width, label=name1, alpha=0.8, color='#3498db')
    ax.bar(x + width/2, ranges2, width, label=name2, alpha=0.8, color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Ch{i+1}' for i in range(input_dim)])
    ax.set_ylabel('Range', fontsize=11, fontweight='bold')
    ax.set_title('Input Range Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 7. Label range comparison
    ax = fig.add_subplot(gs[2, 1])
    ranges1 = stats1['label_range'][:label_dim]
    ranges2 = stats2['label_range'][:label_dim]
    x = np.arange(label_dim)
    ax.bar(x - width/2, ranges1, width, label=name1, alpha=0.8, color='#3498db')
    ax.bar(x + width/2, ranges2, width, label=name2, alpha=0.8, color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Ch{i+1}' for i in range(label_dim)])
    ax.set_ylabel('Range (N-m/kg)', fontsize=11, fontweight='bold')
    ax.set_title('Label Range Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 8. Summary text
    ax = fig.add_subplot(gs[1, 0])
    ax.axis('off')
    mean_diff_input = np.abs(stats1['input_mean'] - stats2['input_mean']).max()
    std_diff_input = np.abs(stats1['input_std'] - stats2['input_std']).max()
    mean_diff_label = np.abs(stats1['label_mean'] - stats2['label_mean']).max()
    std_diff_label = np.abs(stats1['label_std'] - stats2['label_std']).max()
    
    summary_text = f"""
    Dataset Comparison Summary
    
    Sample Counts:
      {name1}: {stats1['num_samples']:,}
      {name2}: {stats2['num_samples']:,}
      Ratio: {stats1['num_samples']/stats2['num_samples']:.2f}x
    
    Maximum Differences:
      Input Mean: {mean_diff_input:.6f}
      Input Std:  {std_diff_input:.6f}
      Label Mean: {mean_diff_label:.6f} N-m/kg
      Label Std:  {std_diff_label:.6f} N-m/kg
    
    Normalization Transfer Impact:
      Large differences may indicate that
      using one dataset's statistics on
      the other could cause distribution
      shift issues.
    """
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 9. Scatter plot of means
    ax = fig.add_subplot(gs[2, 2])
    ax.scatter(stats1['input_mean'], stats2['input_mean'], alpha=0.7, s=100, color='#3498db')
    min_val = min(stats1['input_mean'].min(), stats2['input_mean'].min())
    max_val = max(stats1['input_mean'].max(), stats2['input_mean'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='y=x')
    ax.set_xlabel(f'{name1} Input Mean', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{name2} Input Mean', fontsize=11, fontweight='bold')
    ax.set_title('Input Mean Correlation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Comprehensive Statistics Comparison: {name1} vs {name2}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(os.path.join(output_dir, 'summary_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: summary_comparison.png")


def save_statistics(stats: dict, dataset_name: str, output_dir: str):
    """Save statistics to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy arrays
    safe_name = dataset_name.replace(' ', '_').replace('/', '_')
    np.save(os.path.join(output_dir, f'{safe_name}_input_mean.npy'), stats['input_mean'])
    np.save(os.path.join(output_dir, f'{safe_name}_input_std.npy'), stats['input_std'])
    np.save(os.path.join(output_dir, f'{safe_name}_label_mean.npy'), stats['label_mean'])
    np.save(os.path.join(output_dir, f'{safe_name}_label_std.npy'), stats['label_std'])
    
    # Save as CSV for easier viewing
    input_df = pd.DataFrame({
        'channel': [f'ch_{i+1}' for i in range(len(stats['input_mean']))],
        'mean': stats['input_mean'],
        'std': stats['input_std'],
        'min': stats['input_min'],
        'max': stats['input_max'],
        'range': stats['input_range'],
    })
    input_df.to_csv(os.path.join(output_dir, f'{safe_name}_input_stats.csv'), index=False)
    
    label_df = pd.DataFrame({
        'channel': [f'ch_{i+1}' for i in range(len(stats['label_mean']))],
        'mean': stats['label_mean'],
        'std': stats['label_std'],
        'min': stats['label_min'],
        'max': stats['label_max'],
        'range': stats['label_range'],
    })
    label_df.to_csv(os.path.join(output_dir, f'{safe_name}_label_stats.csv'), index=False)
    
    print(f"\n✓ Statistics saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Compare statistics between Canonical_Camargo and Canonical_MetaMobility datasets'
    )
    parser.add_argument(
        '--camargo_root',
        type=str,
        default='/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Canonical_Camargo',
        help='Path to Canonical_Camargo dataset'
    )
    parser.add_argument(
        '--memo_root',
        type=str,
        default='/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Canonical_MetaMobility',
        help='Path to Canonical_MetaMobility dataset'
    )
    parser.add_argument(
        '--camargo_subjects',
        nargs='+',
        default=None,
        help='Specific subjects to include from Camargo (default: all)'
    )
    parser.add_argument(
        '--memo_subjects',
        nargs='+',
        default=None,
        help='Specific subjects to include from MetaMobility (default: all)'
    )
    parser.add_argument(
        '--camargo_conditions',
        nargs='+',
        default=['treadmill'],
        help='Specific conditions to include for Camargo dataset (default: treadmill)'
    )
    parser.add_argument(
        '--memo_conditions',
        nargs='+',
        default=['0p2mps', '0p4mps', '0p6mps', '0p8mps', '1p0mps', '1p2mps', '1p4mps', 'transient_15sec', 'transient_30sec'],
        help='Specific conditions to include for MetaMobility dataset (default: 0p2mps, 0p4mps, 0p6mps, 0p8mps, 1p0mps, 1p2mps, 1p4mps, transient_15sec, transient_30sec)'
    )
    parser.add_argument(
        '--use_all_conditions',
        action='store_true',
        help='Use all available conditions instead of defaults (overrides --camargo_conditions and --memo_conditions)'
    )
    parser.add_argument(
        '--imu_segments',
        nargs='+',
        default=['femur'],
        help='IMU segments to use (default: femur)'
    )
    parser.add_argument(
        '--label_filter_hz',
        type=float,
        default=6.0,
        help='Low-pass filter frequency for labels (default: 6.0)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./statistics_comparison',
        help='Directory to save statistics files'
    )
    
    args = parser.parse_args()
    
    # Detect dataset types
    camargo_type = detect_dataset_type(args.camargo_root)
    memo_type = detect_dataset_type(args.memo_root)
    
    print("Dataset Statistics Comparison Tool")
    print("=" * 80)
    print(f"Camargo dataset: {args.camargo_root} (type: {camargo_type})")
    print(f"MetaMobility dataset: {args.memo_root} (type: {memo_type})")
    print(f"IMU segments: {args.imu_segments}")
    print(f"Label filter: {args.label_filter_hz} Hz")
    
    # Determine conditions to use for each dataset
    if args.use_all_conditions:
        camargo_conditions = None  # None means use all available
        memo_conditions = None
        print("\nUsing all available conditions for both datasets")
    else:
        camargo_conditions = args.camargo_conditions
        memo_conditions = args.memo_conditions
        print(f"\nCamargo conditions: {camargo_conditions}")
        print(f"MetaMobility conditions: {memo_conditions}")
    
    # Compute statistics for Camargo
    camargo_stats = compute_dataset_statistics(
        data_root=args.camargo_root,
        subjects=args.camargo_subjects,
        conditions=camargo_conditions,
        imu_segments=args.imu_segments,
        label_filter_hz=args.label_filter_hz,
        dataset_type=camargo_type,
    )
    
    # Compute statistics for MetaMobility
    memo_stats = compute_dataset_statistics(
        data_root=args.memo_root,
        subjects=args.memo_subjects,
        conditions=memo_conditions,
        imu_segments=args.imu_segments,
        label_filter_hz=args.label_filter_hz,
        dataset_type=memo_type,
    )
    
    # Print individual statistics
    print_statistics(camargo_stats, "Canonical_Camargo")
    print_statistics(memo_stats, "Canonical_MetaMobility")
    
    # Compare statistics
    compare_statistics(
        camargo_stats, "Canonical_Camargo",
        memo_stats, "Canonical_MetaMobility"
    )
    
    # Create visualizations
    if camargo_stats is not None and memo_stats is not None:
        create_visualizations(
            camargo_stats, "Canonical_Camargo",
            memo_stats, "Canonical_MetaMobility",
            args.output_dir
        )
    
    # Save statistics
    if camargo_stats is not None:
        save_statistics(camargo_stats, "Canonical_Camargo", args.output_dir)
    if memo_stats is not None:
        save_statistics(memo_stats, "Canonical_MetaMobility", args.output_dir)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

