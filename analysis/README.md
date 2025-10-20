# IMU Orientation Analysis

This module provides tools for optimizing IMU orientation using OpenSim simulations and real IMU data from the Final dataset structure.

## Overview

The IMU orientation optimization process:
1. **Generates simulated IMU data** from OpenSim models and motion files
2. **Loads real IMU data** from CSV files in the Final dataset structure
3. **Aligns time series** between simulated and real data
4. **Optimizes orientation** using Kabsch algorithm (SVD) to find optimal rotation matrix
5. **Creates visualizations** and saves results

## Files

- `imu_orientation_optimization.py` - Core optimization algorithm for single trials
- `batch_imu_optimization.py` - Batch processing across multiple subjects/trials
- `run_imu_analysis.py` - Simple runner script with easy commands
- `requirements.txt` - Python dependencies

## Dataset Structure

Expected Final dataset structure:
```
<dataset_root>/
├── <subject>/
│   ├── opensim/
│   │   └── <subject>.osim          # OpenSim model
│   ├── <condition>/
│   │   └── <trial>/
│   │       ├── Input/
│   │       │   └── imu_data.csv    # Real IMU data
│   │       └── opensim/
│   │           └── motion.sto      # OpenSim motion file
```

## Usage

### Single Trial Analysis

```bash
# Basic single trial
python run_imu_analysis.py single \
    --dataset "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Molinaro_Phase1_Phase2" \
    --subject AB01 \
    --condition levelground \
    --trial LG_C0p0_S0p0_BT_1_10 \
    --output results/single_trial/

# With custom parameters
python run_imu_analysis.py single \
    --dataset "/path/to/Final/Molinaro_Phase1_Phase2" \
    --subject AB01 \
    --condition levelground \
    --trial LG_C0p0_S0p0_BT_1_10 \
    --output results/ \
    --segment femur_r \
    --max-frames 1000 \
    --gyro-in-degrees
```

### Batch Analysis

```bash
# Process all available subjects and conditions
python run_imu_analysis.py batch \
    --dataset "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Molinaro_Phase1_Phase2" \
    --output results/batch_analysis/

# Process specific subjects and conditions
python run_imu_analysis.py batch \
    --dataset "/path/to/Final/Molinaro_Phase1_Phase2" \
    --output results/batch/ \
    --subjects AB01,AB02,AB03 \
    --conditions levelground,ramp \
    --max-trials-per-subject 5
```

### Direct Script Usage

```bash
# Single trial (direct script)
python imu_orientation_optimization.py \
    --dataset-root "/path/to/Final/Molinaro_Phase1_Phase2" \
    --subject AB01 \
    --condition levelground \
    --trial LG_C0p0_S0p0_BT_1_10 \
    --output results/ \
    --segment femur_r \
    --max-frames 2000

# Batch processing (direct script)
python batch_imu_optimization.py \
    --dataset-root "/path/to/Final/Molinaro_Phase1_Phase2" \
    --output-root results/batch/ \
    --subjects AB01,AB02 \
    --conditions levelground,ramp \
    --segment femur_r \
    --max-frames 2000
```

## Output

Each analysis produces:

### Files
- `optimization_results.json` - Detailed optimization results
- `optimal_rotation_matrix.txt` - Optimal rotation matrix
- `optimization_axes.png` - 3D visualization of axis alignment
- `optimization_comparison.png` - Before/after comparison plots
- `optimization_summary.png` - Summary statistics

### Batch Results
- `optimization_summary.csv` - Summary table of all trials
- `batch_results.json` - Detailed results for all trials

## Parameters

### Required
- `--dataset-root` / `--dataset` - Path to Final dataset root
- `--subject` - Subject ID (e.g., AB01)
- `--condition` - Condition (e.g., levelground, ramp, stair)
- `--trial` - Trial name (e.g., LG_C0p0_S0p0_BT_1_10)
- `--output` - Output directory

### Optional
- `--segment` - Body segment for IMU attachment (default: femur_r)
- `--max-frames` - Maximum frames to process (default: 2000)
- `--gyro-in-degrees` - Use degrees for gyroscope data (default: radians)
- `--subjects` - Comma-separated subject list for batch processing
- `--conditions` - Comma-separated condition list for batch processing
- `--max-trials-per-subject` - Limit trials per subject in batch (default: 10)

## Algorithm

The optimization uses the **Kabsch algorithm** (SVD-based) to solve the orthogonal Procrustes problem:

```
min ||R*P - Q||_F subject to R^T*R = I
```

Where:
- P = simulated IMU data
- Q = real IMU data  
- R = optimal rotation matrix

This finds the optimal 3D rotation to align simulated and real IMU orientations.

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install opensim numpy pandas matplotlib scipy tqdm
```

## Examples

### Example 1: Single Trial Analysis
```bash
python run_imu_analysis.py single \
    --dataset "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Molinaro_Phase1_Phase2" \
    --subject AB01 \
    --condition levelground \
    --trial LG_C0p0_S0p0_BT_1_10 \
    --output results/AB01_levelground_LG_C0p0_S0p0_BT_1_10/
```

### Example 2: Batch Analysis with Limits
```bash
python run_imu_analysis.py batch \
    --dataset "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Molinaro_Phase1_Phase2" \
    --output results/batch_molinaro/ \
    --subjects AB01,AB02,AB03 \
    --conditions levelground,ramp \
    --max-trials-per-subject 3 \
    --max-frames 1000
```

### Example 3: Process Different Datasets
```bash
# Camargo dataset
python run_imu_analysis.py batch \
    --dataset "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Camargo" \
    --output results/camargo_batch/

# Keaton dataset  
python run_imu_analysis.py batch \
    --dataset "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Keaton" \
    --output results/keaton_batch/
```

## Troubleshooting

### Common Issues

1. **OpenSim not found**: Ensure OpenSim 4.5+ is installed and accessible
2. **Missing files**: Check that model, motion, and IMU files exist in expected locations
3. **Memory issues**: Reduce `--max-frames` for large datasets
4. **Timeout errors**: Increase timeout in batch processing for complex trials

### Debug Mode

For detailed debugging, run the individual scripts directly:
```bash
python imu_orientation_optimization.py --help
python batch_imu_optimization.py --help
```
