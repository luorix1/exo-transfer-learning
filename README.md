# Transfer Learning for Joint Moment Prediction

Deep learning-based joint moment prediction from IMU data using Temporal Convolutional Networks (TCN).

## Overview

This repository contains:
- **Data processing pipeline**: Convert raw IMU data to OpenSim canonical frames
- **Dataset reformatting**: Transform various dataset formats to a unified structure
- **TCN model**: Unilateral joint moment prediction from gyroscope data
- **Training & evaluation**: Scripts for model training, testing, and visualization

## Environment Setup

### Using the Setup Script (Recommended)

```bash
# Make the setup script executable (one-time)
chmod +x setup_env.sh

# Create/update the environment, activate it, and install processing/opensim
./setup_env.sh
```

### Manual Setup

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate transfer-learning

# Install local opensim package
python -m pip install -e processing/opensim
```

### Troubleshooting

If you see an activation error, initialize Conda for zsh:

```bash
conda init zsh
exec zsh
```

To remove/recreate the environment:

```bash
conda env remove -n transfer-learning
conda env create -f environment.yml
```

## Data Processing

### 1. Canonical Frame Conversion

Convert real IMU angular velocity data to OpenSim canonical (segment-aligned) frames:

```bash
python processing/canonical_frame_converter.py \
  --model "/path/to/model.osim" \
  --motion "/path/to/walking_motion_states.sto" \
  --imu "/path/to/real_imu.csv" \
  --output canonical_output \
  --max-frames 4000 \
  --visualize \
  --unilateral \
  --unit deg
```

**Options:**
- `--model`: Path to OpenSim model (.osim)
- `--motion`: Path to motion file (.sto with states)
- `--imu`: Path to real IMU CSV
- `--output`: Output directory
- `--segments`: Comma-separated segments (default: auto-detect)
- `--max-frames`: Maximum frames to process (default: 2000)
- `--visualize`: Plot alignment comparisons
- `--unilateral`: Assume all IMU columns are right side
- `--unit`: Unit of real IMU gyro (rad/deg, default: rad)

### 2. MeMo Dataset Processing

Process MeMo_processed dataset with coordinate frame transformation:

```bash
python processing/process_memo.py \
  --input-root "/path/to/MeMo_processed" \
  --output-root "/path/to/Canonical_MeMo" \
  --conditions 0mps,1p0mps \
  --subjects AB01_Jimin,AB02_Rajiv
```

**Coordinate Frame Transformation:**

MeMo IMU sensors use a different coordinate frame than OpenSim canonical:
- **MeMo Frame**: x=up, y=left, z=back
- **Canonical Frame**: x=forward, y=up, z=right

The script automatically transforms gyroscope data:
```
canonical_x (forward) = -memo_z (flip back to forward)
canonical_y (up)      =  memo_x (up stays up)
canonical_z (right)   = -memo_y (flip left to right)
```

**Options:**
- `--input-root`: Path to MeMo_processed directory
- `--output-root`: Output directory for Canonical format
- `--conditions`: Comma-separated conditions (e.g., `0mps,1p0mps,transient_15sec`)
- `--subjects`: Comma-separated subjects to process (default: all subjects)
- `--no-transform`: Skip coordinate frame transformation (keep original MeMo frame)

**What it does:**
1. Renames label files from `{subject}_{condition}_{trial}.csv` to `joint_moment.csv`
2. Standardizes all column names to lowercase with underscores
3. Transforms gyroscope data from MeMo frame to OpenSim canonical frame
4. Maintains the existing Subject/Condition/Trial/Input,Label structure

### 3. Batch Dataset Reformatting

Convert raw datasets (Camargo, Keaton, etc.) to the standardized Canonical format:

```bash
python processing/batch_reformat.py \
  --input-root "/path/to/raw_dataset" \
  --output-root "/path/to/Canonical" \
  --conditions levelground,treadmill \
  --canonical \
  --unilateral \
  --unit deg \
  --max-frames 4000 \
  --normalize-moment
```

**Options:**
- `--input-root`: Input dataset directory
- `--output-root`: Output directory for reformatted data
- `--conditions`: Comma-separated conditions to include
- `--canonical`: Apply canonical frame conversion (requires model/motion files)
- `--unilateral`: Treat as unilateral data (add _r suffixes)
- `--unit`: IMU gyro unit (rad/deg)
- `--max-frames`: Maximum frames for canonical conversion
- `--normalize-moment`: Normalize joint moments by bodyweight (if SubjectInfo.csv exists)

**Output Structure:**
```
Canonical/
├── Subject1/
│   ├── levelground/
│   │   ├── trial_01/
│   │   │   ├── Input/
│   │   │   │   └── imu_data.csv
│   │   │   └── Label/
│   │   │       └── joint_moment.csv
│   │   └── trial_02/
│   │       └── ...
│   └── treadmill/
│       └── ...
└── Subject2/
    └── ...
```

**Data Format:**
- **IMU columns**: `pelvis_gyro_{x,y,z}`, `{thigh,femur}_r_gyro_{x,y,z}`, etc.
- **Label columns**: `hip_flexion_r_moment`, `knee_angle_r_moment`, etc.
- All column names are lowercase with underscores

## Visualization

### Right hip flexion vs right thigh gyro

Plot the right hip flexion moment against the three axes of the right thigh gyroscope, synchronized by the `time` column (rows with any NaN are dropped before plotting).

Usage:

```bash
python scripts/plot_canonical_right_hip_vs_thigh.py \
  --data_root "/path/to/Canonical" \
  --subject AB06 \
  --condition treadmill \
  --trial treadmill_01_01
```

Examples:

```bash
# Canonical_Camargo example
python scripts/plot_canonical_right_hip_vs_thigh.py \
  --data_root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Canonical_Camargo" \
  --subject AB06 \
  --condition treadmill \
  --trial treadmill_01_01

# Canonical_Molinaro example
python scripts/plot_canonical_right_hip_vs_thigh.py \
  --data_root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Canonical_Molinaro" \
  --subject AB21 \
  --condition levelground \
  --trial levelground_0.0_1.0_01
```

The script saves a PNG in the current directory named like `plot_rhip_vs_rthigh_<subject>_<condition>_<trial>.png`.

## Model Training

### Basic Training

```bash
python src/train.py \
  --data_root "/path/to/Canonical" \
  --save_dir "./checkpoints" \
  --train_subjects AB06 AB07 \
  --test_subjects AB19 \
  --conditions levelground \
  --imu_segments femur \
  --epochs 30 \
  --batch_size 32 \
  --learning_rate 5e-6
```

### Training with Wandb

```bash
python src/train.py \
  --data_root "/path/to/Canonical" \
  --save_dir "./checkpoints" \
  --train_subjects BT01 BT02 BT03 BT06 BT07 BT08 BT09 BT10 \
  --test_subjects BT11 BT12 BT13 BT14 BT15 \
  --conditions levelground \
  --imu_segments pelvis femur \
  --epochs 30 \
  --wandb_project transfer-learning \
  --wandb_name "experiment_1" \
  --wandb_tags baseline tcn
```

## GMF-Based Pipeline

The generalized moment feature (GMF) workflow uses standalone training and evaluation entrypoints so it can develop independently from the original TCN scripts.

### Train the GMF Model

```bash
python src/train_gmf.py \
  --data_root /media/metamobility3/Samsung_T5/Canonical_Camargo \
  --train_subjects AB06 AB07 AB08 AB09 AB10 AB12 AB13 AB14 AB15 AB16 AB17 AB18 AB19 AB20 \
  --test_subjects AB23 AB24 \
  --conditions treadmill \
  --imu_segments femur pelvis \
  --epochs 10 \
  --save_dir ./20251011_Camargo_gmf \
  --gmf_loss_weight 1.0 \
  --decoder_loss_weight 0.05 \
  --no_wandb
```

The CLI mirrors the legacy `train.py` arguments:

* `--train_subjects`, `--test_subjects`, `--conditions`, `--imu_segments`, and `--epochs` behave the same as in the baseline training script.
* `--data_root` should point to a Canonical dataset that includes a `SubjectInfo.csv` file so the loader can pull body-mass and height metadata.
* `--save_dir` is where checkpoints, normalization statistics, and the resolved configuration are written (the directory is created automatically).
* `--gmf_loss_weight` and `--decoder_loss_weight` set the weights for the estimator–generator alignment loss (L1) and decoder reconstruction loss (L2). Leaving them unset falls back to the defaults of 1.0 for each; the example above matches the paper’s 1.0 / 0.05 split.
* Add `--no_wandb` to disable Weights & Biases logging when offline.

### Evaluate a GMF Checkpoint

```bash
python src/test_gmf.py \
  --data_root /media/metamobility3/Samsung_T5/Canonical_Camargo \
  --checkpoint_dir ./20251011_Camargo_gmf \
  --test_subjects AB23 AB24 \
  --conditions treadmill \
  --imu_segments femur pelvis
```

The evaluation entrypoint reuses the normalization statistics and subject-parameter scaling saved during training, so no additional preprocessing is required.

### Key Arguments

- `--data_root`: Path to Canonical dataset
- `--save_dir`: Directory to save model checkpoints and results
- `--train_subjects`: List of training subject IDs
- `--test_subjects`: List of test subject IDs
- `--conditions`: Conditions to use (levelground, treadmill, etc.)
- `--imu_segments`: IMU segments to use:
  - `femur` or `thigh`: Single thigh IMU (3 channels)
  - `pelvis femur`: Dual IMU - pelvis + femur (6 channels)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--window_size`: Temporal window size (default: 100)
- `--no_wandb`: Disable wandb logging
- `--augment`: Enable training-time augmentation (random small IMU rotations ±10°, Gaussian noise)
- `--label_filter_hz`: Low-pass cutoff (Hz) for zero-phase label filtering (default: 6.0)
- `--use_curriculum`: Toggle curriculum training (plumbing available)
- `--curriculum_epochs`: Number of initial curriculum epochs (0 disables)

### Output Files

The training script saves:
- `{save_dir}/config.json`: Experiment configuration
- `{save_dir}/tcn_joint_moment_{timestamp}.pt`: Best model
- `{save_dir}/tcn_joint_moment_{timestamp}_epoch_{N}.pt`: Per-epoch checkpoints
- `{save_dir}/input_mean.npy`, `input_std.npy`: Input normalization params
- `{save_dir}/label_mean.npy`, `label_std.npy`: Label normalization params
- `{save_dir}/prediction_epoch_{N}_joint_0.png`: Per-epoch prediction plots
- `{save_dir}/final_rmse_plot.png`: Training/validation RMSE over epochs

## Model Testing

### Basic Testing

```bash
python src/test.py \
  --model_path "./checkpoints/tcn_joint_moment_best.pt" \
  --data_root "/path/to/Canonical" \
  --save_dir "./checkpoints" \
  --test_subjects BT11 BT12 BT13 BT14 BT15 \
  --conditions levelground
```

### Cross-Dataset Testing

The `test.py` script is fully equipped for cross-dataset evaluation. You can test a model trained on one dataset on a completely different dataset, as long as the data format matches.

#### Example 1: Transfer Learning Evaluation

```bash
# Train on Dataset A (e.g., Camargo)
python src/train.py \
  --data_root "/path/to/Canonical_Camargo" \
  --save_dir "./model_camargo" \
  --train_subjects AB06 AB07 AB08 AB09 AB10 \
  --test_subjects AB11 AB12 \
  --conditions treadmill \
  --imu_segments pelvis femur

# Test on Dataset B (e.g., MoMo) - Transfer Learning
python src/test.py \
  --model_path "./model_camargo/tcn_joint_moment_20250103_123456.pt" \
  --data_root "/path/to/Canonical_MoMo" \
  --save_dir "./model_camargo" \
  --test_subjects Subject01 Subject02 Subject03 \
  --conditions levelground treadmill
```

#### Example 2: Cross-Condition Generalization

```bash
# Train on treadmill condition
python src/train.py \
  --data_root "/path/to/Canonical" \
  --save_dir "./model_treadmill" \
  --train_subjects BT01 BT02 BT03 \
  --test_subjects BT04 BT05 \
  --conditions treadmill \
  --imu_segments femur

# Test on stairs condition (generalization test)
python src/test.py \
  --model_path "./model_treadmill/tcn_joint_moment_20250103_123456.pt" \
  --data_root "/path/to/Canonical" \
  --save_dir "./model_treadmill" \
  --test_subjects BT06 BT07 BT08 \
  --conditions stairs levelground
```

#### Example 3: Different IMU Configuration

```bash
# Train with dual IMU (pelvis + femur)
python src/train.py \
  --data_root "/path/to/Canonical" \
  --save_dir "./model_dual_imu" \
  --imu_segments pelvis femur \
  --train_subjects Subject1 Subject2

# Test with same dual IMU on different subjects
python src/test.py \
  --model_path "./model_dual_imu/tcn_joint_moment_20250103_123456.pt" \
  --data_root "/path/to/Canonical" \
  --save_dir "./model_dual_imu" \
  --test_subjects Subject3 Subject4 \
  --imu_segments pelvis femur
```

#### Example 4: Single IMU Testing

```bash
# Train with single thigh IMU
python src/train.py \
  --data_root "/path/to/Canonical" \
  --save_dir "./model_single_imu" \
  --imu_segments femur \
  --train_subjects Subject1 Subject2

# Test with same single IMU configuration
python src/test.py \
  --model_path "./model_single_imu/tcn_joint_moment_20250103_123456.pt" \
  --data_root "/path/to/Canonical" \
  --save_dir "./model_single_imu" \
  --test_subjects Subject3 Subject4 \
  --imu_segments femur
```

### Cross-Dataset Requirements

For successful cross-dataset testing, ensure:

1. **Same data format**: Test data must be in Canonical format:
   ```
   Subject/Condition/Trial/
   ├── Input/imu_data.csv
   └── Label/joint_moment.csv
   ```

2. **Same IMU segments**: Test data must have the same sensors used during training:
   - If trained with `femur + pelvis`, test data needs both
   - If trained with just `femur`, test data needs just femur

3. **Same label columns**: Test data needs `hip_flexion_r_moment` or `hip_r_moment`

4. **Normalization files**: The `save_dir` must contain:
   - `input_mean.npy`, `input_std.npy`
   - `label_mean.npy`, `label_std.npy`
   - `config.json`
   - Model checkpoint (`.pt` file)

### Test Output

The test script generates:
- `evaluation_scatter.png`: Scatter plot of predictions vs ground truth with R² score
- `timeseries_{subject}_{condition}_{trial}.png`: Time series plots for sample trials (up to 3)
- `evaluation_results.csv`: Detailed prediction results
- Console output: RMSE, MAE, R² score, and sample count
 - Scale/offset diagnostics: mean(|GT|), mean(|Pred|), scale ratio, mean offset, min/max spans, linear fit a,b

### Test Arguments

- `--model_path`: Path to trained model checkpoint (required)
- `--data_root`: Path to Canonical dataset (can be different from training)
- `--save_dir`: Directory containing normalization parameters (from training)
- `--test_subjects`: List of test subject IDs (can be different from training)
- `--conditions`: Conditions to test on (can be different from training)
- `--imu_segments`: IMU segments (optional, reads from config.json if available)
- `--window_size`: Window size (optional, reads from config.json if available)

## Model Architecture

### TCN Configuration

The model uses a Temporal Convolutional Network with:
- **Input**: 3 or 6 gyroscope channels (depending on IMU configuration)
- **Output**: 1 hip flexion moment (unilateral model)
- **Architecture**:
  - 5 TCN layers with [80, 80, 80, 80, 80] channels
  - Kernel size: 5
  - Dilations: [1, 2, 4, 8, 16]
  - Dropout: 0.15
  - Window size: 95 frames (~1 second at 100 Hz)
- **Training**:
  - Loss: MSE on normalized data
  - Optimizer: Adam (lr=5e-6, weight_decay=1e-5)
  - Scheduler: ReduceLROnPlateau (factor=0.5, patience=1)
  - Validation split: 10% (or Leave-One-Subject-Out for multi-subject)

Configuration is defined in `src/config/hyperparameters.py`.

## Training Metrics

The system tracks:
- **Loss**: MSE on normalized data (used for optimization)
- **RMSE**: Root Mean Squared Error in N-m/kg (actual units)
- **Accuracy**: Percentage of predictions within 0.05 N-m/kg of ground truth

## Data Loading

The data loader:
- Loads IMU gyroscope data and joint moment labels
- Handles both unilateral (right-only) and bilateral data
- Filters NaN values per-side before windowing
- Stacks left and right data randomly for unilateral training
- Normalizes inputs and outputs using z-score normalization
- Creates sliding windows for temporal modeling
- Applies zero-phase Butterworth low-pass filter to labels at 6 Hz (Fs=100 Hz); cutoff configurable via `--label_filter_hz`
- Optional train-time augmentation (if `--augment`): per-sensor small rotations (±10°) on gyro triplets and Gaussian noise (std=0.01)

### Domain Check Utility

Compare label scale/range across two datasets (uses the same loader logic):

```bash
python scripts/domain_check.py \
  --dataset_a /path/to/Canonical_Camargo \
  --dataset_b /path/to/Canonical_MeMo \
  --name_a camargo --name_b memo \
  --imu_segments pelvis femur \
  --verbose --out 20251003_1/domain_check
```

Outputs per-trial CSV summaries and overall stats (min/max/span/mean/std/mean|.|) and a scale ratio.

## Environment Details

The conda environment (`environment.yml`) includes:
- Python 3.12
- PyTorch (CPU build with OpenBLAS, no MKL)
- NumPy, SciPy, Pandas, Matplotlib
- OpenSim Python bindings
- Weights & Biases for experiment tracking
- Scientific computing utilities

## Project Structure

```
transfer-learning/
├── processing/
│   ├── opensim/              # Local OpenSim utilities package
│   ├── canonical_frame_converter.py  # IMU canonical frame conversion
│   ├── batch_reformat.py     # Dataset reformatting utility (Camargo, Keaton, etc.)
│   └── process_memo.py       # MeMo dataset processor with frame transformation
├── src/
│   ├── config/
│   │   └── hyperparameters.py  # Model hyperparameters
│   ├── data/
│   │   └── dataloader.py     # Data loading and preprocessing
│   ├── model/
│   │   └── tcn.py            # TCN model architecture
│   ├── loss.py               # Loss functions
│   ├── trainer.py            # Training loop and utilities
│   ├── train.py              # Training script
│   └── test.py               # Testing script
├── environment.yml           # Conda environment specification
├── setup_env.sh              # Environment setup script
└── README.md                 # This file
```

## Notes

- The system is designed for macOS but can be adapted for Linux/Windows
- `num_workers=0` is used for data loading on macOS to avoid multiprocessing overhead
- Models are trained unilaterally but use both left and right data by stacking
- Sign flipping is applied to left-side gyroscope Y-axis for canonical frame consistency
- Experiment configurations are saved for reproducibility
