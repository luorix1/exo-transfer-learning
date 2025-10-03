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

### 2. Batch Dataset Reformatting

Convert datasets to the standardized Canonical format:

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

```bash
python src/test.py \
  --model_path "./checkpoints/tcn_joint_moment_best.pt" \
  --data_root "/path/to/Canonical" \
  --save_dir "./checkpoints" \
  --test_subjects BT11 BT12 BT13 BT14 BT15 \
  --conditions levelground
```

### Test Output

The test script generates:
- `evaluation_scatter.png`: Scatter plot of predictions vs ground truth
- `timeseries_{subject}_{condition}_{trial}.png`: Time series plots for sample trials (up to 3)
- `evaluation_results.csv`: Detailed prediction results
- Console output: RMSE, MAE, and sample count

### Test Arguments

- `--model_path`: Path to trained model checkpoint (required)
- `--data_root`: Path to Canonical dataset
- `--save_dir`: Directory containing normalization parameters
- `--test_subjects`: List of test subject IDs
- `--conditions`: Conditions to test on
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
│   └── batch_reformat.py     # Dataset reformatting utility
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
