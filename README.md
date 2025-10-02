## Environment setup

Use the provided script to create/update and activate the Conda environment, and to install the local `processing/opensim` package.

```bash
# make the setup script executable (one-time)
chmod +x \
  "/Users/luorix/Desktop/MetaMobility Lab (CMU)/projects/F-2025/transfer-learning/setup_env.sh"

# create/update the env, activate it, and install processing/opensim
"/Users/luorix/Desktop/MetaMobility Lab (CMU)/projects/F-2025/transfer-learning/setup_env.sh"
```

Alternatively, you can manage the env manually:

```bash
# create once
conda env create -f \
  "/Users/luorix/Desktop/MetaMobility Lab (CMU)/projects/F-2025/transfer-learning/environment.yml"

# activate
conda activate transfer-learning

# install local opensim package
python -m pip install -e \
  "/Users/luorix/Desktop/MetaMobility Lab (CMU)/projects/F-2025/transfer-learning/processing/opensim"
```

If you see an activation error, initialize Conda for zsh and retry:

```bash
conda init zsh
exec zsh
```

## Run canonical_frame_converter

After activating the environment, run the converter as follows (example from console):

```bash
python processing/canonical_frame_converter.py \
  --model \
  "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Keaton_processed/AB01/osimxml/AB01.osim" \
  --motion \
  "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Keaton_processed/AB01/01_01_2020/levelground/opensim/normal_walk_1-8_06_01/walking_motion_states.sto" \
  --imu \
  "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Keaton_processed/AB01/01_01_2020/levelground/imu/normal_walk_1-8_06_01.csv" \
  --max-frames 4000 \
  --visualize \
  --unilateral \
  --unit deg \
  --out canonical_test
```

## Notes
- The Conda env is defined in `environment.yml` (includes PyTorch for macOS CPU/MPS).
- The setup script installs `processing/opensim` in editable mode so local changes are picked up.
- To remove/recreate the env:

```bash
conda env remove -n transfer-learning
conda env create -f \
  "/Users/luorix/Desktop/MetaMobility Lab (CMU)/projects/F-2025/transfer-learning/environment.yml"
```

## Batch dataset reformat

Convert a Keaton-style dataset to the MeMo-style layout. This does not hardcode any paths; pass them as arguments.

```bash
conda activate transfer-learning
python processing/batch_reformat.py \
  --input-root "/path/to/Keaton_processed" \
  --output-root "/path/to/MeMo_processed" \
  --conditions levelground,treadmill \
  --canonical \
  --unilateral \
  --unit deg \
  --max-frames 4000
```

- **Output layout**: `<output-root>/<Subject>/<Condition>/<Trial>/{Input/imu_data.csv, Label/joint_moment.csv}`
- **Labels**: pulled from `<Condition>/id/*.csv` and matched by trial name.
- **Canonical option**: when `--canonical` is set and model/motion files exist, IMU data are converted to canonical frames; otherwise the raw IMU CSV is copied.

## Training and Testing

### Setup Wandb (Optional but Recommended)

```bash
# Setup wandb for experiment tracking
python setup_wandb.py --entity your_username --project transfer-learning

# Test wandb integration
python setup_wandb.py --test
```

### Training

```bash
# Basic training
python src/train.py --data_root "/path/to/Canonical" --save_dir "./checkpoints"

# Training with custom parameters
python src/train.py \
  --train_subjects BT01 BT02 BT03 BT06 BT07 BT08 BT09 BT10 \
  --test_subjects BT11 BT12 BT13 BT14 BT15 \
  --conditions levelground \
  --epochs 30 \
  --batch_size 32 \
  --learning_rate 5e-6 \
  --wandb_name "experiment_1" \
  --wandb_tags baseline tcn
```

### Testing

```bash
python src/test.py \
  --model_path "./checkpoints/tcn_joint_moment_prediction.pt" \
  --data_root "/path/to/Canonical" \
  --save_dir "./checkpoints" \
  --test_subjects BT11 BT12 BT13 BT14 BT15
```

### Running Experiments

```bash
# Run baseline experiments
python run_experiment.py --experiment_type baseline

# Run hyperparameter sweep
python run_experiment.py --experiment_type sweep

# Run all experiments
python run_experiment.py --experiment_type all
```
