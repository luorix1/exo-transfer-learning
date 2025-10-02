# Wandb Integration Setup

## 🎯 **Complete Wandb Integration**

The transfer learning project now has comprehensive wandb integration for experiment tracking, model versioning, and result visualization.

## 📋 **What's Included**

### **1. Enhanced Training Script** (`src/train.py`)
- ✅ **Automatic run naming** with timestamps
- ✅ **Rich experiment metadata** (subjects, conditions, device info)
- ✅ **Configurable tags** for experiment organization
- ✅ **Robust wandb initialization** with error handling

### **2. Advanced Trainer** (`src/trainer.py`)
- ✅ **Comprehensive metrics logging** (loss, RMSE, accuracy, learning rate)
- ✅ **Model architecture tracking** with artifacts
- ✅ **Prediction visualization** with plots
- ✅ **Model versioning** with complete artifacts (model, normalization, config)

### **3. Experiment Management**
- ✅ **Setup script** (`setup_wandb.py`) for easy configuration
- ✅ **Experiment runner** (`run_experiment.py`) for batch experiments
- ✅ **Configuration file** (`wandb_config.yaml`) for project settings

## 🚀 **Quick Start**

### **1. Setup Wandb**
```bash
# Login and configure wandb
python setup_wandb.py --entity your_username --project transfer-learning

# Test integration
python setup_wandb.py --test
```

### **2. Run Training with Wandb**
```bash
# Basic training with automatic wandb setup
python src/train.py --data_root "/path/to/Canonical" --save_dir "./checkpoints"

# Advanced training with custom tags and naming
python src/train.py \
  --data_root "/path/to/Canonical" \
  --save_dir "./checkpoints" \
  --wandb_name "baseline_experiment" \
  --wandb_tags baseline tcn joint_moment \
  --epochs 30 \
  --batch_size 32
```

### **3. Run Batch Experiments**
```bash
# Run baseline experiments
python run_experiment.py --experiment_type baseline

# Run hyperparameter sweep
python run_experiment.py --experiment_type sweep

# Run all experiments
python run_experiment.py --experiment_type all
```

## 📊 **What Gets Tracked**

### **Metrics**
- Training/validation/test loss, RMSE, accuracy
- Learning rate scheduling
- Early stopping patience
- Best validation loss

### **Artifacts**
- **Model weights** (`.pt` files)
- **Normalization parameters** (mean/std `.npy` files)
- **Model architecture** (text file)
- **Training plots** (RMSE curves, predictions)
- **Configuration** (hyperparameters, data splits)

### **Metadata**
- Subject lists (train/test)
- Conditions used
- Device information
- PyTorch version
- Experiment tags and notes

## 🎨 **Wandb Dashboard Features**

### **Runs Table**
- Sortable by metrics (RMSE, accuracy, loss)
- Filterable by tags and parameters
- Quick comparison of experiments

### **Charts**
- Real-time training curves
- Hyperparameter correlation plots
- Model performance comparisons

### **Artifacts**
- Versioned model storage
- Easy model downloading
- Experiment reproducibility

## 🔧 **Configuration Options**

### **Command Line Arguments**
```bash
python src/train.py \
  --wandb_project "my-project" \
  --wandb_entity "my-team" \
  --wandb_name "custom_experiment" \
  --wandb_tags tag1 tag2 tag3 \
  --no_wandb  # Disable wandb logging
```

### **Configuration File** (`wandb_config.yaml`)
```yaml
project: "transfer-learning"
entity: "your-username"
default_tags: ["tcn", "joint_moment", "imu"]
```

## 📈 **Experiment Types**

### **1. Baseline Experiments**
- Small, default, and large model architectures
- Standard hyperparameters
- Subject-independent evaluation

### **2. Hyperparameter Sweep**
- Learning rate: [1e-6, 5e-6, 1e-5, 5e-5]
- Batch size: [16, 32, 64]
- Window size: [50, 95, 150, 200]

### **3. Subject-Specific Experiments**
- Different train/test splits
- Cross-subject validation
- Subject-specific model evaluation

## 🎯 **Best Practices**

### **Experiment Naming**
```bash
# Use descriptive names
--wandb_name "tcn_baseline_20241201"
--wandb_name "sweep_lr_5e6_batch32"
--wandb_name "subject_split_1"
```

### **Tagging Strategy**
```bash
# Use consistent tags
--wandb_tags baseline tcn
--wandb_tags sweep hyperparameter
--wandb_tags subject_split validation
```

### **Model Versioning**
- Each experiment creates a versioned artifact
- Models are automatically saved with metadata
- Easy to download and reproduce results

## 🔍 **Troubleshooting**

### **Wandb Login Issues**
```bash
# Manual login
wandb login

# Check login status
wandb status
```

### **Permission Issues**
```bash
# Make scripts executable
chmod +x setup_wandb.py run_experiment.py
```

### **Memory Issues**
```bash
# Reduce batch size for large models
python src/train.py --batch_size 16
```

## 📚 **Next Steps**

1. **Setup wandb**: Run `python setup_wandb.py --test`
2. **Run baseline**: `python src/train.py` with your data
3. **Explore results**: Check your wandb dashboard
4. **Run experiments**: Use `run_experiment.py` for batch runs
5. **Compare models**: Use wandb's comparison features

The integration is now complete and ready for comprehensive experiment tracking! 🎉
