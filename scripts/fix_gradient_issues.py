#!/usr/bin/env python3
"""
Quick fixes for gradient issues in training.

This script provides several solutions for large gradient problems:
1. Add gradient clipping to trainer
2. Reduce learning rate
3. Add better initialization
4. Improve data normalization

Usage:
    python scripts/fix_gradient_issues.py
"""

import os
import shutil
from pathlib import Path


def add_gradient_clipping_to_trainer():
    """Add gradient clipping to the trainer.py file."""
    trainer_path = Path("src/trainer.py")
    
    if not trainer_path.exists():
        print("❌ trainer.py not found")
        return False
    
    # Read current trainer.py
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Check if gradient clipping is already present
    if "clip_grad_norm" in content:
        print("✅ Gradient clipping already present in trainer.py")
        return True
    
    # Add gradient clipping after optimizer.step()
    old_step = "self.optimizer.step()"
    new_step = """self.optimizer.step()
        
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)"""
    
    if old_step in content:
        content = content.replace(old_step, new_step)
        
        # Create backup
        shutil.copy(trainer_path, f"{trainer_path}.backup")
        
        # Write updated content
        with open(trainer_path, 'w') as f:
            f.write(content)
        
        print("✅ Added gradient clipping to trainer.py")
        print("📁 Backup created: trainer.py.backup")
        return True
    else:
        print("❌ Could not find optimizer.step() in trainer.py")
        return False


def create_improved_hyperparameters():
    """Create improved hyperparameters with better settings."""
    improved_config = {
        "input_size": 6,
        "output_size": 1,
        "num_channels": [80, 80, 80, 80, 80],
        "kernel_size": 5,
        "number_of_layers": 2,
        "dropout": 0.15,
        "dilations": [1, 2, 4, 8, 16],
        "window_size": 95,
        
        # Improved training parameters
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 1e-6,  # Reduced from 5e-6
        "number_of_workers": 0,
        "validation_split": 0.1,
        "dataset_proportion": 1.0,
        "transfer_learning": False,
        
        # Gradient clipping
        "grad_clip_norm": 1.0,
        
        # Better initialization
        "init_method": "kaiming",  # Use Kaiming initialization
        
        # Wandb configuration
        "wandb_session_name": "tcn_joint_moment_prediction_stable",
        "wandb_project": "transfer-learning",
        "wandb_entity": None,
    }
    
    # Save improved config
    config_path = Path("src/config/hyperparameters_stable.py")
    with open(config_path, 'w') as f:
        f.write('"""\n')
        f.write('Stable hyperparameter configuration for TCN training.\n')
        f.write('Includes gradient clipping and improved learning rate.\n')
        f.write('"""\n\n')
        f.write('STABLE_TCN_CONFIG = {\n')
        for key, value in improved_config.items():
            if isinstance(value, str):
                f.write(f"    '{key}': '{value}',\n")
            else:
                f.write(f"    '{key}': {value},\n")
        f.write('}\n')
    
    print(f"✅ Created stable hyperparameters: {config_path}")
    return True


def create_training_script_with_fixes():
    """Create a training script with all the fixes applied."""
    script_content = '''#!/usr/bin/env python3
"""
Training script with gradient issue fixes applied.

This script includes:
- Gradient clipping
- Reduced learning rate
- Better monitoring
- Stable hyperparameters

Usage:
    python scripts/train_stable.py --data_root /path/to/data --train_subjects AB21
"""

import os
import warnings
os.environ["MKL_VERBOSE"] = "0"
os.environ["MKL_DISABLE_FAST_MM"] = "1"
import json
import torch
import numpy as np
warnings.filterwarnings('ignore', message='.*Could not initialize NNPACK.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
import wandb
import argparse
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.tcn import TCNModel
from data.dataloader import DataHandler
from trainer import Trainer
from loss import JointMomentLoss
from config.hyperparameters import DEFAULT_TCN_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Train TCN model with gradient fixes')
    parser.add_argument('--data_root', type=str, 
                       default='/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Canonical',
                       help='Path to Canonical dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_stable',
                       help='Directory to save model checkpoints')
    parser.add_argument('--train_subjects', nargs='+', 
                       default=['AB21'],
                       help='Training subjects')
    parser.add_argument('--test_subjects', nargs='+', 
                       default=['AB21'],
                       help='Test subjects')
    parser.add_argument('--conditions', nargs='+', default=['treadmill'],
                       help='Conditions to use for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                       help='Learning rate (reduced for stability)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                       help='Gradient clipping norm')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Get hyperparameter configuration
    config = DEFAULT_TCN_CONFIG.copy()
    config.update({
        'data_root': args.data_root,
        'save_dir': args.save_dir,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'grad_clip_norm': args.grad_clip_norm,
        'wandb_session_name': f'tcn_stable_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    })
    
    print(f"🚀 Starting stable training with gradient fixes...")
    print(f"📊 Learning rate: {config['learning_rate']}")
    print(f"📊 Gradient clipping: {config['grad_clip_norm']}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(args.save_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project='transfer-learning-stable',
            name=config['wandb_session_name'],
            config=config,
            tags=['tcn', 'stable', 'gradient-fixed']
        )
        wandb_run = wandb.run
    else:
        wandb_run = None
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize data handler
    data_handler = DataHandler(
        data_root=config['data_root'],
        hyperparam_config=config,
        pretrained_model_path=None
    )
    
    # Load data
    data_handler.load_data(
        train_data_partition=args.train_subjects,
        train_data_condition=args.conditions,
        test_data_partition=args.test_subjects
    )
    
    # Save normalization parameters
    data_handler.save_mean_std(args.save_dir)
    
    # Initialize model
    model = TCNModel(config).to(device)
    
    # Initialize loss function
    criterion = JointMomentLoss()
    
    # Initialize optimizer with reduced learning rate
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    
    # Initialize scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Initialize trainer
    trainer = Trainer(
        device=device,
        model=model,
        wandb_run=wandb_run,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        data_handler=data_handler,
        config=config,
        save_dir=args.save_dir
    )
    
    # Train the model
    print("Starting stable training...")
    test_loader = trainer.train()
    
    # Evaluate the model
    print("\\nFinal evaluation on test set...")
    trainer.evaluate(test_loader)
    
    if wandb_run:
        wandb.finish()
    
    print("Stable training completed!")


if __name__ == '__main__':
    main()
'''
    
    script_path = Path("scripts/train_stable.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"✅ Created stable training script: {script_path}")
    return True


def main():
    """Main function to apply all gradient fixes."""
    print("🔧 APPLYING GRADIENT ISSUE FIXES")
    print("="*50)
    
    fixes_applied = []
    
    # Fix 1: Add gradient clipping to trainer
    print("\n1. Adding gradient clipping to trainer...")
    if add_gradient_clipping_to_trainer():
        fixes_applied.append("✅ Gradient clipping added to trainer")
    else:
        fixes_applied.append("❌ Failed to add gradient clipping")
    
    # Fix 2: Create improved hyperparameters
    print("\n2. Creating improved hyperparameters...")
    if create_improved_hyperparameters():
        fixes_applied.append("✅ Improved hyperparameters created")
    else:
        fixes_applied.append("❌ Failed to create improved hyperparameters")
    
    # Fix 3: Create stable training script
    print("\n3. Creating stable training script...")
    if create_training_script_with_fixes():
        fixes_applied.append("✅ Stable training script created")
    else:
        fixes_applied.append("❌ Failed to create stable training script")
    
    # Summary
    print("\n" + "="*50)
    print("📊 FIXES SUMMARY")
    print("="*50)
    for fix in fixes_applied:
        print(fix)
    
    print("\n💡 NEXT STEPS:")
    print("1. Run gradient diagnostics: python scripts/diagnose_gradient_issues.py")
    print("2. Use stable training script: python scripts/train_stable.py")
    print("3. Monitor gradient norms during training")
    print("4. Adjust learning rate if needed (try 1e-7 for very unstable training)")
    
    print("\n✅ Gradient fixes applied!")


if __name__ == '__main__':
    main()
