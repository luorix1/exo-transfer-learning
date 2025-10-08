#!/usr/bin/env python3
"""
Training script for TCN-based joint moment prediction from IMU data.
Adapted for the Canonical dataset format.
"""

import os
import warnings
os.environ["MKL_VERBOSE"] = "0"
os.environ["MKL_DISABLE_FAST_MM"] = "1"
import json
import torch
# Suppress NNPACK warnings
warnings.filterwarnings('ignore', message='.*Could not initialize NNPACK.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
import wandb
import argparse
import numpy as np
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.tcn import TCNModel
from data.dataloader import DataHandler
from trainer import Trainer
from loss import JointMomentLoss
from config.hyperparameters import DEFAULT_TCN_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Train TCN model for joint moment prediction')
    parser.add_argument('--data_root', type=str, 
                       default='/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Canonical',
                       help='Path to Canonical dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--train_subjects', nargs='+', 
                       default=['BT01', 'BT02', 'BT03', 'BT06', 'BT07', 'BT08', 'BT09', 'BT10'],
                       help='Training subjects')
    parser.add_argument('--test_subjects', nargs='+', 
                       default=['BT11', 'BT12', 'BT13', 'BT14', 'BT15'],
                       help='Test subjects')
    parser.add_argument('--conditions', nargs='+', default=['levelground'],
                       help='Conditions to use for training')
    parser.add_argument('--imu_segments', nargs='+', default=['pelvis', 'femur'],
                       help='IMU segments to use: ["femur"] for single thigh IMU (3 channels), ["pelvis", "femur"] for dual (6 channels)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                       help='Learning rate (default: 5e-6 to match reference)')
    parser.add_argument('--window_size', type=int, default=100,
                       help='Window size for temporal sequences')
    parser.add_argument('--wandb_project', type=str, default='transfer-learning',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb run name (auto-generated if not provided)')
    parser.add_argument('--wandb_tags', nargs='+', default=[],
                       help='Wandb tags for the experiment')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--augment', action='store_true',
                       help='Enable training-time data augmentation')
    parser.add_argument('--use_curriculum', action='store_true',
                       help='Enable curriculum training (pretrain on smoothed labels then fine-tune)')
    parser.add_argument('--curriculum_epochs', type=int, default=0,
                       help='Number of initial epochs using heavier label smoothing (0 disables)')
    parser.add_argument('--label_filter_hz', type=float, default=6.0,
                       help='Low-pass cutoff frequency (Hz) for label smoothing')
    
    args = parser.parse_args()
    
    # Get hyperparameter configuration from config file
    config = DEFAULT_TCN_CONFIG.copy()
    
    # Auto-adjust input_size based on IMU segments
    if len(args.imu_segments) == 1 and args.imu_segments[0].lower() in ['femur', 'thigh']:
        input_size = 3  # Single IMU: 3 gyro channels
    else:
        input_size = 6  # Dual IMU: 6 gyro channels (pelvis + femur, or tibia + femur)
    
    # Update config with command line arguments
    config.update({
        'data_root': args.data_root,
        'save_dir': args.save_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'window_size': args.window_size,
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
        'imu_segments': args.imu_segments,
        'input_size': input_size,
        'augment': args.augment,
        'use_curriculum': args.use_curriculum,
        'curriculum_epochs': args.curriculum_epochs,
        'label_filter_hz': args.label_filter_hz,
    })

    # Unilateral controller: always output_size=1, train using both sides stacked (reference style)
    config['output_size'] = 1
    
    # Generate wandb run name if not provided
    if args.wandb_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['wandb_session_name'] = f"tcn_joint_moment_{timestamp}"
    else:
        config['wandb_session_name'] = args.wandb_name
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration to experiment folder
    config_save_path = os.path.join(args.save_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_save_path}")
    
    # Initialize wandb
    if not args.no_wandb:
        # Add experiment metadata
        experiment_config = {
            **config,
            'train_subjects': args.train_subjects,
            'test_subjects': args.test_subjects,
            'conditions': args.conditions,
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            'pytorch_version': torch.__version__,
        }
        
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            name=config['wandb_session_name'],
            config=experiment_config,
            tags=args.wandb_tags + ['tcn', 'joint_moment', 'imu'],
            notes=f"TCN model for joint moment prediction. Train: {len(args.train_subjects)} subjects, Test: {len(args.test_subjects)} subjects"
        )
        wandb_run = wandb.run
        
        # Log model architecture
        print(f"Wandb run initialized: {wandb_run.url}")
    else:
        wandb_run = None
        print("Wandb logging disabled")
    
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
    
    # Initialize optimizer (match reference with weight decay)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    
    # Initialize scheduler (match reference with patience=1)
    # Increased patience to 3 for more stable training
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
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
    print("Starting training...")
    test_loader = trainer.train()
    
    # Evaluate the model (reuse test_loader from training)
    print("\nFinal evaluation on test set with best model...")
    trainer.evaluate(test_loader)
    
    if wandb_run:
        wandb.finish()
    
    print("Training completed!")


if __name__ == '__main__':
    main()
