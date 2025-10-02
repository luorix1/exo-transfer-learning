#!/usr/bin/env python3
"""
Training script for TCN-based joint moment prediction from IMU data.
Adapted for the Canonical dataset format.
"""

import os
import torch
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


def get_hyperparameter_config():
    """Define hyperparameter configuration."""
    return {
        # Model architecture
        'input_size': 6,  # 6 gyro channels (pelvis + thigh)
        'output_size': 2,  # 2 hip moments (right and left)
        'num_channels': [64, 64, 32, 32],
        'kernel_size': 2,
        'number_of_layers': 2,
        'dropout': 0.2,
        'dilations': [1, 2, 4, 8],
        'window_size': 100,  # 1 second at 100Hz
        
        # Training parameters
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'number_of_workers': 4,
        'validation_split': 0.2,
        'dataset_proportion': 1.0,
        'transfer_learning': False,
        
        # Data paths
        'data_root': '/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Canonical',
        'save_dir': './checkpoints',
        
        # Wandb configuration
        'wandb_session_name': 'tcn_joint_moment_prediction',
        'wandb_project': 'transfer-learning',
        'wandb_entity': None,
    }


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
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--window_size', type=int, default=100,
                       help='Window size for temporal sequences')
    parser.add_argument('--side', type=str, choices=['right', 'left', 'all'], default='all',
                       help='Which side to use: right-only, left-only, or both')
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
    
    args = parser.parse_args()
    
    # Get hyperparameter configuration
    config = get_hyperparameter_config()
    
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
        'side': args.side,
    })

    # Adjust output size based on side selection
    if args.side in ['right', 'left']:
        config['output_size'] = 1
    else:
        # both sides
        config['output_size'] = 2
    
    # Generate wandb run name if not provided
    if args.wandb_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['wandb_session_name'] = f"tcn_joint_moment_{timestamp}"
    else:
        config['wandb_session_name'] = args.wandb_name
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    
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
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    trainer.evaluate()
    
    if wandb_run:
        wandb.finish()
    
    print("Training completed!")


if __name__ == '__main__':
    main()
