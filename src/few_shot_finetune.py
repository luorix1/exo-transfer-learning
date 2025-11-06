#!/usr/bin/env python3
"""Few-shot fine-tuning script for TCN-based joint moment estimation model.

This script fine-tunes a Camargo-trained TCN model on a single MetaMobility walking trial (AB14_Evy).
The model's output is inverted by negating the final linear layer weights to account
for sign convention differences between datasets.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb

from config.hyperparameters import DEFAULT_TCN_CONFIG
from data.dataloader import DataHandler
from model.tcn import TCNModel
from trainer import Trainer
from loss import JointMomentLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Few-shot fine-tune TCN model on MetaMobility trial')
    parser.add_argument('--pretrained_checkpoint', type=str, required=True,
                       help='Path to Camargo-trained checkpoint directory')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to MetaMobility dataset root directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_finetune',
                       help='Directory for saving fine-tuned checkpoints')
    parser.add_argument('--subject', type=str, default='AB14_Evy',
                       help='MetaMobility subject identifier (default: AB14_Evy)')
    parser.add_argument('--conditions', nargs='+', default=['1p2mps'],
                       help='Conditions to use for fine-tuning')
    parser.add_argument('--imu_segments', nargs='+', default=['femur'],
                       help='IMU segments to use')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate for fine-tuning (lower than training)')
    parser.add_argument('--window_size', type=int, default=None,
                       help='Temporal window size (uses pretrained config if not specified)')
    parser.add_argument('--label_filter_hz', type=float, default=None,
                       help='Label low-pass cutoff frequency (uses pretrained config if not specified)')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='transfer-learning',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=1000,
                       help='Random seed for reproducibility')
    parser.add_argument('--no_normalize', action='store_true',
                       help='Disable input normalization (use raw IMU data)')
    return parser.parse_args()


def load_pretrained_config(checkpoint_dir: str) -> dict:
    """Load configuration from pretrained checkpoint."""
    config_path = Path(checkpoint_dir) / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def invert_model_output(model: TCNModel) -> None:
    """Invert the model output by negating the final linear layer weights and bias.
    
    This accounts for sign convention differences between Camargo and MetaMobility datasets.
    The final layer is model.linear, which is a Linear layer.
    
    Mathematical justification:
    - Original output: y = W*x + b
    - Desired inverted output: y' = -y = -(W*x + b) = -W*x - b
    - By negating both weights and bias, we get: y' = (-W)*x + (-b) = -W*x - b ✓
    
    This initialization allows the model to start from the inverted output, and fine-tuning
    can further adjust the weights if needed.
    """
    final_layer = model.linear
    if not isinstance(final_layer, torch.nn.Linear):
        raise ValueError(f"Expected final layer to be Linear, got {type(final_layer)}")
    
    # Negate weights and bias to invert output
    with torch.no_grad():
        final_layer.weight.data = -final_layer.weight.data
        if final_layer.bias is not None:
            final_layer.bias.data = -final_layer.bias.data
    
    print("✓ Inverted model output by negating final linear layer weights and bias")


def load_pretrained_model(checkpoint_dir: str, device: torch.device, 
                         config: dict) -> TCNModel:
    """Load pretrained model from checkpoint."""
    # Determine checkpoint file
    checkpoint_path = Path(checkpoint_dir)
    
    # Try to find the best model (usually the wandb_session_name.pt file)
    # Or look for the latest epoch checkpoint
    best_model_name = config.get('wandb_session_name', 'tcn_joint_moment')
    best_model_path = checkpoint_path / f'{best_model_name}.pt'
    
    if not best_model_path.exists():
        # Try to find latest epoch checkpoint
        checkpoints = sorted(checkpoint_path.glob(f'{best_model_name}_epoch_*.pt'))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
        best_model_path = checkpoints[-1]
    
    if not best_model_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {best_model_path}")
    
    # Build model with same architecture
    model = TCNModel(config).to(device)
    
    # Load pretrained weights
    print(f"Loading pretrained weights from {best_model_path}")
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✓ Loaded pretrained model")
    
    # Invert model output for MetaMobility sign convention
    invert_model_output(model)
    
    return model


def main():
    args = parse_args()
    
    # Load pretrained config
    pretrained_config = load_pretrained_config(args.pretrained_checkpoint)
    
    # Create config for fine-tuning (use pretrained config as base)
    config = DEFAULT_TCN_CONFIG.copy()
    config.update(pretrained_config)  # Override with pretrained settings
    
    # Override with fine-tuning specific settings
    if args.window_size is not None:
        config['window_size'] = args.window_size
    if args.label_filter_hz is not None:
        config['label_filter_hz'] = args.label_filter_hz
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['epochs'] = args.epochs
    config['imu_segments'] = args.imu_segments
    config['data_root'] = args.data_root
    config['save_dir'] = args.save_dir
    config['conditions'] = args.conditions
    config['normalize'] = not args.no_normalize
    
    # Determine input size
    if len(args.imu_segments) == 1 and args.imu_segments[0].lower() in ['femur', 'thigh']:
        input_size = 3
    else:
        input_size = 6
    config['input_size'] = input_size
    
    config['output_size'] = 1
    config['train_subjects'] = [args.subject]
    config['test_subjects'] = [args.subject]  # Use same subject for test (since it's few-shot)
    
    os.makedirs(args.save_dir, exist_ok=True)
    config_path = os.path.join(args.save_dir, 'config.json')
    
    if args.wandb_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"finetune_tcn_{args.subject}_{timestamp}"
    else:
        run_name = args.wandb_name
    config['wandb_session_name'] = run_name
    
    # Set up device and random seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    print(f"Using device: {device}")
    
    # Initialize wandb
    if not args.no_wandb:
        experiment_config = {
            **config,
            'train_subjects': config['train_subjects'],
            'test_subjects': config['test_subjects'],
            'pretrained_checkpoint': args.pretrained_checkpoint,
            'fine_tuning': True,
            'subject': args.subject,
            'device': str(device),
            'pytorch_version': torch.__version__,
        }
        
        wandb.init(
            project=config.get('wandb_project', 'transfer-learning'),
            entity=config.get('wandb_entity'),
            name=f"{run_name}_seed{seed}",
            config=experiment_config,
            tags=['tcn', 'joint_moment', 'imu', 'few-shot', 'finetune'],
            reinit=True,
        )
        wandb_run = wandb.run
        print(f"Wandb run initialized: {wandb_run.url}")
    else:
        wandb_run = None
        print("Wandb logging disabled")
    
    # Set up data handler with pretrained normalization
    data_handler = DataHandler(
        data_root=config['data_root'],
        hyperparam_config=config,
        pretrained_model_path=args.pretrained_checkpoint  # Use pretrained normalization
    )
    
    # Load data (only the single subject for few-shot fine-tuning)
    print(f"\nLoading data for fine-tuning subject: {args.subject}")
    data_handler.load_data(
        train_data_partition=[args.subject],
        train_data_condition=args.conditions,
        test_data_partition=[args.subject]  # Test on same subject
    )
    
    # Save normalization parameters (will be same as pretrained)
    data_handler.save_mean_std(args.save_dir)
    
    # Load pretrained model and invert output
    model = load_pretrained_model(args.pretrained_checkpoint, device, pretrained_config)
    
    # Initialize loss function
    criterion = JointMomentLoss()
    
    # Initialize optimizer with lower learning rate for fine-tuning
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=args.weight_decay)
    
    # Initialize scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Save config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create trainer
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
    
    # Fine-tune the model
    print(f"\nStarting few-shot fine-tuning on {args.subject} for {args.epochs} epochs...")
    train_indices, val_indices = data_handler.get_train_val_indices()
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    test_loader = trainer.train()
    
    # Evaluate the model (reuse test_loader from training)
    print("\nFinal evaluation on test set with best model...")
    trainer.evaluate(test_loader)
    
    if wandb_run is not None:
        wandb.finish()
    
    print("\n✓ Few-shot fine-tuning completed!")


if __name__ == '__main__':
    main()