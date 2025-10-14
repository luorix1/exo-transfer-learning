#!/usr/bin/env python3
"""Training script for the GMF-based joint moment estimation model."""

import argparse
import json
import os
from datetime import datetime
import numpy as np

import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb

from config.hyperparameters import DEFAULT_GMF_CONFIG
from data.dataloader import DataHandler
from model.gmf import GMFModel
from gmf_trainer import GMFTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train GMF-based joint moment estimator')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_gmf', help='Directory for saving checkpoints')
    parser.add_argument('--train_subjects', nargs='+', required=True, help='Training subject identifiers')
    parser.add_argument('--test_subjects', nargs='+', required=True, help='Test subject identifiers')
    parser.add_argument('--conditions', nargs='+', default=['levelground'], help='Conditions to use')
    parser.add_argument('--imu_segments', nargs='+', default=['pelvis', 'femur'], help='IMU segments to use')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--window_size', type=int, default=95, help='Temporal window size')
    parser.add_argument('--label_filter_hz', type=float, default=6.0, help='Label low-pass cutoff frequency')
    parser.add_argument('--augment', action='store_true', help='Enable augmentation during training')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='transfer-learning', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--gmf_loss_weight', type=float, default=1.0, help='Weight for GMF alignment loss (L1)')
    parser.add_argument('--decoder_loss_weight', type=float, default=0.05, help='Weight for decoder reconstruction loss (L2)')
    parser.add_argument('--seed', type=int, default=1000, help='Random seed for reproducibility')
    parser.add_argument('--lr_ge', type=float, default=None, help='Override learning rate for Generator+Estimator')
    parser.add_argument('--lr_gd', type=float, default=None, help='Override learning rate for Generator+Decoder')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizers')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of warmup epochs to run Phase A only')
    parser.add_argument('--phaseA_decoder_coeff', type=float, default=0.0, help='Optional small L2 coefficient in Phase A (no-grad to D)')
    parser.add_argument('--no_normalize', action='store_true', help='Disable normalization of inputs/labels')
    return parser.parse_args()


def main():
    args = parse_args()

    config = DEFAULT_GMF_CONFIG.copy()

    if len(args.imu_segments) == 1 and args.imu_segments[0].lower() in ['femur', 'thigh']:
        input_size = 3
    else:
        input_size = 6

    config.update({
        'data_root': args.data_root,
        'save_dir': args.save_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'window_size': args.window_size,
        'label_filter_hz': args.label_filter_hz,
        'augment': args.augment,
        'normalize': not args.no_normalize,
        'imu_segments': args.imu_segments,
        'gmf_loss_weight': args.gmf_loss_weight,
        'decoder_loss_weight': args.decoder_loss_weight,
        'warmup_epochs': args.warmup_epochs,
        'phaseA_decoder_coeff': args.phaseA_decoder_coeff,
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
        'input_size': input_size,
    })

    config['output_size'] = 1
    config['use_subject_info'] = True
    config['train_subjects'] = args.train_subjects
    config['test_subjects'] = args.test_subjects
    config['conditions'] = args.conditions

    os.makedirs(args.save_dir, exist_ok=True)
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    if args.wandb_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"gmf_joint_moment_{timestamp}"
    else:
        run_name = args.wandb_name
    config['wandb_session_name'] = run_name

    # Single run only
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if not args.no_wandb:
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            name=f"{run_name}_seed{seed}",
            config={**config, 'seed': seed, 'train_subjects': args.train_subjects, 'test_subjects': args.test_subjects},
            tags=['gmf', 'joint_moment', 'imu'],
            reinit=True,
        )
        wandb_run = wandb.run
    else:
        wandb_run = None

    data_handler = DataHandler(args.data_root, config)
    data_handler.load_data(
        train_data_partition=args.train_subjects,
        train_data_condition=args.conditions,
        test_data_partition=args.test_subjects,
    )

    train_indices, val_indices = data_handler.get_train_val_indices()
    train_loader, val_loader = data_handler.create_dataloaders(train_indices, val_indices)
    test_loader = data_handler.create_dataloaders(test_indices=1)

    param_size = 0
    if getattr(data_handler.train_data, 'subject_params', None) is not None:
        param_size = data_handler.train_data.subject_params.shape[1]

    model = GMFModel(
        input_size=config['input_size'],
        output_size=config['output_size'],
        gmf_size=config['gmf_size'],
        generator_hidden_size=config['generator_hidden_size'],
        generator_hidden_layers=config['generator_hidden_layers'],
        estimator_hidden_size=config['estimator_hidden_size'],
        decoder_hidden_size=config['decoder_hidden_size'],
        decoder_hidden_layers=config['decoder_hidden_layers'],
        param_size=param_size,
    ).to(device)

    # Two optimizers: GE (generator+estimator) and GD (generator+decoder)
    params_ge = list(model.generator.parameters()) + list(model.estimator.parameters())
    params_gd = list(model.generator.parameters()) + list(model.decoder.parameters())
    lr_ge = float(args.lr_ge) if args.lr_ge is not None else float(config['learning_rate'])
    lr_gd = float(args.lr_gd) if args.lr_gd is not None else float(config['learning_rate'])
    wd = float(args.weight_decay)
    optimizer_ge = Adam(params_ge, lr=lr_ge, weight_decay=wd)
    optimizer_gd = Adam(params_gd, lr=lr_gd, weight_decay=wd)
    scheduler_ge = ReduceLROnPlateau(optimizer_ge, mode='min', patience=5, factor=0.5, verbose=True)
    scheduler_gd = ReduceLROnPlateau(optimizer_gd, mode='min', patience=5, factor=0.5, verbose=True)

    trainer = GMFTrainer(
        model=model,
        device=device,
        optimizer_ge=optimizer_ge,
        optimizer_gd=optimizer_gd,
        scheduler_ge=scheduler_ge,
        scheduler_gd=scheduler_gd,
        data_handler=data_handler,
        config=config,
        save_dir=args.save_dir,
        wandb_run=wandb_run,
    )

    trainer.fit(train_loader, val_loader)
    best_epoch = trainer.load_best_model()
    if best_epoch is not None:
        print(f"Loaded best checkpoint from epoch {best_epoch}")

    data_handler.save_mean_std(args.save_dir)

    test_metrics = trainer.test(test_loader)
    print(f"Test Loss: {test_metrics['loss']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f} Nm/kg")
    print(f"Test MAE: {test_metrics['mae']:.4f} Nm/kg")

    if wandb_run is not None:
        wandb.log({'test/loss': test_metrics['loss'], 'test/rmse': test_metrics['rmse'], 'test/mae': test_metrics['mae'], 'test/accuracy': test_metrics.get('accuracy', 0.0)})
        wandb.finish()


if __name__ == '__main__':
    main()
