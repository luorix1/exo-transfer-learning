#!/usr/bin/env python3
"""Training script for the GMF-based joint moment estimation model."""

import argparse
import json
import os
from datetime import datetime

import torch
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
    parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds to run')
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

    wandb_run = None

    all_seed_metrics = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for seed_idx in range(args.seeds):
        seed = 1000 + seed_idx
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
        optimizer_ge = Adam(params_ge, lr=config['learning_rate'])
        optimizer_gd = Adam(params_gd, lr=config['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer_ge, mode='min', patience=5, factor=0.5, verbose=True)

        trainer = GMFTrainer(
            model=model,
            device=device,
            optimizer_ge=optimizer_ge,
            optimizer_gd=optimizer_gd,
            scheduler=scheduler,
            data_handler=data_handler,
            config=config,
            save_dir=os.path.join(args.save_dir, f"seed_{seed}"),
            wandb_run=wandb_run,
        )

        trainer.fit(train_loader, val_loader)
        best_epoch = trainer.load_best_model()
        if best_epoch is not None:
            print(f"Loaded best checkpoint from epoch {best_epoch}")

        data_handler.save_mean_std(os.path.join(args.save_dir, f"seed_{seed}"))

        test_metrics = trainer.test(test_loader)
        print(f"Seed {seed} - Test Loss: {test_metrics['loss']:.6f}")
        print(f"Seed {seed} - Test RMSE: {test_metrics['rmse']:.4f} Nm/kg")
        print(f"Seed {seed} - Test MAE: {test_metrics['mae']:.4f} Nm/kg")

        if wandb_run is not None:
            wandb.log({'test/loss': test_metrics['loss'], 'test/rmse': test_metrics['rmse'], 'test/mae': test_metrics['mae'], 'test/accuracy': test_metrics.get('accuracy', 0.0)})
            wandb.finish()

        all_seed_metrics.append(test_metrics)

    # Save aggregate results summary
    summary = {
        'seeds': args.seeds,
        'metrics': all_seed_metrics,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
    }
    with open(os.path.join(args.save_dir, 'results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
