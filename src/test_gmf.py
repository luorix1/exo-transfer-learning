#!/usr/bin/env python3
"""Evaluation script for the GMF-based joint moment estimation model."""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from config.hyperparameters import DEFAULT_GMF_CONFIG
from data.dataloader import DataHandler
from model.gmf import GMFModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a trained GMF-based joint moment estimator')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to a checkpoint file or directory')
    parser.add_argument('--test_subjects', nargs='+', default=None, help='Test subject identifiers (overrides config)')
    parser.add_argument('--conditions', nargs='+', default=None, help='Conditions to evaluate (defaults to config)')
    parser.add_argument('--imu_segments', nargs='+', default=None, help='IMU segments override')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size override')
    parser.add_argument('--window_size', type=int, default=None, help='Temporal window size override')
    parser.add_argument('--no_normalize', action='store_true', help='Disable normalization (use raw values)')
    return parser.parse_args()


def load_config(checkpoint: str) -> Tuple[dict, Path]:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        config_path = checkpoint_path / 'config.json'
        checkpoint_dir = checkpoint_path
    else:
        checkpoint_dir = checkpoint_path.parent
        config_path = checkpoint_dir / 'config.json'

    config = DEFAULT_GMF_CONFIG.copy()
    if config_path.exists():
        with config_path.open('r') as f:
            loaded = json.load(f)
        config.update(loaded)
    else:
        raise FileNotFoundError(f"Could not locate config.json next to checkpoint at {config_path}")

    return config, checkpoint_dir


def resolve_checkpoint_file(checkpoint: str, checkpoint_dir: Path, config: dict) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file():
        return checkpoint_path

    best_checkpoint = config.get('best_checkpoint')
    if best_checkpoint:
        candidate = checkpoint_dir / best_checkpoint
        if candidate.exists():
            return candidate

    checkpoints = sorted(checkpoint_dir.glob('gmf_model_epoch_*.pt'))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    return checkpoints[-1]


def build_model(config: dict, param_size: int, device: torch.device) -> GMFModel:
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
    return model


def evaluate(model: GMFModel, data_handler: DataHandler, test_loader, device: torch.device) -> dict:
    model.eval()
    label_mean = torch.tensor(data_handler.label_mean, device=device)
    label_std = torch.tensor(data_handler.label_std, device=device)

    metrics = {'rmse': 0.0, 'mae': 0.0, 'batches': 0}

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, params = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            params = params.to(device)

            gmf_estimated = model.estimator(inputs)
            decoded = model.decode(params, gmf_estimated)

            preds_denorm = decoded * label_std + label_mean
            targets_denorm = targets * label_std + label_mean
            diff = preds_denorm - targets_denorm

            mse = torch.mean(diff ** 2).item()
            mae = torch.mean(torch.abs(diff)).item()

            metrics['rmse'] += float(np.sqrt(mse))
            metrics['mae'] += mae
            metrics['batches'] += 1

    if metrics['batches'] > 0:
        metrics['rmse'] /= metrics['batches']
        metrics['mae'] /= metrics['batches']

    return {'rmse': metrics['rmse'], 'mae': metrics['mae']}


def main() -> None:
    args = parse_args()

    config, checkpoint_dir = load_config(args.checkpoint)

    if args.window_size is not None:
        config['window_size'] = args.window_size
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.imu_segments is not None:
        config['imu_segments'] = args.imu_segments
    if args.no_normalize:
        config['normalize'] = False

    test_subjects: Optional[List[str]] = args.test_subjects or config.get('test_subjects')
    if not test_subjects:
        raise ValueError('No test subjects specified. Provide --test_subjects or store them in config.json.')

    conditions = args.conditions or config.get('conditions') or ['levelground']

    data_handler = DataHandler(
        args.data_root,
        config,
        pretrained_model_path=str(checkpoint_dir),
    )

    data_handler.load_test_data_only(test_subjects, conditions)
    test_loader = data_handler.create_dataloaders(test_indices=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    param_size = 0
    if getattr(data_handler.test_data, 'subject_params', None) is not None:
        param_size = data_handler.test_data.subject_params.shape[1]
    elif config.get('param_size') is not None:
        param_size = int(config['param_size'])

    model = build_model(config, param_size, device)

    checkpoint_file = resolve_checkpoint_file(args.checkpoint, checkpoint_dir, config)
    state = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    metrics = evaluate(model, data_handler, test_loader, device)
    print(f"Evaluation RMSE: {metrics['rmse']:.4f} Nm/kg")
    print(f"Evaluation MAE: {metrics['mae']:.4f} Nm/kg")


if __name__ == '__main__':
    main()
