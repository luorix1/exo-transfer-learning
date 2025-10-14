"""
Hyperparameter configurations for different model architectures and training scenarios.
"""

# Default TCN configuration for joint moment prediction
# Based on reference biotorque controller training
DEFAULT_TCN_CONFIG = {
    # Model architecture
    'input_size': 6,  # Auto-adjusted based on imu_segments: 3 for single, 6 for dual
    'output_size': 1,  # 1 hip moment (unilateral, trains with both sides stacked)
    'num_channels': [80, 80, 80, 80, 80],  # Match reference architecture
    'kernel_size': 5,  # Match reference
    'number_of_layers': 2,
    'dropout': 0.15,  # Match reference
    'dilations': [1, 2, 4, 8, 16],  # Match reference
    'window_size': 95,  # Match reference (slightly less than 1 second)
    
    # Training parameters
    'epochs': 30,  # Match reference
    'batch_size': 32,  # Match reference
    'learning_rate': 5e-6,  # Match reference init_lr
    'number_of_workers': 0,  # Set to 0 for macOS (multiprocessing overhead), use 4-8 for Linux
    'validation_split': 0.1,  # Match reference
    'dataset_proportion': 1.0,  # Match reference
    'transfer_learning': False,
    
    # Wandb configuration
    'wandb_session_name': 'tcn_joint_moment_prediction',
    'wandb_project': 'transfer-learning',
    'wandb_entity': None,
}

# GMF-based model configuration
DEFAULT_GMF_CONFIG = {
    # Model architecture
    'input_size': 6,  # Auto-adjusted based on imu_segments: 3 for single, 6 for dual
    'output_size': 1,
    'gmf_size': 16,  # Dimension of the generalized moment feature representation
    'generator_hidden_size': 32,
    'generator_hidden_layers': 5,
    'estimator_hidden_size': 16,
    'decoder_hidden_size': 32,
    'decoder_hidden_layers': 5,

    # Training parameters
    'epochs': 60,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'number_of_workers': 0,
    'validation_split': 0.1,
    'dataset_proportion': 1.0,
    'transfer_learning': False,
    'window_size': 95,
    'label_filter_hz': 6.0,
    'augment': False,
    'normalize': True,
    'imu_segments': ['pelvis', 'femur'],
    'use_subject_info': True,

    # Loss weights
    'gmf_loss_weight': 1.0,
    'decoder_loss_weight': 1.0,

    # Wandb configuration
    'wandb_session_name': 'gmf_joint_moment_prediction',
    'wandb_project': 'transfer-learning',
    'wandb_entity': None,
}

# Larger TCN configuration for more complex patterns
LARGE_TCN_CONFIG = {
    **DEFAULT_TCN_CONFIG,
    'num_channels': [128, 128, 64, 64, 32],
    'number_of_layers': 3,
    'dropout': 0.3,
    'dilations': [1, 2, 4, 8, 16],
    'batch_size': 16,  # Smaller batch size for larger model
    'learning_rate': 0.0005,
}

# Smaller TCN configuration for faster training
SMALL_TCN_CONFIG = {
    **DEFAULT_TCN_CONFIG,
    'num_channels': [32, 32, 16],
    'number_of_layers': 2,
    'dropout': 0.1,
    'dilations': [1, 2, 4],
    'batch_size': 64,  # Larger batch size for smaller model
    'learning_rate': 0.002,
}

# Transfer learning configuration
TRANSFER_LEARNING_CONFIG = {
    **DEFAULT_TCN_CONFIG,
    'transfer_learning': True,
    'learning_rate': 0.0001,  # Lower learning rate for fine-tuning
    'epochs': 50,  # Fewer epochs for transfer learning
    'dataset_proportion': 0.5,  # Use less data for transfer learning
}

# Subject-specific configurations
SUBJECT_SPECIFIC_CONFIGS = {
    'BT01': {**DEFAULT_TCN_CONFIG, 'window_size': 150},
    'BT02': {**DEFAULT_TCN_CONFIG, 'window_size': 120},
    'BT03': {**DEFAULT_TCN_CONFIG, 'window_size': 100},
    # Add more subject-specific configs as needed
}

# Condition-specific configurations
CONDITION_SPECIFIC_CONFIGS = {
    'levelground': {**DEFAULT_TCN_CONFIG, 'window_size': 100},
    'treadmill': {**DEFAULT_TCN_CONFIG, 'window_size': 80},
    'stairs': {**DEFAULT_TCN_CONFIG, 'window_size': 120},
}

def get_config(config_name: str = 'default'):
    """Get hyperparameter configuration by name."""
    configs = {
        'default': DEFAULT_TCN_CONFIG,
        'gmf': DEFAULT_GMF_CONFIG,
        'large': LARGE_TCN_CONFIG,
        'small': SMALL_TCN_CONFIG,
        'transfer': TRANSFER_LEARNING_CONFIG,
    }
    
    if config_name in configs:
        return configs[config_name]
    else:
        raise ValueError(f"Unknown configuration: {config_name}. Available: {list(configs.keys())}")

def get_subject_config(subject: str):
    """Get subject-specific configuration."""
    return SUBJECT_SPECIFIC_CONFIGS.get(subject, DEFAULT_TCN_CONFIG)

def get_condition_config(condition: str):
    """Get condition-specific configuration."""
    return CONDITION_SPECIFIC_CONFIGS.get(condition, DEFAULT_TCN_CONFIG)
