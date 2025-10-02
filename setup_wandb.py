#!/usr/bin/env python3
"""
Setup script for wandb integration.
Run this to configure wandb for the transfer learning project.
"""

import os
import yaml
import wandb
import argparse


def setup_wandb(entity=None, project="transfer-learning"):
    """Setup wandb configuration."""
    print("Setting up wandb for transfer learning project...")
    
    # Login to wandb (will prompt for API key if not already logged in)
    try:
        wandb.login()
        print("✅ Wandb login successful")
    except Exception as e:
        print(f"❌ Wandb login failed: {e}")
        print("Please run 'wandb login' manually and provide your API key")
        return False
    
    # Create wandb config file
    config = {
        'project': project,
        'entity': entity,
        'default_tags': ['tcn', 'joint_moment', 'imu', 'biomechanics'],
        'notes': 'TCN model for joint moment prediction from IMU data'
    }
    
    with open('wandb_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Wandb configuration saved to wandb_config.yaml")
    print(f"Project: {project}")
    if entity:
        print(f"Entity: {entity}")
    
    return True


def test_wandb():
    """Test wandb integration with a simple run."""
    print("Testing wandb integration...")
    
    try:
        # Initialize a test run
        run = wandb.init(
            project="transfer-learning",
            name="test_run",
            config={
                'test': True,
                'model': 'tcn',
                'task': 'joint_moment_prediction'
            },
            tags=['test', 'setup']
        )
        
        # Log some test metrics
        wandb.log({
            'test_metric': 0.95,
            'test_loss': 0.05
        })
        
        # Finish the run
        wandb.finish()
        
        print("✅ Wandb test successful")
        return True
        
    except Exception as e:
        print(f"❌ Wandb test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Setup wandb for transfer learning project')
    parser.add_argument('--entity', type=str, default=None,
                       help='Wandb entity name (your username or team)')
    parser.add_argument('--project', type=str, default='transfer-learning',
                       help='Wandb project name')
    parser.add_argument('--test', action='store_true',
                       help='Test wandb integration')
    
    args = parser.parse_args()
    
    # Setup wandb
    if setup_wandb(entity=args.entity, project=args.project):
        print("\n🎉 Wandb setup completed successfully!")
        print("\nNext steps:")
        print("1. Run training: python src/train.py")
        print("2. View results: Check your wandb dashboard")
        print("3. Use tags: --wandb_tags experiment1 baseline")
        
        if args.test:
            print("\nTesting wandb integration...")
            test_wandb()
    else:
        print("\n❌ Wandb setup failed. Please check your configuration.")


if __name__ == '__main__':
    main()
