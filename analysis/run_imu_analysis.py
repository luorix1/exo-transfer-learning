#!/usr/bin/env python3
"""
Simple runner script for IMU orientation analysis

This script provides easy-to-use commands for running IMU orientation optimization
on the Final dataset structure.

Usage:
    # Single trial analysis
    python run_imu_analysis.py single \
        --dataset "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Molinaro_Phase1_Phase2" \
        --subject AB01 --condition levelground --trial LG_C0p0_S0p0_BT_1_10 \
        --output results/single_trial/
    
    # Batch analysis
    python run_imu_analysis.py batch \
        --dataset "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Molinaro_Phase1_Phase2" \
        --subjects AB01,AB02,AB03 --conditions levelground,ramp \
        --output results/batch_analysis/
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_single_analysis(args):
    """Run single trial IMU optimization"""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "imu_orientation_optimization.py"),
        "--dataset-root", args.dataset,
        "--subject", args.subject,
        "--condition", args.condition,
        "--trial", args.trial,
        "--output", args.output,
        "--segments", args.segments,
        "--max-frames", str(args.max_frames)
    ]
    
    if args.gyro_in_degrees:
        cmd.append("--gyro-in-degrees")
    
    if args.debug:
        cmd.append("--debug")
    
    print(f"🚀 Running single trial analysis...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Single trial analysis completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Single trial analysis failed with return code {e.returncode}")
        sys.exit(1)


def run_batch_analysis(args):
    """Run batch IMU optimization"""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "batch_imu_optimization.py"),
        "--dataset-root", args.dataset,
        "--output-root", args.output,
        "--condition", args.condition,
        "--trial", args.trial,
        "--segments", args.segments,
        "--max-frames", str(args.max_frames)
    ]
    
    if args.subjects:
        cmd.extend(["--subjects", args.subjects])
    
    if args.gyro_in_degrees:
        cmd.append("--gyro-in-degrees")
    
    if args.debug:
        cmd.append("--debug")
    
    print(f"🚀 Running batch analysis...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Batch analysis completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Batch analysis failed with return code {e.returncode}")
        sys.exit(1)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="IMU Orientation Analysis Runner")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single trial parser
    single_parser = subparsers.add_parser('single', help='Run single trial analysis')
    single_parser.add_argument("--dataset", required=True, help="Path to Final dataset root")
    single_parser.add_argument("--subject", required=True, help="Subject ID")
    single_parser.add_argument("--condition", required=True, help="Condition")
    single_parser.add_argument("--trial", required=True, help="Trial name")
    single_parser.add_argument("--output", required=True, help="Output directory")
    single_parser.add_argument("--segments", default="all", help="Comma-separated segments or 'all' (default: all)")
    single_parser.add_argument("--max-frames", type=int, default=100000, help="Max frames (default: 100000)")
    single_parser.add_argument("--gyro-in-degrees", action="store_true", help="Use degrees for gyro data")
    single_parser.add_argument("--debug", action="store_true", help="Enable debug mode with separate IMU plots")
    
    # Batch parser
    batch_parser = subparsers.add_parser('batch', help='Run batch analysis across subjects')
    batch_parser.add_argument("--dataset", required=True, help="Path to Final dataset root")
    batch_parser.add_argument("--output", required=True, help="Output directory")
    batch_parser.add_argument("--subjects", help="Comma-separated subjects (default: all)")
    batch_parser.add_argument("--condition", required=True, help="Condition name")
    batch_parser.add_argument("--trial", required=True, help="Trial name")
    batch_parser.add_argument("--segments", default="all", help="Comma-separated segments or 'all' (default: all)")
    batch_parser.add_argument("--max-frames", type=int, default=100000, help="Max frames (default: 100000)")
    batch_parser.add_argument("--gyro-in-degrees", action="store_true", help="Use degrees for gyro data")
    batch_parser.add_argument("--debug", action="store_true", help="Enable debug mode with separate IMU plots")
    
    args = parser.parse_args()
    
    if args.command == 'single':
        run_single_analysis(args)
    elif args.command == 'batch':
        run_batch_analysis(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
