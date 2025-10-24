#!/usr/bin/env python3
"""
IMU Position Analysis Runner
---------------------------
Runner script for IMU position optimization using pre-determined orientations.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_single_position_analysis(args):
    """Run position optimization for a single trial."""
    cmd = [
        sys.executable, "analysis/imu_position_optimization.py",
        "--dataset-root", args.dataset_root,
        "--subject", args.subject,
        "--condition", args.condition,
        "--trial", args.trial,
        "--segments", args.segments,
        "--output", args.output,
        "--orientation-results", args.orientation_results,
        "--max-frames", str(args.max_frames),
        "--cost-type", args.cost_type,
        "--n-starts", str(args.n_starts)
    ]
    
    if args.multi_start:
        cmd.append("--multi-start")
    
    
    print(f"🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Position optimization failed:")
        print(result.stderr)
        return False
    
    print("✅ Position optimization completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="IMU Position Analysis Runner")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Single trial analysis
    single_parser = subparsers.add_parser("single", help="Run position optimization for single trial")
    single_parser.add_argument("--dataset-root", required=True, help="Path to dataset root")
    single_parser.add_argument("--subject", required=True, help="Subject ID")
    single_parser.add_argument("--condition", required=True, help="Condition name")
    single_parser.add_argument("--trial", required=True, help="Trial name")
    single_parser.add_argument("--segments", default="femur_r", help="Comma-separated segments or 'all'")
    single_parser.add_argument("--output", required=True, help="Output directory")
    single_parser.add_argument("--orientation-results", required=True, help="Path to orientation optimization results")
    single_parser.add_argument("--max-frames", type=int, default=2000, help="Maximum frames to process")
    single_parser.add_argument("--cost-type", default="correlation", choices=["mse", "correlation", "weighted_mse"], 
                               help="Cost function type for optimization")
    single_parser.add_argument("--multi-start", action="store_true", default=True, 
                               help="Use multi-start optimization (default: True)")
    single_parser.add_argument("--n-starts", type=int, default=5, 
                               help="Number of random starts for multi-start optimization")
    
    args = parser.parse_args()
    
    if args.command == "single":
        success = run_single_position_analysis(args)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
