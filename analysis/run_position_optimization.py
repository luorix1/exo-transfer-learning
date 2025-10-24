#!/usr/bin/env python3
"""
Simple runner script for IMU position optimization

This script provides easy-to-use commands for running IMU position optimization
on the Final dataset structure, using pre-computed orientation results.

Usage:
    # Single segment position optimization
    python run_position_optimization.py \
        --dataset "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Camargo" \
        --subject AB21 --condition treadmill --trial treadmill_03_01 \
        --segment femur_r \
        --orientation-results results/camargo_multisegment/femur_r/optimization_results.json \
        --output results/position_optimization/camargo/femur_r/
    
    # Batch processing for all segments
    python run_position_optimization.py batch \
        --dataset "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Final/Camargo" \
        --subject AB21 --condition treadmill --trial treadmill_03_01 \
        --orientation-base results/camargo_multisegment/ \
        --output results/position_optimization/camargo/
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json


def run_single_optimization(args):
    """Run single segment position optimization"""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "imu_position_optimization.py"),
        "--dataset-root", args.dataset,
        "--subject", args.subject,
        "--condition", args.condition,
        "--trial", args.trial,
        "--segment", args.segment,
        "--orientation-results", args.orientation_results,
        "--output", args.output,
        "--max-frames", str(args.max_frames),
        "--optimization-method", args.method
    ]
    
    if args.imu_data_name:
        cmd.extend(["--imu-data-name", args.imu_data_name])
    
    print(f"🚀 Running position optimization for segment: {args.segment}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Position optimization completed successfully for {args.segment}!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Position optimization failed with return code {e.returncode}")
        sys.exit(1)


def run_batch_optimization(args):
    """Run batch position optimization for multiple segments"""
    
    # Load combined orientation results to get list of segments
    combined_file = Path(args.orientation_base) / "combined_results.json"
    
    if not combined_file.exists():
        print(f"❌ Combined orientation results not found: {combined_file}")
        print(f"   Make sure you've run orientation optimization first.")
        sys.exit(1)
    
    try:
        with open(combined_file, 'r') as f:
            combined_data = json.load(f)
        
        segments = list(combined_data['segments'].keys())
        print(f"📋 Found {len(segments)} segments with orientation results: {segments}")
        
    except Exception as e:
        print(f"❌ Error reading combined results: {e}")
        sys.exit(1)
    
    # Process each segment
    failed_segments = []
    
    for segment in segments:
        print(f"\n{'='*70}")
        print(f"🎯 Processing segment: {segment}")
        print(f"{'='*70}")
        
        # Find orientation results for this segment
        orientation_file = Path(args.orientation_base) / segment / "optimization_results.json"
        
        if not orientation_file.exists():
            print(f"⚠️  Orientation results not found for {segment}: {orientation_file}")
            failed_segments.append(segment)
            continue
        
        # Create output directory for this segment
        segment_output = Path(args.output) / segment
        segment_output.mkdir(exist_ok=True, parents=True)
        
        # Build command
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "imu_position_optimization.py"),
            "--dataset-root", args.dataset,
            "--subject", args.subject,
            "--condition", args.condition,
            "--trial", args.trial,
            "--segment", segment,
            "--orientation-results", str(orientation_file),
            "--output", str(segment_output),
            "--max-frames", str(args.max_frames),
            "--optimization-method", args.method
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Position optimization completed for {segment}!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Position optimization failed for {segment} with return code {e.returncode}")
            failed_segments.append(segment)
            if not args.continue_on_error:
                sys.exit(1)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 Batch Position Optimization Summary")
    print(f"{'='*70}")
    print(f"Total segments: {len(segments)}")
    print(f"Successful: {len(segments) - len(failed_segments)}")
    print(f"Failed: {len(failed_segments)}")
    
    if failed_segments:
        print(f"Failed segments: {failed_segments}")
    else:
        print("✅ All segments completed successfully!")
    
    # Create combined results
    if len(failed_segments) < len(segments):
        create_combined_results(args, segments, failed_segments)


def create_combined_results(args, all_segments, failed_segments):
    """Create a combined results file for all segments"""
    print(f"\n📝 Creating combined position optimization results...")
    
    combined_data = {
        "configuration": {
            "dataset": args.dataset,
            "subject": args.subject,
            "condition": args.condition,
            "trial": args.trial,
            "orientation_base": args.orientation_base
        },
        "segments": {}
    }
    
    for segment in all_segments:
        if segment in failed_segments:
            continue
        
        result_file = Path(args.output) / segment / "position_optimization_results.json"
        
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    segment_data = json.load(f)
                
                combined_data["segments"][segment] = {
                    "optimal_position": segment_data["optimization"]["optimal_position"],
                    "final_cost": segment_data["optimization"]["final_cost"],
                    "origin_cost": segment_data["optimization"]["origin_cost"],
                    "improvement_percent": segment_data["optimization"]["improvement_percent"]
                }
            except Exception as e:
                print(f"⚠️  Could not load results for {segment}: {e}")
    
    # Save combined results
    combined_output = Path(args.output) / "combined_position_results.json"
    with open(combined_output, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"✓ Combined results saved: {combined_output}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="IMU Position Optimization Runner")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single optimization parser
    single_parser = subparsers.add_parser('single', help='Run single segment position optimization')
    single_parser.add_argument("--dataset", required=True, help="Path to Final dataset root")
    single_parser.add_argument("--subject", required=True, help="Subject ID")
    single_parser.add_argument("--condition", required=True, help="Condition")
    single_parser.add_argument("--trial", required=True, help="Trial name")
    single_parser.add_argument("--segment", required=True, help="Segment name (e.g., femur_r)")
    single_parser.add_argument(
        "--orientation-results", required=True,
        help="Path to orientation optimization results JSON"
    )
    single_parser.add_argument("--output", required=True, help="Output directory")
    single_parser.add_argument("--imu-data-name", help="IMU data column name (default: inferred)")
    single_parser.add_argument("--max-frames", type=int, default=1000, help="Max frames (default: 1000)")
    single_parser.add_argument(
        "--method", default="differential_evolution",
        choices=["differential_evolution", "nelder-mead"],
        help="Optimization method (default: differential_evolution)"
    )
    
    # Batch optimization parser
    batch_parser = subparsers.add_parser('batch', help='Run batch position optimization')
    batch_parser.add_argument("--dataset", required=True, help="Path to Final dataset root")
    batch_parser.add_argument("--subject", required=True, help="Subject ID")
    batch_parser.add_argument("--condition", required=True, help="Condition")
    batch_parser.add_argument("--trial", required=True, help="Trial name")
    batch_parser.add_argument(
        "--orientation-base", required=True,
        help="Base directory containing orientation results (with combined_results.json)"
    )
    batch_parser.add_argument("--output", required=True, help="Output base directory")
    batch_parser.add_argument("--max-frames", type=int, default=1000, help="Max frames (default: 1000)")
    batch_parser.add_argument(
        "--method", default="differential_evolution",
        choices=["differential_evolution", "nelder-mead"],
        help="Optimization method (default: differential_evolution)"
    )
    batch_parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue processing other segments if one fails"
    )
    
    # If no command specified, show help or run single
    args = parser.parse_args()
    
    if args.command == 'single':
        run_single_optimization(args)
    elif args.command == 'batch':
        run_batch_optimization(args)
    elif args.command is None:
        # No subcommand - treat as single mode with required args
        parser.print_help()
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

