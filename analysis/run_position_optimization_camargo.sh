#!/bin/bash
# Quick runner for Camargo processed dataset position optimization
# Usage: ./run_position_optimization_camargo.sh [segment]
# Example: ./run_position_optimization_camargo.sh femur_r

SEGMENT=${1:-femur_r}

python imu_position_optimization_camargo.py \
    --dataset-root "/Users/luorix/Desktop/MetaMobility Lab (CMU)/data/Camargo_processed" \
    --subject AB21 \
    --date 01_27_2019 \
    --condition treadmill \
    --trial treadmill_03_01 \
    --segment "$SEGMENT" \
    --orientation-results results/camargo_multisegment/${SEGMENT}/optimization_results.json \
    --output results/position_optimization_camargo/${SEGMENT}/ \
    --max-frames 1000 \
    --optimization-method two-stage

