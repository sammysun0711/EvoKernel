#!/bin/bash
# EvoKernel Demo: 5-step depthwise conv3d optimization journey
# Requires: AMD MI300X/MI350X (gfx942/gfx950), ROCm, PyTorch with ROCm
set -e
cd "$(dirname "$0")"
echo "EvoKernel: AI reads 10 GPU kernels, beats the hand-tuned best"
echo ""
python benchmark.py "$@"
