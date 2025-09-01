#!/bin/bash

# Memory-optimized training script
# This script sets up the environment for running the neural IR training with memory optimizations

echo "Setting up memory-optimized environment for training..."

# Set PyTorch memory allocator configuration
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Limit PyTorch data loader workers to reduce memory pressure
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Set Python hash seed for reproducibility
export PYTHONHASHSEED=42

# Clear GPU memory before starting
echo "Clearing GPU memory..."
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU memory cleared')"

# Check available memory
echo "Checking system resources..."
nvidia-smi
echo "Free disk space:"
df -h /

# Kill any existing Python processes that might be using GPU memory
echo "Cleaning up any existing GPU processes..."
pkill -f python || true
sleep 2

# Run the training
echo "Starting training with memory optimizations..."
cd /workspace/2404170001/tas_project_3

# Use python with memory optimizations
exec python3 -u train.py 2>&1 | tee training.log
