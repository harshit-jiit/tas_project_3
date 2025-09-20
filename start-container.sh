#!/bin/bash

# Print system information
echo "=== Container Started ==="
echo "CUDA Version: $(nvcc --version 2>/dev/null || echo 'CUDA not found')"
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.free,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
  awk -F', ' '{printf "GPU %s: %s - Free: %s MB, Used: %s MB, Total: %s MB\n", $1, $2, $3, $4, $5}' || \
  echo "No GPU detected or nvidia-smi not available"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Python Version: $(python --version)"
echo "Working Directory: $(pwd)"
echo "User: $(whoami)"
echo "========================="

# Start supervisor
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf