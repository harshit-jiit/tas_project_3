# Use NVIDIA CUDA base image with Ubuntu (compatible with your CUDA 12.3)
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Update system and install essential packages
RUN apt-get update && apt-get install -y \
    # Basic system tools
    curl \
    wget \
    git \
    vim \
    nano \
    htop \
    tree \
    unzip \
    zip \
    build-essential \
    software-properties-common \
    # Network tools
    net-tools \
    iputils-ping \
    # Process management
    supervisor \
    tmux \
    screen \
    # Development tools
    cmake \
    pkg-config \
    # System libraries
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11 and make it the default
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv python3-pip && \
    # Remove any existing python symlinks
    rm -f /usr/bin/python /usr/bin/python3 && \
    # Create new symlinks pointing to python3.11
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    # Create pip symlink to point to the correct pip version
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    # Upgrade pip
    python3.11 -m pip install --upgrade pip

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda && \
    rm miniconda.sh && \
    /opt/miniconda/bin/conda clean -afy

# Install virtualenv and virtualenvwrapper for additional venv management
RUN python3.11 -m pip install --no-cache-dir virtualenv virtualenvwrapper

# Add conda to PATH
ENV PATH="/opt/miniconda/bin:$PATH"

# Initialize conda for bash
RUN conda init bash && \
    conda config --set auto_activate_base false
    
# Install Node.js (LTS)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs

# Install Docker CLI (static binary, no admin required)
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-24.0.7.tgz | \
    tar xzf - --strip-components=1 -C /usr/local/bin docker/docker

# Install PyTorch with CUDA 12.1 support (step by step)
RUN python3.11 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 

# Install other packages from PyPI (separate command)
RUN python3.11 -m pip install --no-cache-dir --ignore-installed \
    jupyter jupyterlab notebook ipywidgets 

# Install additional GPU monitoring tools
RUN python3.11 -m pip install --no-cache-dir \
    gpustat \
    nvidia-ml-py3 \
    py3nvml

# Set up supervisor configuration for keeping container running
RUN mkdir -p /var/log/supervisor

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY start-container.sh /usr/local/bin/start-container.sh
RUN chmod +x /usr/local/bin/start-container.sh

# Create all necessary directories with full permissions
RUN mkdir -p /workspace && \
    mkdir -p /root/.virtualenvs && \
    mkdir -p /root/projects && \
    mkdir -p /root/data && \
    mkdir -p /var/log/supervisor && \
    # Set permissive permissions on all directories
    chmod -R 777 /workspace && \
    chmod -R 777 /root && \
    chmod -R 777 /opt/miniconda && \
    chmod -R 777 /var/log/supervisor

# Set working directory
WORKDIR /workspace

# Set up shell environment for root
RUN echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc && \
    echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'alias gpu="nvidia-smi"' >> ~/.bashrc && \
    echo 'alias gpustat="gpustat -cp"' >> ~/.bashrc && \
    echo 'alias gpumem="nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu --format=csv"' >> ~/.bashrc && \
    echo 'alias gpufree="nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr"' >> ~/.bashrc && \
    echo 'alias python311="/usr/bin/python3.11"' >> ~/.bashrc && \
    echo 'echo "Python version: $(python --version)"' >> ~/.bashrc

# Set up conda environment for root
RUN echo 'export PATH="/opt/miniconda/bin:$PATH"' >> ~/.bashrc && \
    echo 'alias condaenv="conda env list"' >> ~/.bashrc && \
    echo 'alias condacreate="conda create -n"' >> ~/.bashrc && \
    echo 'alias condaactivate="conda activate"' >> ~/.bashrc && \
    /opt/miniconda/bin/conda init bash

# Set up Python virtual environments for root
RUN echo 'export WORKON_HOME=$HOME/.virtualenvs' >> ~/.bashrc && \
    echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.11' >> ~/.bashrc && \
    echo 'export VIRTUALENVWRAPPER_VIRTUALENV=/opt/miniconda/bin/virtualenv' >> ~/.bashrc && \
    echo 'source /opt/miniconda/bin/virtualenvwrapper.sh' >> ~/.bashrc && \
    echo 'alias venvlist="lsvirtualenv"' >> ~/.bashrc && \
    echo 'alias venvmake="mkvirtualenv"' >> ~/.bashrc && \
    echo 'alias venvwork="workon"' >> ~/.bashrc && \
    echo 'alias venvdeactivate="deactivate"' >> ~/.bashrc && \
    echo 'alias venvremove="rmvirtualenv"' >> ~/.bashrc && \
    echo 'alias venvcreate="python -m venv"' >> ~/.bashrc

# Set environment variables for PyTorch temp directories
ENV TMPDIR=/workspace/tmp
ENV TMP=/workspace/tmp
ENV TEMP=/workspace/tmp
ENV TORCH_HOME=/workspace/torch_cache
ENV PYTORCH_KERNEL_CACHE_PATH=/workspace/pytorch_kernel_cache

# Create PyTorch temp directories with full permissions
RUN mkdir -p /workspace/tmp && \
    mkdir -p /workspace/torch_cache && \
    mkdir -p /workspace/pytorch_kernel_cache && \
    chmod -R 777 /workspace/tmp && \
    chmod -R 777 /workspace/torch_cache && \
    chmod -R 777 /workspace/pytorch_kernel_cache

# Create an entrypoint script to ensure permissions are always correct
RUN echo '#!/bin/bash\n\
# Ensure all workspace directories have correct permissions\n\
chmod -R 777 /workspace 2>/dev/null || true\n\
\n\
# Create temp directories if they dont exist\n\
mkdir -p /workspace/tmp /workspace/torch_cache /workspace/pytorch_kernel_cache\n\
chmod -R 777 /workspace/tmp /workspace/torch_cache /workspace/pytorch_kernel_cache\n\
\n\
# Execute the original command\n\
exec "$@"\n' > /usr/local/bin/ensure-permissions.sh && \
    chmod +x /usr/local/bin/ensure-permissions.sh

# Expose common ports
EXPOSE 8888 8000 5000 3000 2026 2027 

# Use the permission-ensuring entrypoint (but stay as root)
ENTRYPOINT ["/usr/local/bin/ensure-permissions.sh"]

# Set the default command
CMD ["/usr/local/bin/start-container.sh"]

# Note: This container runs as root for development convenience
# This eliminates permission issues but sacrifices security
# Use only in trusted development environments