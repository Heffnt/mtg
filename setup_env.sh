#!/bin/bash
set -e

echo "Creating conda environment with Python 3.12..."
conda env create -f environment.yml

echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate mtgenv

echo "Installing uv for fast package installation..."
pip install uv

echo "Installing PyTorch 2.5.1 with CUDA 12.1 support (Unsloth compatible)..."
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "Installing core ML/AI frameworks..."
uv pip install \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    wandb \
    deepspeed \
    bitsandbytes \
    einops \
    xformers

echo "Installing Unsloth..."
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

echo "Installing unsloth_zoo..."
uv pip install unsloth_zoo

echo ""
echo "========================================="
echo "Environment setup complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate mtgenv"
echo ""
echo "To test, run:"
echo "  python -c 'from unsloth import FastLanguageModel; print(\"Unsloth loaded successfully!\")'"
echo ""
