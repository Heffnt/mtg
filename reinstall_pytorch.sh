#!/bin/bash
# Run this in your mtgenv environment

echo "Removing all existing torch packages..."
pip uninstall -y torch torchvision torchaudio triton xformers 2>/dev/null || true
conda remove -y pytorch torchvision torchaudio pytorch-cuda 2>/dev/null || true

echo "Installing PyTorch 2.5.1 via conda (ensures binary compatibility)..."
conda install -y pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

echo "Testing installation..."
python -c "import torch; import torchvision; print(f'PyTorch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}')"

echo "If the test passed, reinstall unsloth:"
echo "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' --force-reinstall"
