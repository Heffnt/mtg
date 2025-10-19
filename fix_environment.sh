#!/bin/bash
set -e

echo "============================================="
echo "Fixing PyTorch/Unsloth Environment"
echo "============================================="
echo ""

# Ensure we have uv
echo "Step 1: Installing uv..."
pip install -q uv

# Remove all torch packages
echo "Step 2: Removing all existing PyTorch packages..."
uv pip uninstall -y torch torchvision torchaudio triton torchao 2>/dev/null || true

# Install PyTorch 2.5.1 with version pinning
echo "Step 3: Installing PyTorch 2.5.1 and torchao 0.12.0 (compatible with Unsloth)..."
uv pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Pin torchao to version compatible with PyTorch 2.5.1
uv pip install torchao==0.12.0

# Verify PyTorch installation
echo ""
echo "Step 4: Verifying PyTorch installation..."
python -c "import torch; import torchvision; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ torchvision: {torchvision.__version__}')" || {
    echo "ERROR: PyTorch verification failed!"
    exit 1
}

# Install core dependencies without touching PyTorch
echo ""
echo "Step 5: Installing ML frameworks (without upgrading PyTorch)..."
uv pip install --no-deps \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    wandb \
    bitsandbytes \
    einops \
    sentencepiece \
    protobuf

# Install their dependencies (but not torch upgrades)
echo "Step 6: Installing dependencies..."
uv pip install \
    huggingface_hub \
    safetensors \
    pyyaml \
    regex \
    requests \
    packaging \
    numpy \
    filelock \
    fsspec \
    tokenizers \
    psutil \
    dill \
    multiprocess \
    pyarrow \
    pandas \
    xxhash \
    aiohttp

# Install unsloth WITHOUT allowing it to upgrade torch
echo ""
echo "Step 7: Installing Unsloth (preventing PyTorch upgrade)..."
uv pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
uv pip install --no-deps unsloth_zoo

# Install unsloth dependencies (except torch)
uv pip install \
    tyro \
    cut_cross_entropy \
    mistral_common \
    hf_transfer

# Final verification
echo ""
echo "============================================="
echo "Final Verification"
echo "============================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
python -c "from unsloth import FastLanguageModel; print('✓ Unsloth loaded successfully!')"

echo ""
echo "✓ Environment setup complete!"
echo ""
