# MTG Draft Training

Full fine-tuning of LLMs for Magic: The Gathering card knowledge using Unsloth.

## Hardware Requirements

This training script is optimized for:
- **8x NVIDIA H200 GPUs** (140GB VRAM each)
- Uses DeepSpeed ZeRO-3 for distributed training across all GPUs
- Total model size: ~32.8B parameters (BF16)

Can be adapted for different GPU configurations by adjusting batch size and gradient accumulation steps in `train.py`.

## Installation

This project uses conda for environment management, configured via `environment.yml`.

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate mtgenv
```

The environment is configured for:
- Python 3.11+
- PyTorch with CUDA 12.4 (compatible with CUDA 12.7 drivers on Turing HPC)
- All training dependencies (transformers, datasets, DeepSpeed, Unsloth, etc.)

### 2. Login to WandB

```bash
wandb login
```

You'll need to authenticate with your WandB API key.

## Usage

### Quick Start

Simply run the training script:

```bash
# On a single node with 8 GPUs
python train.py
```

For multi-node training with DeepSpeed:

```bash
deepspeed --num_gpus=8 train.py
```

### Configuration

All hyperparameters are configured at the top of `train.py`. Key settings:

**Model Settings:**
- `MODEL_NAME`: HuggingFace model to fine-tune (default: `qihoo360/TinyR1-32B`)
- `MAX_SEQ_LENGTH`: Maximum sequence length (default: 2048)

**Training Hyperparameters:**
- `PER_DEVICE_TRAIN_BATCH_SIZE`: Batch size per GPU (default: 2)
- `GRADIENT_ACCUMULATION_STEPS`: Gradient accumulation (default: 4)
  - Effective batch size = 2 × 8 GPUs × 4 = 64
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `NUM_TRAIN_EPOCHS`: Number of training epochs (default: 3)

**WandB Logging:**
- `WANDB_ENTITY`: WandB entity name
- `WANDB_PROJECT`: WandB project name
- `WANDB_RUN_NAME`: Optional run name (auto-generated if None)

**Optimization:**
- `USE_GRADIENT_CHECKPOINTING`: Enable gradient checkpointing (default: True)
- `BF16`: Use bfloat16 precision (default: True)
- `USE_DEEPSPEED`: Enable DeepSpeed ZeRO-3 (default: True)

### Data Format

The training script expects data in JSONL format with ChatML structure:

```json
{
  "messages": [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User question"},
    {"role": "assistant", "content": "Assistant response"}
  ]
}
```

Default data path: `splits_full/train.jsonl`

## Outputs

Training outputs are saved to the `outputs/` directory:

- **Checkpoints**: Saved every 500 steps (configurable with `SAVE_STEPS`)
- **Final model**: Saved to `outputs/final_model/`
- **Logs**: TensorBoard logs in `outputs/logs/`
- **DeepSpeed config**: `outputs/ds_config.json`

## Monitoring

Training progress is logged to WandB in real-time. Access your dashboard at:
```
https://wandb.ai/heffnt-worcester-polytechnic-institute/mtg-draft-training
```

## Notes

- This script performs **full fine-tuning** (not LoRA), which updates all model parameters
- DeepSpeed ZeRO-3 distributes the model, optimizer states, and gradients across all 8 GPUs
- With 140GB VRAM per GPU, parameters are kept on GPU (no CPU offloading needed)
- Training time will depend on dataset size; expect several hours for large datasets

## Troubleshooting

**Out of Memory (OOM) errors:**
- Decrease `PER_DEVICE_TRAIN_BATCH_SIZE` (try 1)
- Increase `GRADIENT_ACCUMULATION_STEPS` to maintain effective batch size
- Reduce `MAX_SEQ_LENGTH`

**Slow training:**
- Increase `PER_DEVICE_TRAIN_BATCH_SIZE` if you have VRAM headroom
- Ensure DeepSpeed is properly configured (`USE_DEEPSPEED = True`)

**WandB not logging:**
- Ensure you've run `wandb login`
- Check `WANDB_ENTITY` and `WANDB_PROJECT` are correct

