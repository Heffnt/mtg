# MTG Draft Training

Teaching language models to truly understand Magic: The Gathering.

**Authors**: Tom Heffernan & Claude

## Vision

Magic: The Gathering has over 25,000 unique cards spanning 30+ years of design. Each card is a complex artifact with mechanics, rules interactions, flavor text, and strategic implications. This project aims to imbue large language models with deep, comprehensive MTG knowledge through full parameter fine-tuning.

### Why This Matters

Modern LLMs have limited and often outdated knowledge of MTG cards. They hallucinate card names, confuse mechanics, and lack the nuanced understanding that makes MTG strategically rich. This project seeks to create models that:

- **Know every card**: From Alpha to the latest set, with accurate oracle text
- **Understand interactions**: Rules, combos, synergies, and edge cases
- **Support players**: Draft advice, deck building, rules questions
- **Preserve history**: MTG's evolution as both game and cultural phenomenon

### The Approach

This isn't LoRA or parameter-efficient fine-tuning. This is **full fine-tuning** of 32B+ parameter models using 2.7M+ training examples. We're teaching the model to internalize MTG knowledge at a fundamental level, updating every parameter to create true domain expertise.

Think of it as the difference between memorizing flash cards versus years of playing the game.

## Technical Highlights

- **Full parameter fine-tuning**: All 32.8B parameters are updated, not just adapter layers
- **Massive scale**: 2.7M+ training examples across train/validation/test splits
- **Production-grade training**: DeepSpeed ZeRO-3 for efficient multi-GPU parallelism
- **Modern architecture**: Built on Unsloth for 2x faster training with lower memory usage
- **Reproducible**: Complete environment specification and training configuration
- **Observable**: Full WandB integration for real-time monitoring and analysis

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

### Dataset

The training dataset contains **2.7M+ examples** teaching comprehensive MTG knowledge:

- **Card descriptions**: Complete oracle text for every card
- **Rules interactions**: How mechanics work together
- **Strategic context**: When and why to use cards
- **Historical knowledge**: Set information, rarity, artist details

The dataset is split into:
- `splits_full/train.jsonl` - 2.73M training examples
- `splits_full/valid.jsonl` - 27.8K validation examples
- `splits_full/test.jsonl` - 27.8K test examples

### Data Format

The training script expects data in JSONL format with ChatML structure:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert Magic: The Gathering player..."},
    {"role": "user", "content": "Describe Markov Waltzer."},
    {"role": "assistant", "content": "Markov Waltzer {2}{R}{W}\nUncommon\nCreature — Vampire\n\nFlying, haste\nAt the beginning of combat on your turn, up to two target creatures you control each get +1/+0 until end of turn.\n1/3"}
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

## Future Directions

This project is a foundation for deeper MTG-AI integration:

- **Draft assistants**: Real-time advice during drafts based on pack contents and deck synergies
- **Rules engines**: Automated judge calls and comprehensive rules explanations
- **Deck builders**: AI that understands meta, budget constraints, and playstyle preferences
- **Historical analysis**: Tracking how cards, mechanics, and strategies evolved over 30 years
- **Accessibility**: Making MTG knowledge available to new players and veterans alike

The goal isn't to replace human creativity or expertise, but to make MTG knowledge more accessible and help players engage more deeply with the game they love.

## Contributing

This is an open project. If you have ideas for improving the training data, architecture, or applications, contributions are welcome. The codebase is designed to be modular and hackable.

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

