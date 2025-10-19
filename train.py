"""
Unsloth Full Fine-Tuning Script for MTG Card Training
Supports both single-GPU and multi-GPU training
"""

import json
import os
from datasets import Dataset
import wandb
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import torch

# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================

# GPU Configuration
NUM_GPUS = 1  # Set to 1 for single GPU, 8 for multi-GPU, or None for auto-detect
AUTO_DETECT_GPUS = NUM_GPUS is None

# Model Configuration
MODEL_NAME = "qihoo360/TinyR1-32B"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # Auto-detect (will use bfloat16 for H200)
LOAD_IN_4BIT = False  # Set to False for full fine-tuning

# Data Configuration
TRAIN_DATA_PATH = "splits_full/train.jsonl"

# WandB Configuration
WANDB_ENTITY = "heffnt-worcester-polytechnic-institute"
WANDB_PROJECT = "mtg-draft-training"
WANDB_RUN_NAME = None  # Auto-generate if None

# Training Hyperparameters (defaults for single GPU; auto-adjusted for multi-GPU)
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 32  # Default: single GPU (effective batch ~64)
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Training Optimization
USE_GRADIENT_CHECKPOINTING = True
BF16 = True  # Use bfloat16 precision
FP16 = False

# DeepSpeed Configuration
USE_DEEPSPEED = True  # Enable DeepSpeed ZeRO-3
CPU_OFFLOAD = True  # Default: True for single GPU, auto-adjusted for multi-GPU

# Checkpoint & Logging
OUTPUT_DIR = "outputs"
LOGGING_STEPS = 10
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 3

# ============================================================================
# DATA LOADING
# ============================================================================

def load_jsonl(file_path):
    """Load JSONL file with ChatML formatted data."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_chat_template(example):
    """
    Format the messages into a single training text.
    Assumes the tokenizer has a chat template.
    """
    return {"text": example["messages"]}

def prepare_dataset():
    """Load and prepare the training dataset."""
    print(f"Loading training data from {TRAIN_DATA_PATH}...")
    train_data = load_jsonl(TRAIN_DATA_PATH)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(train_data)
    
    # Apply formatting
    dataset = dataset.map(format_chat_template)
    
    print(f"Loaded {len(dataset)} training examples")
    return dataset

# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model_and_tokenizer():
    """Load model and tokenizer using Unsloth for full fine-tuning."""
    print(f"Loading model: {MODEL_NAME}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # Configure for full fine-tuning (no LoRA adapters)
    model = FastLanguageModel.get_peft_model(
        model,
        r=0,  # r=0 means full fine-tuning, not LoRA
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=0,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth" if USE_GRADIENT_CHECKPOINTING else False,
        random_state=3407,
    )
    
    # Ensure tokenizer has chat template
    if tokenizer.chat_template is None:
        # Set a default chat template if none exists
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n' }}{% endif %}{% endfor %}"
    
    return model, tokenizer

# ============================================================================
# GPU DETECTION AND AUTO-CONFIGURATION
# ============================================================================

def get_gpu_config():
    """Detect GPU count and adjust training configuration."""
    global NUM_GPUS, GRADIENT_ACCUMULATION_STEPS, CPU_OFFLOAD

    if AUTO_DETECT_GPUS:
        NUM_GPUS = torch.cuda.device_count()
        print(f"Auto-detected {NUM_GPUS} GPU(s)")
    else:
        print(f"Using configured {NUM_GPUS} GPU(s)")

    # Auto-adjust gradient accumulation for multi-GPU to maintain effective batch size ~64
    # Single GPU: batch=2, grad_accum=32 -> effective=64
    # 8 GPUs: batch=2, grad_accum=4 -> effective=64
    if NUM_GPUS > 1:
        GRADIENT_ACCUMULATION_STEPS = max(1, 32 // NUM_GPUS)
        CPU_OFFLOAD = False  # Keep on GPU for multi-GPU
        print(f"Multi-GPU detected: Adjusted gradient_accumulation_steps to {GRADIENT_ACCUMULATION_STEPS}")
        print(f"Multi-GPU detected: Disabled CPU offloading")
    else:
        print(f"Single GPU: gradient_accumulation_steps={GRADIENT_ACCUMULATION_STEPS}")
        print(f"Single GPU: CPU offloading={'enabled' if CPU_OFFLOAD else 'disabled'}")

    return NUM_GPUS

# ============================================================================
# DEEPSPEED CONFIGURATION
# ============================================================================

def create_deepspeed_config():
    """Create DeepSpeed ZeRO-3 configuration, adapting to single or multi-GPU."""
    if not USE_DEEPSPEED:
        return None

    # Configure offloading based on CPU_OFFLOAD setting
    if CPU_OFFLOAD:
        # Single GPU: offload to CPU to save VRAM
        offload_config = {"device": "cpu", "pin_memory": True}
    else:
        # Multi-GPU: keep on GPU for performance
        offload_config = {"device": "none"}

    deepspeed_config = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": offload_config,
            "offload_param": offload_config,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
        },
        "bf16": {
            "enabled": BF16
        },
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "gradient_clipping": MAX_GRAD_NORM,
        "steps_per_print": LOGGING_STEPS,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }

    # Save config to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config_path = os.path.join(OUTPUT_DIR, "ds_config.json")
    with open(config_path, 'w') as f:
        json.dump(deepspeed_config, f, indent=2)

    print(f"DeepSpeed config saved to {config_path}")
    return config_path

# ============================================================================
# TRAINING
# ============================================================================

def train():
    """Main training function."""
    # Detect and configure GPU setup
    num_gpus = get_gpu_config()

    # Initialize WandB
    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model": MODEL_NAME,
            "max_seq_length": MAX_SEQ_LENGTH,
            "num_gpus": num_gpus,
            "per_device_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * num_gpus,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_TRAIN_EPOCHS,
            "warmup_steps": WARMUP_STEPS,
            "cpu_offload": CPU_OFFLOAD,
        }
    )

    # Load dataset
    dataset = prepare_dataset()

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # Create DeepSpeed config if enabled
    deepspeed_config = create_deepspeed_config()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        
        # Precision
        bf16=BF16,
        fp16=FP16,
        
        # Logging
        logging_steps=LOGGING_STEPS,
        logging_dir=f"{OUTPUT_DIR}/logs",
        report_to="wandb",
        
        # Checkpointing
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        
        # DeepSpeed
        deepspeed=deepspeed_config if USE_DEEPSPEED else None,
        
        # Other
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if USE_DEEPSPEED else None,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,  # Disable packing for chat format
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    
    print("Training complete!")
    wandb.finish()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    train()

