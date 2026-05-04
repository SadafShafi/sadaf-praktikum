#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) training script for Llama 3.1 8B.
Uses LoRA via PEFT + trl SFTTrainer.

Usage:
    sbatch SFT/run_sft.sh
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.load_model import load_model, register_version

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_FILE = "SFT/data/sft_train.jsonl"
VAL_FILE = "SFT/data/sft_val.jsonl"
OUTPUT_DIR = "SFT/checkpoints/sft-v1"
LOG_DIR = "SFT/logs"

# LoRA Hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training Hyperparameters
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 512
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
SAVE_STEPS = 50
LOGGING_STEPS = 10
# ==========================================


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def main():
    import torch
    from transformers import TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected! Exiting.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load base model
    print("\nLoading base model...")
    model, tokenizer = load_model("base")

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Set max length for truncation
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    # Apply LoRA
    print("\nApplying LoRA configuration...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print(f"\nLoading training data from {TRAIN_FILE}...")
    train_data = load_jsonl(TRAIN_FILE)
    val_data = load_jsonl(VAL_FILE)

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples:   {len(val_data)}")

    # Convert to HF Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Formatting function: apply chat template
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    # Training arguments
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOG_DIR,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="epoch",
        bf16=True,
        report_to="none",  # No wandb on air-gapped cluster
        gradient_checkpointing=True,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
    )

    # Train
    print("\n========================================")
    print("STARTING SFT TRAINING")
    print("========================================")
    trainer.train()

    # Save final adapter
    print(f"\nSaving adapter to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Register in model registry
    register_version(
        version_id="sft-v1",
        stage="sft",
        path=OUTPUT_DIR,
        parent="base",
        method="LoRA",
        dataset="data/System1/train.json",
        key_hparams={
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "epochs": NUM_EPOCHS,
            "lr": LEARNING_RATE,
            "batch_size": PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        },
        notes="SFT baseline: LoRA fine-tuning on System1 train set (Rank 1 commands only)"
    )

    print("\n========================================")
    print("SFT TRAINING COMPLETE")
    print(f"Adapter saved to: {OUTPUT_DIR}")
    print("Registered as version: sft-v1")
    print("========================================")


if __name__ == "__main__":
    main()
