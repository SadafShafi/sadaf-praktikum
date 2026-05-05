#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) training script for Llama 3.1 8B.
Uses LoRA via PEFT + trl DPOTrainer.

Usage:
    sbatch DPO/run_dpo.sh
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
TRAIN_FILE = "DPO/data/dpo_train.jsonl"
VAL_FILE = "DPO/data/dpo_val.jsonl"
OUTPUT_DIR = "DPO/checkpoints/dpo-v1"
LOG_DIR = "DPO/logs"

# Base model to start from (we align the base model directly)
# We use "base" from the registry
BASE_MODEL_VERSION = "base"

# LoRA Hyperparameters (applied on top of the base model)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# DPO Hyperparameters
BETA = 0.1  # The beta factor in DPO loss

# Training Hyperparameters
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5  # Usually lower than SFT
PER_DEVICE_BATCH_SIZE = 2  # DPO needs 2x memory because of reference model
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 512
MAX_PROMPT_LENGTH = 256
WARMUP_RATIO = 0.1
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
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected! Exiting.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load base model for DPO
    print(f"\nLoading model version {BASE_MODEL_VERSION}...")
    # NOTE: DPO requires a reference model. PEFT + DPOTrainer is smart enough 
    # to use the disabled adapter as the reference model if we load the base 
    # model and add an adapter. 
    # We load the base model, then we will add a new adapter for DPO.
    model, tokenizer = load_model(BASE_MODEL_VERSION, merge=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

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
    # The reference model will be the original base model (without this new adapter)
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

    # Format the dataset for DPOTrainer
    # DPOTrainer can take chat formatted lists directly and format them if tokenizer is provided
    # Let's pre-format them to text using the tokenizer's chat template
    def process(row):
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        # For chosen and rejected, we just extract the text and add EOS token
        chosen = row["chosen"][0]["content"] + tokenizer.eos_token
        rejected = row["rejected"][0]["content"] + tokenizer.eos_token
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }

    print("Formatting datasets...")
    train_dataset = train_dataset.map(process)
    val_dataset = val_dataset.map(process)

    # Training arguments
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    training_args = DPOConfig(
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
        report_to="none",
        gradient_checkpointing=True,
        beta=BETA,
        max_length=MAX_SEQ_LENGTH,
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT handles the reference model automatically when using peft_config or when model is a PeftModel
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\n========================================")
    print("STARTING DPO TRAINING")
    print("========================================")
    trainer.train()

    # Save final adapter
    print(f"\nSaving adapter to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Register in model registry
    register_version(
        version_id="dpo-v1",
        stage="dpo",
        path=OUTPUT_DIR,
        parent=BASE_MODEL_VERSION,
        method="LoRA",
        dataset="DPO/data/dpo_train.jsonl",
        key_hparams={
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "beta": BETA,
            "epochs": NUM_EPOCHS,
            "lr": LEARNING_RATE,
            "batch_size": PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        },
        notes="DPO alignment: LoRA fine-tuning directly on top of the base model using Rank 1 vs Rank 3 preferences."
    )

    print("\n========================================")
    print("DPO TRAINING COMPLETE")
    print(f"Adapter saved to: {OUTPUT_DIR}")
    print("Registered as version: dpo-v1")
    print("========================================")


if __name__ == "__main__":
    main()
