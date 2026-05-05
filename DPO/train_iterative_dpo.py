#!/usr/bin/env python3
"""
Iterative Online DPO (Conceptual/Skeleton)
This script demonstrates how you would transition from offline to online DPO.
Here, our "environment" is a proxy that checks if the generated command 
matches the dataset's known solutions.
"""

import sys
import os
import json
import torch
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.load_model import load_model

# Configuration
BASE_MODEL_VERSION = "base"
TRAIN_DATA_SOURCE = "data/System1/train.json"
OUTPUT_DIR = "DPO/checkpoints/online-dpo-v1"

def load_proxy_environment(path):
    """Loads the original dataset to act as our proxy 'environment' oracle."""
    with open(path, "r") as f:
        return json.load(f)

def run_iterative_loop():
    # 1. Load Model & Tokenizer
    print("Loading model...")
    model, tokenizer = load_model(BASE_MODEL_VERSION, merge=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # 2. Apply LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 3. Load Proxy Environment
    env_data = load_proxy_environment(TRAIN_DATA_SOURCE)
    
    # 4. ITERATIVE ONLINE LOOP
    epochs = 3
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        dynamic_dpo_data = []

        # Step 4a: Generate and get feedback (Proxy interaction)
        print("Generating candidate solutions and interacting with proxy environment...")
        for item in env_data[:50]:  # Using a subset for demonstration
            original_cmd = item["interaction"]["original_command"]
            error_output = item["interaction"]["error_output"]
            
            prompt = f"Original Command:\n{original_cmd}\nError Output:\n{error_output}\nRespond with the bash command to fix."
            
            # --- GENERATION STEP (Agent predicts) ---
            # In a real setup, you use model.generate() here!
            # For this wrapper skeleton, we assume it generates `generated_cmd`
            generated_cmd = "sudo " + original_cmd # simulated generation
            
            # --- PROXY ENVIRONMENT FEEDBACK ---
            # We look up what the actual Rank 1 (best) and Rank 3 (worst) were in our dataset
            best_cmd = next((c["command"] for c in item.get("candidates", []) if c.get("rank") == 1), None)
            worst_cmd = next((c["command"] for c in item.get("candidates", []) if c.get("rank") >= 2), None)
            
            if not best_cmd or not worst_cmd: continue

            # Evaluate: If the model's generated command matches our 'best', we keep it as Chosen.
            # If it failed, it becomes our Rejected!
            if generated_cmd.strip() == best_cmd.strip():
                chosen = generated_cmd
                rejected = worst_cmd
            else:
                # Model failed to generate the perfect command.
                # So we tell DPO: "What the model generated is rejected. The optimal dataset answer is chosen."
                chosen = best_cmd
                rejected = generated_cmd 
            
            dynamic_dpo_data.append({
                "prompt": [{"role": "user", "content": prompt}],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}]
            })

        # Step 4b: Train immediately on this newly gathered experience
        if not dynamic_dpo_data:
            continue
            
        print(f"Training on {len(dynamic_dpo_data)} dynamically gathered samples...")
        
        # Format dataset
        def format_ds(row):
            return {
                "prompt": tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True),
                "chosen": row["chosen"][0]["content"] + tokenizer.eos_token,
                "rejected": row["rejected"][0]["content"] + tokenizer.eos_token
            }
            
        train_dataset = Dataset.from_list(dynamic_dpo_data).map(format_ds)
        
        training_args = DPOConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1, # Just 1 epoch per iteration
            per_device_train_batch_size=2,
            learning_rate=5e-5,
            remove_unused_columns=False,
            report_to="none"
        )
        
        trainer = DPOTrainer(
            model=model,
            ref_model=None, 
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        
        trainer.train()
        
    print("Saving online DPO model...")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    run_iterative_loop()
