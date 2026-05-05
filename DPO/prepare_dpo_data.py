#!/usr/bin/env python3
"""
Converts train.json into chat-formatted JSONL for DPO training.
Splits into train/val sets (90/10).

Usage:
    python3 DPO/prepare_dpo_data.py
"""

import json
import os
import random

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_FILE = "data/System1/train.json"
OUTPUT_DIR = "DPO/data"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "dpo_train.jsonl")
VAL_FILE = os.path.join(OUTPUT_DIR, "dpo_val.jsonl")
VAL_RATIO = 0.1  # 10% for validation
SEED = 42
# ==========================================

SYSTEM_PROMPT = "You are an expert Linux system administrator. Given a failed command and its error output, respond with ONLY the exact bash command to fix the issue."

def build_messages(item):
    """Convert a single data item into DPO format (prompt, chosen, rejected)."""
    original_cmd = item["interaction"]["original_command"]
    error_output = item["interaction"]["error_output"]

    # Extract chosen (Rank 1) and rejected (Rank 3 or 2)
    chosen_cmd = None
    rejected_cmd = None

    candidates = item.get("candidates", [])
    
    # Try to find by rank
    for c in candidates:
        if c.get("rank") == 1:
            chosen_cmd = c["command"]
        elif c.get("rank") == 3:
            rejected_cmd = c["command"]
            
    # Fallback to rank 2 if rank 3 is not available
    if rejected_cmd is None:
        for c in candidates:
            if c.get("rank") == 2:
                rejected_cmd = c["command"]
                break

    # If we couldn't determine both, skip this sample
    if chosen_cmd is None or rejected_cmd is None:
        return None

    user_content = f"""A user tried to run a command and encountered an error.
Your task is to provide the EXACT bash command to fix the issue.

Original Command:
{original_cmd}

Error Output:
{error_output}

Respond with ONLY the exact bash command to fix the error."""

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    chosen_message = [{"role": "assistant", "content": chosen_cmd.strip()}]
    rejected_message = [{"role": "assistant", "content": rejected_cmd.strip()}]

    return {
        "prompt": prompt_messages,
        "chosen": chosen_message,
        "rejected": rejected_message
    }

def main():
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: {SOURCE_FILE} not found!")
        return

    with open(SOURCE_FILE, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {SOURCE_FILE}")

    converted = []
    skipped = 0
    for item in data:
        result = build_messages(item)
        if result is not None:
            converted.append(result)
        else:
            skipped += 1

    print(f"Converted: {len(converted)} | Skipped: {skipped}")

    random.seed(SEED)
    random.shuffle(converted)

    val_size = int(len(converted) * VAL_RATIO)
    val_data = converted[:val_size]
    train_data = converted[val_size:]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(TRAIN_FILE, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(VAL_FILE, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nTrain samples: {len(train_data)} → {TRAIN_FILE}")
    print(f"Val samples:   {len(val_data)} → {VAL_FILE}")

    # Print a sample
    if train_data:
        sample = train_data[0]
        print("\n--- Sample Entry ---")
        print(f"Prompt System: {sample['prompt'][0]['content'][:80]}...")
        print(f"Prompt User:   {sample['prompt'][1]['content'][:80]}...")
        print(f"Chosen:        {sample['chosen'][0]['content']}")
        print(f"Rejected:      {sample['rejected'][0]['content']}")

if __name__ == "__main__":
    main()
