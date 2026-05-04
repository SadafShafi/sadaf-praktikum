#!/usr/bin/env python3
"""
Converts train.json into chat-formatted JSONL for SFT training.
Splits into train/val sets (90/10).

Usage:
    python3 SFT/prepare_sft_data.py
"""

import json
import os
import random

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_FILE = "data/System1/train.json"
OUTPUT_DIR = "SFT/data"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "sft_train.jsonl")
VAL_FILE = os.path.join(OUTPUT_DIR, "sft_val.jsonl")
VAL_RATIO = 0.1  # 10% for validation
SEED = 42
# ==========================================

SYSTEM_PROMPT = "You are an expert Linux system administrator. Given a failed command and its error output, respond with ONLY the exact bash command to fix the issue."


def build_messages(item):
    """Convert a single data item into chat messages for SFT."""
    original_cmd = item["interaction"]["original_command"]
    error_output = item["interaction"]["error_output"]

    # Find the Rank 1 (best_practice) command
    rank1_cmd = None
    for c in item["candidates"]:
        if c.get("rank") == 1:
            rank1_cmd = c["command"]
            break
    
    # Fallback: use fix_B if no rank field
    if rank1_cmd is None:
        for c in item["candidates"]:
            if c["id"] == "fix_B":
                rank1_cmd = c["command"]
                break

    if rank1_cmd is None:
        return None  # Skip if we can't find a target

    user_content = f"""A user tried to run a command and encountered an error.
Your task is to provide the EXACT bash command to fix the issue.

Original Command:
{original_cmd}

Error Output:
{error_output}

Respond with ONLY the exact bash command to fix the error."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": rank1_cmd.strip()}
    ]

    return {"messages": messages}


def main():
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: {SOURCE_FILE} not found!")
        return

    with open(SOURCE_FILE, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {SOURCE_FILE}")

    # Convert to chat format
    converted = []
    skipped = 0
    for item in data:
        result = build_messages(item)
        if result is not None:
            converted.append(result)
        else:
            skipped += 1

    print(f"Converted: {len(converted)} | Skipped: {skipped}")

    # Shuffle and split
    random.seed(SEED)
    random.shuffle(converted)

    val_size = int(len(converted) * VAL_RATIO)
    val_data = converted[:val_size]
    train_data = converted[val_size:]

    # Write JSONL files
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(TRAIN_FILE, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(VAL_FILE, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nTrain samples: {len(train_data)} → {TRAIN_FILE}")
    print(f"Val samples:   {len(val_data)} → {VAL_FILE}")

    # Print a sample for verification
    sample = train_data[0]
    print("\n--- Sample Entry ---")
    print(f"System: {sample['messages'][0]['content'][:80]}...")
    print(f"User:   {sample['messages'][1]['content'][:80]}...")
    print(f"Assist: {sample['messages'][2]['content']}")


if __name__ == "__main__":
    main()
