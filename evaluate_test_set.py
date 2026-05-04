#!/usr/bin/env python3
"""
Evaluates the Base Llama 3.1 8B model on the Test Set to establish a baseline.
It asks the model to predict the fix command for each error scenario and calculates
the exact match accuracy against the gold standard (fix_B).

Usage:
    sbatch run_inference.sh evaluate_test_set.py --version base
"""

import argparse
import sys
import os
import json
import time
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.load_model import load_model

# Paths
TEST_FILE = "data/System1/test.json"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = f"data/System1/analysis_output/baseline_System1_no_state_{TIMESTAMP}.json"


def build_prompt(item):
    return f"""You are an expert Linux system administrator.
A user tried to run a command and encountered an error.
Your task is to provide the EXACT bash command to fix the issue.

Original Command:
{item['interaction']['original_command']}

Error Output:
{item['interaction']['error_output']}

Respond with ONLY the exact bash command to fix the error. Do not include any explanations, markdown formatting, or code blocks."""


def generate_json(model, tokenizer, prompt, max_new_tokens=256):
    import torch
    messages = [
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temp for deterministic evaluation
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_response(raw_text: str):
    # Remove markdown backticks if the model ignores the instruction
    cleaned = raw_text.strip()
    if cleaned.startswith("```bash"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
        
    return cleaned.strip()

def jaccard_similarity(cmd1: str, cmd2: str) -> float:
    set1 = set(cmd1.strip().split())
    set2 = set(cmd2.strip().split())
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline Model")
    parser.add_argument("--version", "-v", default="base", help="Model version ID")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of test samples (for quick testing)")
    args = parser.parse_args()

    if not os.path.exists(TEST_FILE):
        print(f"Error: {TEST_FILE} not found!")
        sys.exit(1)

    with open(TEST_FILE, "r") as f:
        test_data = json.load(f)

    if args.limit > 0:
        test_data = test_data[:args.limit]

    print(f"Loading model version: {args.version}...")
    model, tokenizer = load_model(args.version)
    model.eval()
    
    import torch
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU detected! Exiting.")
        sys.exit(1)

    print(f"\nStarting evaluation on {len(test_data)} samples...")
    
    results = []
    
    # Track statistics
    stats = {
        1: {"exact": 0, "jaccard_sum": 0.0, "substring": 0},
        2: {"exact": 0, "jaccard_sum": 0.0, "substring": 0},
        3: {"exact": 0, "jaccard_sum": 0.0, "substring": 0}
    }
    
    start_time = time.time()
    
    for item in tqdm(test_data):
        prompt = build_prompt(item)
        raw_text = generate_json(model, tokenizer, prompt, max_new_tokens=100)
        predicted_cmd = parse_response(raw_text)
        pred_norm = predicted_cmd.strip()
        
        matches = {}
        for c in item["candidates"]:
            gold_norm = c["command"].strip()
            rank = c.get("rank", 3)
            
            jaccard = jaccard_similarity(pred_norm, gold_norm)
            exact = (pred_norm == gold_norm)
            substring = (gold_norm in pred_norm)
            
            matches[f"rank_{rank}"] = {
                "id": c["id"],
                "type": c.get("type", "unknown"),
                "command": gold_norm,
                "jaccard": jaccard,
                "exact": exact,
                "is_substring": substring
            }
            
            if exact:
                stats[rank]["exact"] += 1
            if substring:
                stats[rank]["substring"] += 1
            stats[rank]["jaccard_sum"] += jaccard
        
        entry = {
            "id": item["id"],
            "taxonomy": item["taxonomy"],
            "original_command": item["interaction"]["original_command"],
            "predicted_command": predicted_cmd,
            "raw_output": raw_text,
            "matches": matches
        }
                
        results.append(entry)

    elapsed = time.time() - start_time
    
    # Save results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    total = len(test_data)
    print("\n========================================")
    print("EVALUATION COMPLETE")
    print("========================================")
    print(f"Total evaluated:   {total}")
    for rank in [1, 2, 3]:
        exact_pct = (stats[rank]['exact'] / total) * 100
        substring_pct = (stats[rank]['substring'] / total) * 100
        avg_jaccard = (stats[rank]['jaccard_sum'] / total) * 100
        print(f"--- RANK {rank} Matches ---")
        print(f"  Exact Match Acc:   {stats[rank]['exact']}/{total} ({exact_pct:.1f}%)")
        print(f"  Substring Acc:     {stats[rank]['substring']}/{total} ({substring_pct:.1f}%)")
        print(f"  Average Jaccard:   {avg_jaccard:.1f}%")
        
    print(f"\nTime elapsed:      {elapsed/60:.1f} minutes")
    print(f"Results saved to:  {RESULTS_FILE}")

if __name__ == "__main__":
    main()
