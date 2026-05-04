#!/usr/bin/env python3
"""
Splits the generated System1 dataset into Train and Test sets.
Uses a stratified split based on the taxonomy to ensure both sets
have a balanced representation of all error types.
"""

import json
import os
import random
from collections import defaultdict

DATA_DIR = "data/System1"
INPUT_FILE = os.path.join(DATA_DIR, "generated_errors.json")
TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
TEST_FILE = os.path.join(DATA_DIR, "test.json")

TEST_SPLIT_RATIO = 0.1  # 10% for testing (~200 samples)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # Group by taxonomy for stratified splitting
    grouped_data = defaultdict(list)
    for item in data:
        grouped_data[item["taxonomy"]].append(item)

    train_set = []
    test_set = []

    # Shuffle and split each taxonomy group
    random.seed(42)  # For reproducibility
    for taxonomy, items in grouped_data.items():
        random.shuffle(items)
        split_idx = int(len(items) * (1 - TEST_SPLIT_RATIO))
        
        train_set.extend(items[:split_idx])
        test_set.extend(items[split_idx:])

    # Shuffle the final sets so taxonomies are mixed
    random.shuffle(train_set)
    random.shuffle(test_set)

    # Save the splits
    with open(TRAIN_FILE, "w") as f:
        json.dump(train_set, f, indent=2)
    
    with open(TEST_FILE, "w") as f:
        json.dump(test_set, f, indent=2)

    print("========================================")
    print("Dataset Split Complete!")
    print("========================================")
    print(f"Total original samples: {len(data)}")
    print(f"Train set: {len(train_set)} samples -> {TRAIN_FILE}")
    print(f"Test set:  {len(test_set)} samples -> {TEST_FILE}")
    print("\nTaxonomy distribution in Test Set:")
    
    test_taxonomies = defaultdict(int)
    for item in test_set:
        test_taxonomies[item["taxonomy"]] += 1
        
    for tax, count in test_taxonomies.items():
        print(f"  - {tax}: {count}")

if __name__ == "__main__":
    main()
