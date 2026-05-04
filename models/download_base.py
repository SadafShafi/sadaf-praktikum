#!/usr/bin/env python3
"""
Download the base Llama 3.1 8B Instruct model from Hugging Face.

Prerequisites:
  1. Accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  2. Set your HF token:  export HF_TOKEN=hf_xxxx   (or add to .env)

Usage:
    python3 models/download_base.py
"""

import os
import sys
import json
from datetime import date
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN not found.")
    print("Set it via:  export HF_TOKEN=hf_xxxx")
    print("Or add HF_TOKEN=hf_xxxx to your .env file.")
    sys.exit(1)

# Use unsloth's ungated mirror (no Meta approval wait needed).
# Same weights, just no gating. Switch to "meta-llama/Llama-3.1-8B-Instruct"
# once your access request is approved if you prefer the official source.
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
DOWNLOAD_DIR = os.path.join("models", "base", "meta-llama-3.1-8b-instruct")
REGISTRY_PATH = os.path.join("models", "model_registry.json")

print(f"Downloading {MODEL_ID} → {DOWNLOAD_DIR}")
print("This is ~16 GB, it will take a while...\n")

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=DOWNLOAD_DIR,
    token=HF_TOKEN,
    ignore_patterns=["*.gguf", "*.bin", "original/*"],  # skip GGUF/bin duplicates
)

# Update the registry
if os.path.exists(REGISTRY_PATH):
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
else:
    registry = {"base": {}, "versions": []}

registry["base"] = {
    "path": DOWNLOAD_DIR,
    "source": MODEL_ID,
    "downloaded": str(date.today()),
}

with open(REGISTRY_PATH, "w") as f:
    json.dump(registry, f, indent=2)

print(f"\n✅ Download complete!")
print(f"   Model saved to: {DOWNLOAD_DIR}")
print(f"   Registry updated: {REGISTRY_PATH}")
