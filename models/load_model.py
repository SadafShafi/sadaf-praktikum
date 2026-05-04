#!/usr/bin/env python3
"""
Load any model version (base, SFT, DPO, RL) by its registry ID.

Usage as CLI:
    python3 models/load_model.py base
    python3 models/load_model.py sft-v1
    python3 models/load_model.py dpo-v1 --merge

Usage as library:
    from models.load_model import load_model
    model, tokenizer = load_model("base")
    model, tokenizer = load_model("sft-v1")
    model, tokenizer = load_model("dpo-v1", merge=True)
"""

import argparse
import json
import os
import sys

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "model_registry.json")


def get_registry(registry_path=None):
    """Load the model registry."""
    path = registry_path or REGISTRY_PATH
    if not os.path.exists(path):
        print(f"Error: Registry not found at {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def list_versions(registry_path=None):
    """List all available model versions."""
    reg = get_registry(registry_path)
    print(f"\n{'ID':<20} {'Stage':<8} {'Parent':<15} {'Method':<10} {'Path'}")
    print("-" * 90)
    print(f"{'base':<20} {'base':<8} {'—':<15} {'—':<10} {reg['base'].get('path', 'NOT DOWNLOADED')}")
    for v in reg.get("versions", []):
        print(f"{v['id']:<20} {v['stage']:<8} {v.get('parent','—'):<15} {v.get('method','—'):<10} {v['path']}")
    print()


def load_model(version_id, registry_path=None, merge=False, device_map="auto",
               torch_dtype=None):
    """
    Load a model by its registry version ID.

    Args:
        version_id: "base", "sft-v1", "dpo-v1", "rl-v1", etc.
        registry_path: Path to model_registry.json (default: auto-detect).
        merge: If True, merge adapter weights into base (required for serving).
        device_map: Device placement strategy (default "auto").
        torch_dtype: Override dtype (default: bfloat16).

    Returns:
        (model, tokenizer) tuple
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    reg = get_registry(registry_path)
    base_path = reg["base"]["path"]

    if not os.path.exists(base_path):
        print(f"Error: Base model not found at '{base_path}'.")
        print("Run:  python3 models/download_base.py")
        sys.exit(1)

    # --- Load base model ---
    if version_id == "base":
        print(f"Loading base model from {base_path}...")
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=True,
        )
        print("✅ Base model loaded.")
        return model, tokenizer

    # --- Find the requested version ---
    version = None
    for v in reg.get("versions", []):
        if v["id"] == version_id:
            version = v
            break

    if version is None:
        print(f"Error: Version '{version_id}' not found in registry.")
        print("Available versions:")
        list_versions(registry_path)
        sys.exit(1)

    adapter_path = version["path"]

    # Check for a pre-merged version first
    merged_path = adapter_path.rstrip("/") + "__merged"
    if merge and os.path.exists(merged_path):
        print(f"Loading pre-merged model from {merged_path}...")
        tokenizer = AutoTokenizer.from_pretrained(merged_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=True,
        )
        print(f"✅ Merged model '{version_id}' loaded from cache.")
        return model, tokenizer

    # Load base + adapter
    print(f"Loading base model from {base_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        local_files_only=True,
    )

    print(f"Loading adapter from {adapter_path}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)

    if merge:
        print("Merging adapter into base weights...")
        model = model.merge_and_unload()
        print(f"✅ Merged model '{version_id}' ready.")
    else:
        print(f"✅ Model '{version_id}' loaded (adapter active, not merged).")

    return model, tokenizer


def register_version(version_id, stage, path, parent, method,
                     dataset="", key_hparams=None, eval_metrics=None,
                     notes="", registry_path=None):
    """
    Register a new model version in the registry.

    Call this after training completes to record the new version.
    """
    from datetime import date

    reg = get_registry(registry_path)

    # Check for duplicate
    for v in reg.get("versions", []):
        if v["id"] == version_id:
            print(f"Warning: Version '{version_id}' already exists. Updating.")
            reg["versions"].remove(v)
            break

    entry = {
        "id": version_id,
        "stage": stage,
        "path": path,
        "parent": parent,
        "method": method,
        "dataset": dataset,
        "key_hparams": key_hparams or {},
        "eval_metrics": eval_metrics or {},
        "created": str(date.today()),
        "notes": notes,
    }
    reg.setdefault("versions", []).append(entry)

    path_to_write = registry_path or REGISTRY_PATH
    with open(path_to_write, "w") as f:
        json.dump(reg, f, indent=2)

    print(f"✅ Registered version '{version_id}' → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Load or list model versions.")
    sub = parser.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="List all available model versions")

    # load
    load_p = sub.add_parser("load", help="Load a model version (interactive test)")
    load_p.add_argument("version", help="Version ID (e.g., base, sft-v1, dpo-v1)")
    load_p.add_argument("--merge", action="store_true", help="Merge adapter into base")

    args = parser.parse_args()

    if args.command == "list":
        list_versions()
    elif args.command == "load":
        model, tokenizer = load_model(args.version, merge=args.merge)
        print(f"\nModel type: {type(model).__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
