#!/usr/bin/env python3
"""
Sanity check script — run this BEFORE submitting a SLURM job.
Validates imports, file paths, data format, and config without needing a GPU.

Usage:
    python3 SFT/sanity_check.py
"""

import sys
import os
import json

# Always resolve paths relative to the project root
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
errors = 0


def check(label, condition, msg_pass="OK", msg_fail="FAILED"):
    global errors
    if condition:
        print(f"  {PASS} {label}: {msg_pass}")
    else:
        print(f"  {FAIL} {label}: {msg_fail}")
        errors += 1


def warn(label, condition, msg_pass="OK", msg_warn="Warning"):
    if condition:
        print(f"  {PASS} {label}: {msg_pass}")
    else:
        print(f"  {WARN} {label}: {msg_warn}")


print("=" * 50)
print("  SFT SANITY CHECK")
print("=" * 50)

# ---- 1. Check imports ----
print("\n1. Checking Python imports...")

try:
    import torch
    check("torch", True, f"v{torch.__version__}")
except ImportError:
    check("torch", False, msg_fail="torch not installed")

try:
    import transformers
    check("transformers", True, f"v{transformers.__version__}")
except ImportError:
    check("transformers", False, msg_fail="transformers not installed")

try:
    import peft
    check("peft", True, f"v{peft.__version__}")
except ImportError:
    check("peft", False, msg_fail="peft not installed — run: pip install peft")

try:
    import trl
    check("trl", True, f"v{trl.__version__}")
except ImportError:
    check("trl", False, msg_fail="trl not installed — run: pip install trl")

try:
    import datasets
    check("datasets", True, f"v{datasets.__version__}")
except ImportError:
    check("datasets", False, msg_fail="datasets not installed — run: pip install datasets")

try:
    from tqdm import tqdm
    check("tqdm", True)
except ImportError:
    check("tqdm", False, msg_fail="tqdm not installed")

# ---- 2. Check file paths ----
print("\n2. Checking file paths...")

check("train.json", os.path.exists("data/System1/train.json"),
      msg_fail="data/System1/train.json not found")
check("test.json", os.path.exists("data/System1/test.json"),
      msg_fail="data/System1/test.json not found")
check("sft_train.jsonl", os.path.exists("SFT/data/sft_train.jsonl"),
      msg_fail="SFT/data/sft_train.jsonl not found — run: python3 SFT/prepare_sft_data.py")
check("sft_val.jsonl", os.path.exists("SFT/data/sft_val.jsonl"),
      msg_fail="SFT/data/sft_val.jsonl not found — run: python3 SFT/prepare_sft_data.py")
check("model_registry.json", os.path.exists("models/model_registry.json"),
      msg_fail="models/model_registry.json not found")

# ---- 3. Check model registry ----
print("\n3. Checking model registry...")

if os.path.exists("models/model_registry.json"):
    with open("models/model_registry.json") as f:
        registry = json.load(f)
    check("base model entry", "base" in registry,
          msg_fail="No 'base' entry in model_registry.json")
    if "base" in registry:
        base_path = registry["base"]["path"]
        check("base model files", os.path.isdir(base_path),
              msg_pass=base_path,
              msg_fail=f"Base model directory not found: {base_path}")

# ---- 4. Validate training data format ----
print("\n4. Validating training data format...")

if os.path.exists("SFT/data/sft_train.jsonl"):
    with open("SFT/data/sft_train.jsonl") as f:
        lines = f.readlines()

    check("train file not empty", len(lines) > 0,
          msg_pass=f"{len(lines)} samples",
          msg_fail="File is empty!")

    if lines:
        sample = json.loads(lines[0])
        check("has 'messages' key", "messages" in sample,
              msg_fail="First line missing 'messages' key")

        if "messages" in sample:
            msgs = sample["messages"]
            check("has 3 messages (system/user/assistant)", len(msgs) == 3,
                  msg_pass=f"{len(msgs)} messages",
                  msg_fail=f"Expected 3 messages, got {len(msgs)}")

            roles = [m["role"] for m in msgs]
            check("correct role order", roles == ["system", "user", "assistant"],
                  msg_pass=str(roles),
                  msg_fail=f"Expected [system, user, assistant], got {roles}")

            # Check assistant response is a clean command (no markdown)
            assistant_msg = msgs[-1]["content"]
            check("assistant msg is clean (no ```)",
                  "```" not in assistant_msg,
                  msg_pass=f"'{assistant_msg[:60]}...'",
                  msg_fail=f"Contains markdown: '{assistant_msg[:60]}'")

            check("assistant msg is short (<200 chars)",
                  len(assistant_msg) < 200,
                  msg_pass=f"{len(assistant_msg)} chars",
                  msg_fail=f"Suspiciously long: {len(assistant_msg)} chars")

if os.path.exists("SFT/data/sft_val.jsonl"):
    with open("SFT/data/sft_val.jsonl") as f:
        val_lines = f.readlines()
    check("val file not empty", len(val_lines) > 0,
          msg_pass=f"{len(val_lines)} samples")

# ---- 5. Check train_sft.py can be parsed ----
print("\n5. Checking train_sft.py syntax...")

try:
    import py_compile
    py_compile.compile("SFT/train_sft.py", doraise=True)
    check("train_sft.py syntax", True)
except py_compile.PyCompileError as e:
    check("train_sft.py syntax", False, msg_fail=str(e))

try:
    py_compile.compile("SFT/evaluate_sft.py", doraise=True)
    check("evaluate_sft.py syntax", True)
except py_compile.PyCompileError as e:
    check("evaluate_sft.py syntax", False, msg_fail=str(e))

# ---- 6. Check load_model ----
print("\n6. Checking load_model import...")

try:
    from models.load_model import load_model, register_version
    check("load_model importable", True)
except Exception as e:
    check("load_model importable", False, msg_fail=str(e))

# ---- 7. GPU check (informational only) ----
print("\n7. GPU availability (informational)...")

try:
    import torch
    if torch.cuda.is_available():
        check("CUDA available", True, msg_pass=torch.cuda.get_device_name(0))
    else:
        warn("CUDA available", False, msg_warn="No GPU on login node — that's expected, SLURM will provide one")
except:
    warn("CUDA check", False, msg_warn="Could not check GPU")

# ---- 8. SLURM script check ----
print("\n8. Checking SLURM script...")

if os.path.exists("SFT/run_sft.sh"):
    with open("SFT/run_sft.sh") as f:
        slurm_content = f.read()
    check("run_sft.sh exists", True)
    check("has --gres=gpu", "--gres=gpu" in slurm_content,
          msg_fail="Missing GPU request in SLURM script!")
    check("has --export=NONE", "--export=NONE" in slurm_content,
          msg_fail="Missing --export=NONE for clean environment")
    check("has HF_HUB_OFFLINE=1", "HF_HUB_OFFLINE=1" in slurm_content,
          msg_fail="Missing HF_HUB_OFFLINE=1 for air-gapped cluster")
else:
    check("run_sft.sh exists", False, msg_fail="SFT/run_sft.sh not found")

# ---- Summary ----
print("\n" + "=" * 50)
if errors == 0:
    print(f"  {PASS} ALL CHECKS PASSED — safe to submit!")
    print("  Run: sbatch SFT/run_sft.sh")
else:
    print(f"  {FAIL} {errors} CHECK(S) FAILED — fix before submitting!")
print("=" * 50)

sys.exit(errors)
