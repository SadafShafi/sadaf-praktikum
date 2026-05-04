#!/usr/bin/env python3
"""
Structured JSON Inference Script for Llama 3.1 8B.
Uses the same model loader as inference.py, but prompts the model
to respond in a strict JSON schema and validates it with Pydantic.

Usage:
    sbatch run_inference.sh dspy_inference.py --version base --prompt "Explain segfaults"
    ./run_inference.sh dspy_inference.py --version base --prompt "Explain segfaults"
"""

import argparse
import sys
import os
import json

# Ensure project root is on path so 'models' package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.load_model import load_model

try:
    from pydantic import BaseModel, Field
except ImportError:
    print("Error: pydantic is not installed. pip install pydantic")
    sys.exit(1)


# 1. Define the desired JSON output schema using Pydantic
class StructuredResponse(BaseModel):
    summary: str = Field(description="A one-sentence summary of the answer.")
    explanation: str = Field(description="A detailed technical explanation.")
    examples: list[str] = Field(description="A list of 2-3 concrete examples or scenarios.")


# 2. Build the system prompt that instructs the model to output JSON
SYSTEM_PROMPT = """You are a technical expert. You MUST respond with ONLY valid JSON matching this exact schema — no markdown, no extra text:

{
  "summary": "<one-sentence summary>",
  "explanation": "<detailed technical explanation>",
  "examples": ["<example 1>", "<example 2>", "<example 3>"]
}

Respond with ONLY the JSON object. Do not wrap it in ```json blocks."""


def generate_json(model, tokenizer, prompt, max_new_tokens=512, temperature=0.3):
    """Generate a structured JSON response from the model."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raw_text


def parse_and_validate(raw_text: str) -> StructuredResponse:
    """Parse the model's raw text output into a validated Pydantic object."""
    # Strip markdown code fences if the model wraps its output
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]  # remove first ``` line
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    parsed = json.loads(cleaned)
    return StructuredResponse(**parsed)


def main():
    parser = argparse.ArgumentParser(description="Structured JSON Inference")
    parser.add_argument("--version", "-v", default="base", help="Model version ID")
    parser.add_argument("--prompt", "-p", required=True, help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens")
    parser.add_argument("--merge", action="store_true", help="Merge adapter first")
    args = parser.parse_args()

    print(f"Loading model version: {args.version}...")
    model, tokenizer = load_model(args.version, merge=args.merge)
    model.eval()

    # --- Diagnostic: check if model is actually on GPU ---
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    device = next(model.parameters()).device
    print(f"Model is on: {device}")

    if not torch.cuda.is_available():
        print("\n❌ ERROR: No GPU detected! Inference on CPU would take hours.")
        print("Make sure to run this via SLURM: sbatch run_inference.sh dspy_inference.py ...")
        sys.exit(1)

    print("Model loaded. Generating structured response...\n")

    raw_text = generate_json(model, tokenizer, args.prompt, max_new_tokens=args.max_tokens)

    print("--- Raw model output ---")
    print(raw_text)
    print("------------------------\n")

    try:
        result = parse_and_validate(raw_text)
        print("================ Validated JSON ================")
        print(result.model_dump_json(indent=2))
        print("=================================================")
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️  Could not parse model output as valid JSON: {e}")
        print("The raw output is shown above. You may need to adjust the prompt or temperature.")


if __name__ == "__main__":
    main()
