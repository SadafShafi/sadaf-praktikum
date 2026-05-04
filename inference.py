#!/usr/bin/env python3
"""
Run inference on any registered model version.

Usage:
    # Interactive mode (type prompts, get responses)
    python3 inference.py --version base

    # Single prompt
    python3 inference.py --version base --prompt "Explain what a segfault is"

    # With generation params
    python3 inference.py --version base --prompt "Hello" --max-tokens 256 --temperature 0.7
"""

import argparse
import sys
import os

# Ensure project root is on path so 'models' package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.load_model import load_model, list_versions


def generate(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7,
             top_p=0.9, do_sample=True):
    """Generate a response for a given prompt using the chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    # Apply the chat template (Llama 3.1 Instruct uses a specific format)
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (skip the input)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response


def interactive_loop(model, tokenizer, max_new_tokens, temperature, top_p):
    """Interactive REPL for chatting with the model."""
    print("\n" + "=" * 60)
    print("Interactive mode — type your prompt and press Enter.")
    print("Commands:  /quit  /clear  /params")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt == "/quit":
            print("Bye!")
            break
        if prompt == "/params":
            print(f"  max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}")
            continue
        if prompt == "/clear":
            os.system("clear")
            continue

        print("Model: ", end="", flush=True)
        response = generate(model, tokenizer, prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature, top_p=top_p)
        print(response)
        print()


def main():
    parser = argparse.ArgumentParser(description="Run inference on a registered model.")
    parser.add_argument("--version", "-v", default="base",
                        help="Model version ID (default: base). Use 'list' to see all.")
    parser.add_argument("--prompt", "-p", default=None,
                        help="Single prompt (if omitted, starts interactive mode).")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max new tokens to generate (default: 512).")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7).")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p / nucleus sampling (default: 0.9).")
    parser.add_argument("--merge", action="store_true",
                        help="Merge adapter into base before inference.")
    args = parser.parse_args()

    if args.version == "list":
        list_versions()
        return

    print(f"Loading model version: {args.version}...")
    model, tokenizer = load_model(args.version, merge=args.merge)
    model.eval()
    print("Ready!\n")

    if args.prompt:
        # Single-shot mode
        response = generate(model, tokenizer, args.prompt,
                            max_new_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p)
        print(f"Prompt: {args.prompt}\n")
        print(f"Response: {response}")
    else:
        # Interactive mode
        interactive_loop(model, tokenizer, args.max_tokens,
                         args.temperature, args.top_p)


if __name__ == "__main__":
    main()
