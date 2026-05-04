# Sadaf Praktikum

## Setup

```bash
source .venv_new/bin/activate
```

## Data Generation

Generate synthetic error scenarios using LLM:

```bash
python3 generate_data.py
```

Analyze generated data:

```bash
python3 analyze_data.py
```

Output: `Data Generator /System1/analysis_output/`

## Model Management

Download base model (requires `HF_TOKEN` in `.env`):

```bash
python3 models/download_base.py
```

List registered model versions:

```bash
python3 inference.py --version list
```

## Inference

### Via SLURM (GPU — recommended)

```bash
# Single prompt
sbatch run_inference.sh --prompt "Explain what a segfault is"

# Specific model version
sbatch run_inference.sh --version sft-v1 --prompt "Fix this error: permission denied"

# Custom generation params
sbatch run_inference.sh --prompt "Hello" --max-tokens 256 --temperature 0.3

# Check output
cat slurm_logs/inference_<JOB_ID>.out
```

### Local (CPU — slow, for testing only)

```bash
python3 inference.py --version base --prompt "Hello"

# Interactive mode
python3 inference.py --version base
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--version` | `base` | Model version ID |
| `--prompt` | — | Input prompt (omit for interactive mode) |
| `--max-tokens` | 512 | Max tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling |
| `--merge` | — | Merge adapter into base |
