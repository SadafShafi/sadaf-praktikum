# SFT (Supervised Fine-Tuning) Plan

## Objective
Fine-tune Llama 3.1 8B on our curated System1 training dataset (1,805 samples) to teach it to predict the **correct bash fix command** given an error scenario. This serves as our **"learnability check"** — proving the dataset contains learnable patterns before moving to DPO and RL.

---

## 1. Data Preparation

### 1.1 Source Data
- **File:** `data/System1/train.json` (1,805 samples)
- **Each sample contains:**
  - `interaction.original_command` — the broken command the user ran
  - `interaction.error_output` — the error trace
  - `candidates[].command` — three ranked fix candidates (Rank 1 = Best Practice)
  - `state_snapshot` — system state at time of error

### 1.2 Format Conversion
We need to convert each sample into a **prompt-completion pair** for SFT.

**Two variants to prepare:**

#### Variant A: System1 (No State) — matches our baseline
```
[PROMPT]
You are an expert Linux system administrator.
A user tried to run a command and encountered an error.
Your task is to provide the EXACT bash command to fix the issue.

Original Command:
{original_command}

Error Output:
{error_output}

Respond with ONLY the exact bash command to fix the error.

[COMPLETION]
{rank_1_command}
```

#### Variant B: System2 (With State) — for comparison
Same as above but includes `state_snapshot` in the prompt.

### 1.3 Script: `SFT/prepare_sft_data.py`
- Read `train.json`
- Convert each sample to Llama 3.1 chat format using the tokenizer's `apply_chat_template()`
- Output: `SFT/data/sft_train.jsonl` and `SFT/data/sft_val.jsonl` (90/10 split from train.json)
- Each line: `{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

### 1.4 Target Label
- **Primary target:** Rank 1 (`best_practice`) command only
- We train the model to always predict the best practice fix
- DPO will later teach it *why* Rank 1 is better than Rank 2

---

## 2. Training Configuration

### 2.1 Method: LoRA (Low-Rank Adaptation)
- We do **not** fine-tune all 8B parameters (too expensive, risks catastrophic forgetting)
- Instead, we train a small **LoRA adapter** (~0.5-2% of parameters)
- The base model weights are frozen

### 2.2 Key Hyperparameters
| Parameter | Value | Rationale |
|---|---|---|
| `lora_r` | 16 | Rank of the low-rank matrices |
| `lora_alpha` | 32 | Scaling factor (alpha/r = 2 is standard) |
| `lora_dropout` | 0.05 | Light regularization |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj"]` | Standard for Llama attention layers |
| `learning_rate` | 2e-4 | Standard for LoRA fine-tuning |
| `num_epochs` | 3 | Start with 3, check for overfitting |
| `batch_size` | 4 | Per-device batch size |
| `gradient_accumulation` | 4 | Effective batch size = 16 |
| `max_seq_length` | 512 | Our prompts are short (commands + errors) |
| `warmup_ratio` | 0.05 | 5% of steps for warmup |
| `weight_decay` | 0.01 | Standard regularization |
| `bf16` | True | Use bfloat16 for A100 efficiency |

### 2.3 Libraries
- `transformers` — model loading
- `peft` — LoRA adapter
- `trl` — `SFTTrainer` for supervised fine-tuning
- `datasets` — data loading

### 2.4 Hardware
- **GPU:** 1x NVIDIA A100 (or equivalent on Grete cluster)
- **SLURM partition:** `grete:shared` or `grete:interactive`
- **Estimated training time:** ~30-60 minutes for 3 epochs on 1,805 samples
- **GPU memory:** ~16-20 GB (LoRA + 8B model in bf16)

---

## 3. Training Script

### 3.1 Script: `SFT/train_sft.py`
```
1. Load base model using `models/load_model.py` (version="base")
2. Apply LoRA config using `peft.LoraConfig`
3. Load training data from `SFT/data/sft_train.jsonl`
4. Initialize `SFTTrainer` from `trl`
5. Train for N epochs
6. Save adapter to `models/checkpoints/sft-v1/`
7. Register in model_registry.json using `register_version()`
```

### 3.2 SLURM Job Script: `SFT/run_sft.sh`
- Based on existing `run_inference.sh` template
- Request GPU with sufficient memory (at least 20GB)
- Set `--export=NONE` for clean environment
- Set `HF_HUB_OFFLINE=1` for air-gapped cluster

---

## 4. Evaluation

### 4.1 Post-Training Evaluation
After SFT training completes:

1. **Load the SFT adapter:**
   ```python
   model, tokenizer = load_model("sft-v1")
   ```

2. **Run evaluation on test set (205 samples):**
   - Use existing `evaluate_test_set.py` (System1, no state)
   - Use existing `evaluate_test_set_with_state.py` (System2, with state)
   - Both scripts already calculate: Exact Match, Substring, Jaccard (per rank)

3. **Run LLM Judge:**
   - Use `llm_judge.py` with Qwen3 Coder to get semantic match + similarity score

4. **Generate comparison report:**
   - Use `analyze_results.py` to generate plots
   - Compare SFT results against baseline

### 4.2 Expected Outcome
| Metric | Baseline (no SFT) | Expected After SFT |
|---|---|---|
| Exact Match (Rank 1) | 2.4% | 30-60% |
| Avg Jaccard (Rank 1) | 30.2% | 60-80% |
| LLM Judge Match (Any) | 55.6% | 70-90% |
| Avg Similarity Score | 4.6/10 | 7-9/10 |

If SFT achieves significant improvement → **Data is predictive, proceed to DPO/RL**
If SFT shows minimal improvement → **Revisit data quality or prompt design**

---

## 5. File Structure
```
SFT/
├── sft_plan.md              ← This document
├── prepare_sft_data.py      ← Data formatting script
├── train_sft.py             ← Main training script
├── run_sft.sh               ← SLURM job submission script
├── data/
│   ├── sft_train.jsonl      ← Formatted training data
│   └── sft_val.jsonl        ← Formatted validation data
└── logs/                    ← Training logs and checkpoints
```

---

## 6. Step-by-Step Execution Order

1. [ ] **Create `prepare_sft_data.py`** — converts train.json → chat-formatted JSONL
2. [ ] **Run `prepare_sft_data.py`** — generates sft_train.jsonl + sft_val.jsonl
3. [ ] **Create `train_sft.py`** — LoRA training script using trl SFTTrainer
4. [ ] **Create `run_sft.sh`** — SLURM job script for GPU training
5. [ ] **Submit training job** — `sbatch SFT/run_sft.sh`
6. [ ] **Monitor training** — check loss curves in logs
7. [ ] **Register model** — save adapter path in model_registry.json
8. [ ] **Evaluate on test set** — run both evaluation scripts with version="sft-v1"
9. [ ] **Run LLM Judge** — evaluate with Qwen3 Coder
10. [ ] **Generate comparison plots** — baseline vs SFT side-by-side
11. [ ] **Document findings** — update Notes.md with SFT results

---

## 7. Important Notes

- **Do NOT use test.json during training.** Only train.json is used for SFT.
- **Save checkpoints every epoch** to allow recovery and to pick the best checkpoint.
- **Watch for overfitting:** If training loss drops but validation loss increases after epoch 2, use early stopping or reduce to 2 epochs.
- **The model should ONLY output the command** — no explanations, no markdown. The SFT data formatting must enforce this by having short, clean completions.
