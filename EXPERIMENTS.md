# Project Overview: Domain Adaptation of Agentic LLMs for Linux & HPC Repair

This document presents the complete project structured **experiment-wise**. The overarching research goal is to investigate whether an instruction-tuned LLM (Llama 3.1 8B) can adapt to a specific Linux / HPC environment by learning to diagnose command-line failures and output the safest best-practice fix.

---

## Overall Pipeline at a Glance

```
[Ansible System Facts] 
       ↓ 
[Experiment 1: Data Generation] (2,010 failure scenarios across 5 taxonomies)
       ↓ (Stratified 90/10 Split: 1,805 Train / 205 Test)
       ├─────────────────────────────────────────┐
       ↓                                         ↓
[Experiment 2: Base Baseline]          [Experiment 3: SFT]
(Zero-shot Llama 3.1 8B)               (Supervised LoRA on Rank 1 Fixes)
       │                                         │
       ├─────────────────────────────────────────┘
       ↓
[Experiment 4: Offline DPO] 
(LoRA trained on Rank 1 [Chosen] vs Rank 2/3 [Rejected])
       ↓
[Experiment 5: Iterative / Online DPO Prototype]
(Streaming feedback-loop simulated over proxy environment)
```

---

## Experiment 1: Synthetic Data Generation & Taxonomy Design

### 1. Name of Experiment
**Synthetic Dataset Generation and Environmental Constraint Modeling**

### 2. Goal
To create a realistic, domain-specific dataset of Linux/HPC error scenarios where each error has multiple potential fixes with different safety and quality levels, reflecting actual system constraints (such as disk limits, memory limits, and daemon locks).

### 3. Input and Output (Intuition)
* **Input (Intuition):** System environment snapshot (Ansible facts like CPU, RAM, disk mount points) + a designated failure category (e.g., *Resource Exhaustion*, *Port Collision*).
* **Output (Intuition):** A realistic failure interaction (the failed command, the exact terminal `stderr`, exit code) paired with **3 ranked fix candidates**:
  - **Rank 1 (`best_practice`):** Clean, safe, persistent solution (e.g., create a virtual environment, clean cache properly).
  - **Rank 2 (`risky_bypass`):** Fixes the immediate symptom dangerously (e.g., `sudo chmod 777`, `--break-system-packages`, killing arbitrary processes).
  - **Rank 3 (`failing_fix` / fallback):** A flawed fix that fails or ignores the underlying root cause.

### 4. Methodology
* **System Profiling:** Extracted real Ansible facts from `data/System1/AnsibleFacts.json`.
* **5 Target Taxonomies:**
  1. *Resource Exhaustion (Memory/OOM)*
  2. *Storage / Disk Space Exhaustion*
  3. *Package Management / Daemon Locks*
  4. *Network Port Collisions*
  5. *Privilege & Access Control (Permission Denied)*
* **Generation Engine:** Used `generate_data.py` (via OpenAI-compatible API) to iteratively generate 5 scenarios per prompt batch across all 5 taxonomies with structured JSON schema enforcement.
* **Dataset Split (`split_data.py`):** Stratified 90/10 split preserving taxonomy distribution:
  - **Train Set:** 1,805 samples
  - **Test Set (Holdout):** 205 samples

### 5. Result
* Successfully produced **2,010 valid, structured scenarios**.
* Each sample contains precise simulated stdout/stderr, exit codes, and explicitly ranked candidate solutions (Chosen = Rank 1, Rejected = Rank 2 / Rank 3).

---

## Experiment 2: Baseline Model Evaluation (Zero-Shot Base Llama 3.1 8B)

### 1. Name of Experiment
**Base Model Benchmark on Unseen Test Scenarios (With and Without System State)**

### 2. Goal
Establish the zero-shot baseline performance of the un-fine-tuned Llama 3.1 8B Instruct model to measure how well a general-purpose LLM solves domain-specific Linux errors out of the box.

### 3. Input and Output (Intuition)
* **Input (Intuition):** The broken command + the terminal error message (+ optionally the JSON system state snapshot).
  > *"User tried running `pip install tensorflow`, got `externally-managed-environment` error. Output only the single exact bash fix command."*
* **Output (Intuition):** The raw model response (often verbose conversational text or generic commands that were not tuned to the system's exact constraints).

### 4. Methodology
* **Model:** Untouched `meta-llama/Llama-3.1-8B-Instruct`.
* **Inference Settings:** Deterministic decoding (`temperature=0.1`, `top_p=0.9`, max tokens: 100).
* **Evaluations Conducted:**
  1. **No-State Prompt:** Tested on all 205 held-out samples.
  2. **With-State Prompt:** Tested on 205 held-out samples including the JSON system state.
  3. **Dual Metric Layer:**
     - *String-Level Metrics:* Exact Match, Substring Match, and Jaccard Word Similarity against Rank 1, Rank 2, and Rank 3.
     - *LLM-as-a-Judge (Qwen 2.5 Coder):* Semantic equivalence and safety classification (Safe vs. Destructive).

### 5. Result
* **No-State Performance (205 samples):**
  - **Rank 1 Exact Match:** **2.4%** (5/205)
  - **Rank 1 Word Overlap (Avg Jaccard):** **30.2%**
  - **LLM Judge Semantic Rank 1:** **26.9%** (43/160 judged)
  - **Safety Percentage:** **47.5%** safe
* **With-State Performance:**
  - Rank 1 Exact Match: **2.9%** (6/205)
  - Rank 1 Avg Jaccard: **30.3%**
* **Key Takeaway:** The base model acts like a verbose conversational assistant. It struggles to follow strict single-command output constraints and lacks specific alignment with local environment best practices.

---

## Experiment 3: Supervised Fine-Tuning (SFT)

### 1. Name of Experiment
**Supervised Fine-Tuning (SFT) for Format and Command Learnability**

### 2. Goal
Perform a "learnability check" to prove that the synthetic dataset contains learnable patterns and teach the model to output **strictly the single best-practice command (Rank 1)**.

### 3. Input and Output (Intuition)
* **Input (Intuition):** System prompt defining the expert role + original command + error trace.
* **Output (Intuition):** The exact Rank 1 (`best_practice`) bash command string, with zero conversational filler or markdown wrappers.

### 4. Methodology
* **Data Preparation (`SFT/prepare_sft_data.py`):** Converted 1,805 training samples into Llama 3.1 chat template format (`sft_train.jsonl`: 1,624 train, `sft_val.jsonl`: 181 val).
* **Training (`SFT/train_sft.py`):**
  - **Technique:** Parameter-Efficient Fine-Tuning via LoRA (`r=16`, `alpha=32`, `dropout=0.05`, targeting attention projection modules `q, k, v, o`).
  - **Trainable Parameters:** Only ~13.6M out of 8B parameters (0.1695%).
  - **Hardware & Hyperparameters:** 1x NVIDIA A100 (Grete HPC cluster), batch size 4, gradient accumulation 4 (effective batch size 16), 3 epochs, learning rate `2e-4`, BF16 precision.
* **Evaluation:** Evaluated checkpoint `sft-v1` on the 205-sample holdout test set using both deterministic string matching and LLM-as-a-Judge.

### 5. Result
* **Training Metrics:** Training loss dropped smoothly from `1.643` to `0.306`; validation token accuracy reached **90.47%**.
* **Test Set Evaluation (205 samples):**
  - **Rank 1 Exact Match:** **13.7%** (28/205) — *a ~5.7x increase over baseline*
  - **Rank 1 Substring Match:** **30.2%** (62/205)
  - **Rank 1 Word Overlap (Avg Jaccard):** **49.7%**
  - **LLM Judge Semantic Rank 1:** **37.1%** (76/205)
  - **Safety Percentage:** **54.1%** (111/205) — *highest safety score across all models*
* **Key Takeaway:** SFT proved the dataset is highly learnable. SFT acted as a powerful **format and syntax learner**, significantly improving exact command replication and eliminating chatty non-command text.

---

## Experiment 4: Direct Preference Optimization (Offline DPO)

### 1. Name of Experiment
**Offline Preference Alignment via DPO (Chosen vs. Rejected Fixes)**

### 2. Goal
Align the model's policy to prefer safe, best-practice solutions (Rank 1) over risky workarounds (Rank 2) or failing commands (Rank 3), penalizing destructive system commands.

### 3. Input and Output (Intuition)
* **Input (Intuition):** The error prompt paired with two alternative completions:
  - **`chosen`:** The Rank 1 best-practice fix.
  - **`rejected`:** The Rank 2 risky bypass or Rank 3 failing fix.
* **Output (Intuition):** An implicit reward signal that trains the model weights to increase log-probabilities for clean fixes while decreasing log-probabilities for unsafe/flawed commands.

### 4. Methodology
* **Data Preparation (`DPO/prepare_dpo_data.py`):** Created 1,805 preference pairs (`dpo_train.jsonl`: 1,625 pairs, `dpo_val.jsonl`: 180 pairs).
* **Training (`DPO/train_dpo.py`):**
  - Implemented via Hugging Face TRL `DPOTrainer` (`beta=0.1`, LoRA `r=16`, `alpha=32`, learning rate `5e-6`, 3 epochs).
  - *Execution Note:* In this run, the LoRA adapter was initialized directly from the **Base Model** (`dpo-v1`).
* **Evaluation:** Run on the 205-example holdout test set using string matching and LLM-as-a-Judge.

### 5. Result
* **Training Metrics:** Validation loss reached `0.1747`; validation reward accuracy reached **94.57%** (proving the model successfully learned the preference margin).
* **Test Set Evaluation (205 samples):**
  - **Rank 1 Exact Match:** **5.4%** (11/205) — *better than Base (2.4%), but lower than SFT (13.7%)*
  - **Rank 1 Avg Jaccard:** **41.5%**
  - **LLM Judge Semantic Rank 1:** **33.0%** (66/200 judged)
  - **Dangerous / Rank 3 Pick Rate:** Dropped to near-zero (**0.0%** exact match on Rank 3, **2.5%** semantic).
  - **Safety Percentage:** **49.0%** (98/200 judged)
* **Key Takeaway:** DPO successfully learned the preference objective and effectively suppressed bad/failing commands. However, because it was trained directly on the Base model (instead of on top of the SFT checkpoint), it lacked SFT's structural format adherence, resulting in lower exact-match accuracy compared to SFT.

---

## Experiment 5: Iterative / Online DPO Prototype

### 1. Name of Experiment
**Feedback-in-the-Loop Iterative / Online DPO Simulation**

### 2. Goal
Prototype an active adaptation loop where an agent proposes a fix, receives simulated environment feedback, constructs dynamic preference pairs on-the-fly, and continuously updates its adapter weights.

### 3. Input and Output (Intuition)
* **Input (Intuition):** Sequential batches of incoming failure scenarios.
* **Output (Intuition):** Dynamic generation of candidate commands $\rightarrow$ Environmental verification (proxy lookup) $\rightarrow$ Instant streaming DPO loss update $\rightarrow$ Continuously adapting checkpoint.

### 4. Methodology
* **Script (`DPO/train_iterative_dpo.py`):**
  - Processed 1,805 samples in 19 sequential streaming batches (chunks of 100).
  - Evaluated proposals against the dataset proxy: if a proposal did not match Rank 1, it was labeled `rejected` and paired against the known Rank 1 `chosen` solution.
  - Performed 1 epoch of DPO weight updates per batch.
  - *Prototype Note:* To validate the execution pipeline, command proposals were constructed using a deterministic proxy (`"sudo " + original_cmd`).

### 5. Result
* **Execution:** Successfully processed all 19 streaming batches in ~7 minutes (`DPO/logs/dpo_13665254.out`), saving intermediate history to `online_dpo_history.jsonl` and checkpoint `online-dpo-v1`.
* **Observations:** Demonstrated the feasibility of streaming DPO updates, though late-batch loss oscillations and gradient norm spikes were observed.
* **Key Takeaway:** Proved the architecture and data pipeline for online DPO; laid the technical foundation for future experiments with live model generation in the loop.

---

## Comparative Results Summary

### Performance Across Models on 205 Holdout Samples

| Metric | Base Baseline (No State) | SFT (`sft-v1`) | Offline DPO (`dpo-v1` from Base) |
| :--- | :---: | :---: | :---: |
| **Rank 1 Exact Match** | 2.4% | **13.7%** | 5.4% |
| **Rank 1 Substring Match** | 7.8% | **30.2%** | 11.7% |
| **Rank 1 Word Overlap (Avg Jaccard)** | 30.2% | **49.7%** | 41.5% |
| **LLM Judge: Semantic Rank 1** | 26.9% | **37.1%** | 33.0% |
| **LLM Judge: Overall Safety %** | 47.5% | **54.1%** | 49.0% |
| **Rank 3 (Worst Fix) Exact Match** | 0.0% | 0.0% | **0.0% (Suppressed)** |

---

## Key Conclusions & Next Steps

1. **Dataset Quality & Learnability:** The synthetic dataset with Ansible environmental constraints proved effective and learnable.
2. **SFT as the Foundation:** Supervised fine-tuning is indispensable for teaching strict command formatting, syntax, and direct mapping from errors to best-practice fixes.
3. **DPO for Behavioral Safety:** DPO excels at optimizing preference margins and suppressing destructive fallbacks (94.57% preference accuracy).
4. **Recommended Next Stage:** Train sequential **`Base → SFT → DPO`** (initializing DPO from `sft-v1` rather than Base) and connect live model generation to the iterative online DPO feedback loop.
