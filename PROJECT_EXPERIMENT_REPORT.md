# Domain Adaptation Experiments for an Agentic Linux/HPC Assistant

## 1. Executive Summary

This project investigated whether a Llama 3.1 8B instruction model can be adapted to environment-specific Linux and HPC failure-repair tasks using synthetic data, supervised fine-tuning (SFT), Direct Preference Optimization (DPO), and an iterative/online DPO prototype.

The implemented workflow was:

1. Define environment-specific error taxonomies and system state.
2. Generate synthetic failure scenarios and three candidate repairs for each scenario.
3. Split the data into a stratified training set and a held-out test set.
4. Measure the base model before training.
5. Train SFT on the best-practice repair command and evaluate it.
6. Train offline DPO on preference pairs and evaluate it.
7. Execute an iterative DPO prototype using the dataset as a proxy environment.

The main conclusion is that the dataset contains learnable patterns, and SFT was the strongest measured method for exact command reproduction and word-level similarity. On the 205-example holdout, SFT improved Rank 1 exact match from 2.4% to 13.7% and average Rank 1 Jaccard similarity from 30.2% to 49.7%. Offline DPO, as actually executed, trained directly from the base model rather than from SFT. It achieved strong preference-training metrics, but its held-out command metrics were below SFT: 5.4% exact match and 41.5% average Rank 1 Jaccard similarity.

The intended final system is therefore sequential training, `Base -> SFT -> DPO`, but that sequential experiment was not completed in the recorded runs. The available DPO result is a `Base -> DPO` result and must not be interpreted as evidence against sequential SFT followed by DPO.

## 2. Project Goal and Hypothesis

The project goal was to make an agentic language model adaptable to a particular system environment. The motivating hypothesis was that recurring failures in an environment have recognizable patterns, and that an assistant can learn to select repairs that are not merely successful, but also safe and appropriate for that environment.

The long-term vision included an agent that proposes commands, executes or tests them in an environment, receives feedback, and updates its behavior. The implemented work used a synthetic dataset and a proxy feedback environment to test the learnability and alignment parts of that vision before connecting the system to a live environment.

The model was instructed to output only the exact Bash command that repairs the reported failure. This made the task measurable with exact match, substring match, and token-overlap metrics, while an LLM judge provided a separate semantic and safety assessment.

## 3. Scenario and Dataset Design

### 3.1 Environment model

The synthetic scenarios were based on Ansible-derived system facts. The intended System1 environment included constraints such as:

- Approximately 145 MB of free RAM.
- No swap space.
- Approximately 125 MB of free disk space.
- Active `apt`/`dpkg` locks.
- Occupied ports including 80, 8080, and 5432.
- Root-owned or otherwise protected files.

These facts were used to make the same command fail for environment-specific reasons rather than only for generic syntax errors.

### 3.2 Error taxonomies

The dataset covered five categories:

1. Resource Exhaustion (Memory/OOM)
2. Storage / Disk Space Exhaustion
3. Package Management / Daemon Locks
4. Network Port Collisions
5. Privilege & Access Control (Permission Denied)

### 3.3 Structure of each scenario

Each record contains:

- A unique scenario ID and environment description.
- A state snapshot.
- The original command that failed.
- The error output and nonzero exit code.
- Three candidate repair commands.
- An execution result and safety assessment for each candidate.
- A rank describing the candidate's quality.

The candidates were designed as:

- Rank 1: best practice and safe repair.
- Rank 2: a risky bypass that may work but can damage or weaken the system.
- Rank 3: a failing, destructive, or inappropriate fallback/system-level option.

The generated records used an LLM through the Academic Cloud OpenAI-compatible API. `generate_data.py` generated the scenarios, while `analyze_data.py` checked completeness, fields, candidate outcomes, command diversity, and produced plots.

## 4. Data Generation, Analysis, and Preparation

### 4.1 Generation

The project generated 2,010 scenarios in `data/System1/generated_errors.json`. The final taxonomy counts were:

| Taxonomy | Count |
|---|---:|
| Resource Exhaustion (Memory/OOM) | 403 |
| Storage / Disk Space Exhaustion | 403 |
| Package Management / Daemon Locks | 402 |
| Network Port Collisions | 401 |
| Privilege & Access Control | 401 |
| **Total** | **2,010** |

The target was approximately 400 examples per taxonomy. The generated data was analyzed before model training to check structural validity and candidate quality. The project also retained `generation_progress.json`, the Ansible facts, taxonomy definitions, and system limitations documentation.

### 4.2 Train/test split

`split_data.py` created a stratified split using random seed `42` and a 10% test ratio. The final files were:

- `data/System1/train.json`: 1,805 records.
- `data/System1/test.json`: 205 records.

The holdout was balanced at 41 examples per taxonomy. The test set was not used to prepare or train either SFT or offline DPO.

### 4.3 SFT and DPO formatting

The preparation commands were:

```bash
python3 generate_data.py
python3 analyze_data.py
python3 split_data.py
python3 SFT/prepare_sft_data.py
python3 DPO/prepare_dpo_data.py
```

The training set was split again into:

- 1,625 formatted training examples.
- 180 formatted validation examples.

For SFT, only the Rank 1 command was used as the assistant completion. This tests whether the dataset contains a learnable mapping from the error context to the desired best-practice command.

For DPO, Rank 1 was used as `chosen`; Rank 3, or Rank 2 when necessary, was used as `rejected`. This was intended to teach the model to prefer safe best practices over risky or failing alternatives.

## 5. Experiment 1: Early Base-Model Baseline

### Purpose

The first experiment established how the unmodified Llama 3.1 8B model behaved on the test set before any fine-tuning.

### Procedure

The model was loaded from `models/base/meta-llama-3.1-8b-instruct` and evaluated using the existing inference/evaluation scripts. Generation used a low temperature, top-p sampling, and a short output limit because the target was a single command.

The project notes record an early run as 205 evaluated examples, 6 exact matches (2.9%), and 20 close matches (9.8%), completed in 1.9 minutes. Earlier exploratory SLURM runs also recorded values between 1.5% and 2.0% exact match for no-state and with-state variants. These were exploratory results and were later superseded by timestamped reproducible baseline artifacts.

### Interpretation

The base model had very limited ability to reproduce the desired repair command exactly. It also tended to behave like a general conversational model, producing extra text or choosing commands that were plausible in isolation but wrong for the supplied failure and environment.

## 6. Experiment 2: Reproducible Base Baselines

### Purpose

The reproducible baseline established the comparison point for later SFT and DPO runs. Two prompt conditions were tested:

- System1: error and command only, without state.
- System2: error, command, and state snapshot.

### 6.1 Base model without state

Job `13651038` evaluated all 205 test examples on an NVIDIA A100. The primary saved artifacts were:

- `data/System1/analysis_output/baseline_System1_no_state_20260504_094936.json`
- `data/System1/analysis_output/report_baseline_System1_no_state_20260504_094936.txt`

Results:

| Metric | Rank 1 | Rank 2 | Rank 3 |
|---|---:|---:|---:|
| Exact match | 5/205 (2.4%) | 6/205 (2.9%) | 0/205 (0.0%) |
| Substring match | 13/205 (6.3%) | 10/205 (4.9%) | 3/205 (1.5%) |
| Average Jaccard | 30.2% | 26.1% | 16.6% |

The LLM judge evaluated 160 of the 205 predictions in this report:

- Semantic Rank 1: 43/160 (26.9%).
- Any semantic match: 89/160 (55.6%).
- Safe predictions: 76/160 (47.5%).
- Unsafe predictions: 84/160 (52.5%).
- Mean judge similarity: 4.6/10.

By Rank 1 Jaccard similarity, the strongest category was Package Management / Daemon Locks at 41.3%. Storage was the weakest at 16.2%. This showed that the base model could sometimes recognize familiar command patterns, but was unreliable on environment-specific repairs and had a substantial safety problem.

### 6.2 Base model with state

The corresponding with-state run was saved as `baseline_System2_with_state_20260504_094936.json`; the SLURM output recorded:

- Rank 1 exact match: 4/205 (2.0%).
- Rank 1 substring match: 10/205 (4.9%).
- Rank 1 average Jaccard: 26.3%.
- Rank 2 exact match: 9/205 (4.4%).
- Rank 3 exact match: 1/205 (0.5%).

Adding the state snapshot did not improve this base-model run. In this setup, the extra state information either did not help the base model usefully condition its answer or introduced additional context without enough learned task structure to interpret it.

### Baseline conclusion

The no-state baseline was selected as the main reference for SFT because it matched the System1 prompt used by the SFT evaluation. The with-state run remained a useful comparison, but it did not provide evidence that state alone improved repair selection.

## 7. Experiment 3: Supervised Fine-Tuning (SFT)

### Purpose

SFT was designed as a learnability check. Before testing preference optimization, the project needed to establish that the synthetic dataset contained a learnable relationship between failure context and best-practice repair command.

### Training setup

The SFT model used LoRA on top of the base Llama 3.1 8B model:

- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Rank: 16.
- Alpha: 32.
- Dropout: 0.05.
- Learning rate: `2e-4`.
- Epochs: 3.
- Per-device batch size: 4.
- Gradient accumulation: 4, giving effective batch size 16.
- Maximum sequence length: 512.
- BF16 training on one A100.

Only 13,631,488 of 8,043,892,736 parameters were trainable, or 0.1695% of the model.

### Execution issues and successful run

The first SFT attempt failed because a script referred to a nonexistent `total_mem` field instead of `total_memory`. Several jobs were retried before successful training.

Successful job `SFT/logs/sft_13653850.out` reported:

- Training examples: 1,625.
- Validation examples: 180.
- Final training loss: 0.3219.
- Final validation loss: 0.2148.
- Final validation token accuracy: 94.04%.
- Training runtime: 388.5 seconds.
- Adapter: `SFT/checkpoints/sft-v1`.

The adapter was registered as `sft-v1`.

### Held-out evaluation

The evaluation job was `SFT/logs/sft_13654162.out`. Results were saved in `SFT/results/sft_v1_evaluation_20260504_134740.json` and summarized in the corresponding report.

| Metric | Rank 1 | Rank 2 | Rank 3 |
|---|---:|---:|---:|
| Exact match | 28/205 (13.7%) | 7/205 (3.4%) | 0/205 (0.0%) |
| Substring match | 36/205 (17.6%) | 16/205 (7.8%) | 3/205 (1.5%) |
| Average Jaccard | 49.7% | 32.1% | 22.9% |

The LLM judge evaluated all 205 examples:

- Semantic Rank 1: 76/205 (37.1%).
- Any semantic match: 107/205 (52.2%).
- Safe predictions: 111/205 (54.1%).
- Unsafe predictions: 94/205 (45.9%).
- Mean similarity: 4.7/10.

Compared with the reproducible no-state baseline, SFT improved:

- Rank 1 exact match by 11.3 percentage points, from 2.4% to 13.7%.
- Rank 1 average Jaccard by 19.5 percentage points, from 30.2% to 49.7%.
- Semantic Rank 1 match by 10.2 percentage points on the available judge evaluations, from 26.9% to 37.1%.
- Safe prediction rate by 6.6 percentage points, from 47.5% to 54.1%, although the baseline judge covered only 160 examples while SFT covered all 205.

### SFT conclusion

SFT clearly passed the learnability check. The model learned both the output format and many of the dataset's command-to-error patterns. It was the best measured model for strict Rank 1 command reproduction. However, exact match remained low in absolute terms, and almost half of the SFT predictions were still judged unsafe. This left room for preference optimization, but also showed that the synthetic labels and command-level evaluation needed careful interpretation.

## 8. Experiment 4: Offline DPO

### Purpose

Offline DPO was intended to use the preference structure in the dataset to reinforce Rank 1 best practices and suppress Rank 2 or Rank 3 behavior.

### Provenance correction

The DPO README describes training on top of the SFT model. The executed implementation does not do that. `DPO/train_dpo.py` sets `BASE_MODEL_VERSION = "base"`, and successful job `DPO/logs/dpo_13664972.out` confirms that it loaded the base model.

Consequently, the measured experiment is:

```text
Base model -> DPO adapter
```

It is not:

```text
Base model -> SFT adapter -> DPO adapter
```

This distinction is essential when comparing DPO with SFT. The DPO result tests direct preference training from the base model, not the intended sequential pipeline.

### Training setup and execution

The main DPO configuration was:

- LoRA rank 16, alpha 32, dropout 0.05.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Learning rate: `5e-5`.
- DPO beta: 0.1.
- Epochs: 3.
- Per-device batch size: 2.
- Gradient accumulation: 8, giving effective batch size 16.
- Maximum sequence length: 512.
- BF16 training on one A100.

Some initial DPO jobs failed because of a TRL/PyTorch incompatibility involving `torch.distributed.fsdp.FSDPModule`. After retries, job `13664972` completed:

- Training examples: 1,625.
- Validation examples: 180.
- Final training loss: 0.1583.
- Final validation loss: 0.1747.
- Final validation reward accuracy: 94.57%.
- Runtime: 946.6 seconds, approximately 15.8 minutes.
- Adapter: `DPO/checkpoints/dpo-v1`.

The adapter was registered as `dpo-v1` with the base model as its parent.

### Held-out evaluation

Job `DPO/logs/dpo_13665625.out` evaluated all 205 holdout examples. Results were saved in `DPO/results/dpo_v1_evaluation_20260505_132758.json`.

| Metric | Rank 1 | Rank 2 | Rank 3 |
|---|---:|---:|---:|
| Exact match | 11/205 (5.4%) | 7/205 (3.4%) | 0/205 (0.0%) |
| Substring match | 24/205 (11.7%) | 13/205 (6.3%) | 4/205 (2.0%) |
| Average Jaccard | 41.5% | 31.6% | 19.6% |

The LLM judge evaluated 200 of the 205 examples:

- Semantic Rank 1: 66/200 (33.0%).
- Any semantic match: 105/200 (52.5%).
- Safe predictions: 98/200 (49.0%).
- Unsafe predictions: 102/200 (51.0%).
- Mean similarity: 4.5/10.

Compared with the no-state baseline, direct Base-to-DPO improved Rank 1 exact match from 2.4% to 5.4% and Rank 1 Jaccard from 30.2% to 41.5%, but it remained below SFT's 13.7% exact match and 49.7% Jaccard. DPO also did not improve safety over SFT or the no-state baseline in the final judge sample: 49.0% safe versus 54.1% for SFT and 47.5% for the baseline. The comparison is not perfectly paired because the judge evaluated 160 baseline samples, 205 SFT samples, and 200 DPO samples.

### Offline DPO conclusion

The DPO training objective itself learned the supplied preferences, as shown by 94.57% validation reward accuracy. That training success did not translate into the strongest held-out command performance. The most likely interpretation is that direct DPO from the base model did not preserve or acquire the output-format and command-reproduction ability that SFT provided. The result supports testing the intended sequential `Base -> SFT -> DPO` pipeline rather than treating this direct DPO run as the final alignment experiment.

## 9. Experiment 5: Iterative / Online DPO Prototype

### Intended purpose

The online DPO idea was to simulate an agent interacting with an environment:

1. The model proposes a repair.
2. The environment evaluates the proposal.
3. A successful or preferred repair becomes `chosen`.
4. The model proposal becomes `rejected` when it fails.
5. The model is updated immediately on the new feedback.

### What was actually executed

The implementation is in `DPO/train_iterative_dpo.py`. It processes the 1,805 training records in chunks of 100, with a final chunk of 5, for 19 batches total. Each batch creates preference examples and performs one epoch of DPO training.

The important limitation is that the script does not call the model to generate a command. It explicitly uses:

```python
generated_cmd = "sudo " + original_cmd
```

The proxy environment then looks up the known Rank 1 and non-Rank 1 commands from the same dataset. Therefore this run is a working prototype of the data flow and training loop, but not a genuine online agent experiment. It demonstrates how feedback would be recorded and consumed, while the proposed commands are deterministic placeholders rather than model outputs.

### Recorded outcome

Job `DPO/logs/dpo_13665254.out` completed in approximately 7 minutes. It processed all 19 batches and saved the interaction history in `DPO/results/online_dpo_history.jsonl` and the model under `DPO/checkpoints/online-dpo-v1`.

The log shows that preference margins often increased during early batches, but later batches also showed unstable losses, large gradient norms, and batches in which reward accuracy dropped substantially. No comparable 205-example held-out evaluation report was produced for `online-dpo-v1`.

### Online DPO conclusion

The prototype validated that the project can stream proxy feedback into repeated DPO updates. It did not yet measure whether a model improves through autonomous interaction, because generation was stubbed and no held-out evaluation was run. It should be treated as infrastructure work and a conceptual demonstration, not as evidence of successful online adaptation.

## 10. Cross-Experiment Comparison

The most comparable results are the no-state base baseline, SFT, and direct Base-to-DPO evaluation on the same 205-example holdout:

| Model | Parent used in training | Rank 1 exact | Rank 1 Jaccard | Semantic Rank 1 | Safe predictions |
|---|---|---:|---:|---:|---:|
| Base | None | 2.4% | 30.2% | 26.9%* | 47.5%* |
| SFT (`sft-v1`) | Base | **13.7%** | **49.7%** | **37.1%** | **54.1%** |
| DPO (`dpo-v1`) | Base | 5.4% | 41.5% | 33.0%* | 49.0%* |

`*` The judge sample sizes differ: 160 for Base, 205 for SFT, and 200 for DPO.

The experiments show three different capabilities:

- The base model provides general language and Linux knowledge, but has weak task-specific command selection.
- SFT teaches the task format and the mapping to the dataset's Rank 1 commands most effectively.
- DPO can strongly separate chosen and rejected training responses, but direct DPO from the base model did not match SFT's held-out command performance.

## 11. Final Conclusions

### What was demonstrated

1. The synthetic dataset was structurally usable and approximately balanced across five environment-specific failure taxonomies.
2. The task is learnable: SFT substantially improved held-out command reproduction.
3. Low-rank adaptation was computationally practical, training only 0.1695% of the base model parameters.
4. Preference training successfully optimized the offline DPO objective, reaching 94.57% validation reward accuracy.
5. The project implemented an end-to-end prototype for iterative feedback collection and immediate DPO updates.

### What was not demonstrated

1. The intended sequential `Base -> SFT -> DPO` model was not trained and evaluated in the recorded experiments.
2. The online DPO run did not use actual model-generated proposals; it used a placeholder command.
3. The online DPO model has no saved held-out evaluation comparable to the other models.
4. The with-state condition was tested for the base model, but a complete with-state SFT/DPO comparison was not recorded.
5. Exact command match remains low even after SFT, so a high validation token accuracy should not be treated as high task accuracy.

### Overall conclusion

The project supports the central learnability hypothesis, but the strongest evidence is for SFT rather than for the currently executed DPO variants. SFT is the best current model for reproducing safe Rank 1 commands on the holdout. Direct Base-to-DPO improves over the untouched base model on several string metrics, yet loses much of SFT's task-format and command-reproduction advantage. The next scientifically decisive experiment is to initialize DPO from `sft-v1`, evaluate the resulting adapter with exactly the same prompt, decoding, holdout, and judge protocol, and then replace the online placeholder generation with real model generation before evaluating iterative adaptation.

## 12. Reproducibility and Source Artifacts

Core design and documentation:

- `README.md`
- `Notes.md`
- `pad.md`
- `experiment_design.md`
- `SFT/sft_plan.md`
- `DPO/README.md`

Data and preparation:

- `generate_data.py`
- `analyze_data.py`
- `split_data.py`
- `SFT/prepare_sft_data.py`
- `DPO/prepare_dpo_data.py`

Training and evaluation:

- `SFT/train_sft.py`
- `SFT/evaluate_sft.py`
- `DPO/train_dpo.py`
- `DPO/evaluate_dpo.py`
- `DPO/train_iterative_dpo.py`

Main result artifacts:

- `data/System1/analysis_output/report_baseline_System1_no_state_20260504_094936.txt`
- `SFT/results/report_sft_v1_evaluation_20260504_134740.txt`
- `DPO/results/report_dpo_v1_evaluation_20260505_132758.txt`
- `DPO/results/online_dpo_history.jsonl`

The SLURM logs in `slurm_logs/`, `SFT/logs/`, and `DPO/logs/` preserve failed attempts, retries, hardware details, training metrics, and execution times.