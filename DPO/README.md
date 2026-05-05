# Direct Preference Optimization (DPO) Pipeline

This directory contains the entire pipeline for aligning the model using DPO.

## 1. Prepare DPO Dataset
First, we format the `train.json` dataset into DPO preference pairs (chosen vs rejected). 
We use the Rank 1 solutions as the `chosen` responses and the Rank 3 (or Rank 2) solutions as the `rejected` responses.

```bash
python3 DPO/prepare_dpo_data.py
```
This generates `DPO/data/dpo_train.jsonl` and `DPO/data/dpo_val.jsonl`.

## 2. Train DPO Model
We use `trl`'s `DPOTrainer` to train the DPO adapter on top of the SFT model.

```bash
sbatch DPO/run_dpo.sh DPO/train_dpo.py
```
* **Logs:** Check the training progress in `DPO/logs/`
* **Checkpoints:** The final model is saved to `DPO/checkpoints/dpo-v1/`

## 3. Evaluate DPO Model
We evaluate the DPO-aligned model on the test set using the exact same metrics (Exact Match, Substring Match, Jaccard Similarity) as in SFT to allow for direct comparison.

```bash
sbatch DPO/run_dpo.sh DPO/evaluate_dpo.py
```
* **Results:** The evaluation JSON will be saved in `DPO/results/`
