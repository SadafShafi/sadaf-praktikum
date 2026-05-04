#!/usr/bin/env python3
"""
Analyze generated_errors.json — produces summary, quality checks, and plots.
Run anytime (even while generate_data.py is still running).

Usage:
    python3 analyze_data.py
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "System1", "generated_errors.json")
OUTPUT_DIR = os.path.join("data", "System1", "analysis_output")
TARGET_PER_TAXONOMY = 400
EXPECTED_TAXONOMIES = [
    "Resource Exhaustion (Memory/OOM)",
    "Storage / Disk Space Exhaustion",
    "Package Management / Daemon Locks",
    "Network Port Collisions",
    "Privilege & Access Control (Permission Denied)",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("Error: expected a JSON array at top level.")
        sys.exit(1)
    return data


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


# ---------------------------------------------------------------------------
# 1. Dataset Summary
# ---------------------------------------------------------------------------

def dataset_summary(data):
    print("\n" + "=" * 60)
    print("1. DATASET SUMMARY")
    print("=" * 60)
    print(f"Total scenarios: {len(data)}")

    tax_counts = Counter(d.get("taxonomy", "MISSING") for d in data)
    print(f"Unique taxonomies: {len(tax_counts)}")
    total_target = TARGET_PER_TAXONOMY * len(EXPECTED_TAXONOMIES)
    print(f"Overall completeness: {len(data)}/{total_target} ({100*len(data)/total_target:.1f}%)\n")

    print(f"{'Taxonomy':<50} {'Count':>6} {'Target':>7} {'%':>6}")
    print("-" * 72)
    for tax in EXPECTED_TAXONOMIES:
        c = tax_counts.get(tax, 0)
        pct = 100 * c / TARGET_PER_TAXONOMY if TARGET_PER_TAXONOMY else 0
        print(f"{tax:<50} {c:>6} {TARGET_PER_TAXONOMY:>7} {pct:>5.1f}%")

    # unexpected taxonomies
    for tax, c in tax_counts.items():
        if tax not in EXPECTED_TAXONOMIES:
            print(f"{'⚠ ' + tax:<50} {c:>6}  (UNEXPECTED)")

    return tax_counts


# ---------------------------------------------------------------------------
# 2. Distribution Plots
# ---------------------------------------------------------------------------

def plot_taxonomy_distribution(tax_counts):
    labels = [t[:35] + "…" if len(t) > 35 else t for t in EXPECTED_TAXONOMIES]
    counts = [tax_counts.get(t, 0) for t in EXPECTED_TAXONOMIES]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, counts, color="#4C72B0")
    ax.axvline(TARGET_PER_TAXONOMY, color="red", linestyle="--", label=f"Target ({TARGET_PER_TAXONOMY})")
    ax.set_xlabel("Number of Scenarios")
    ax.set_title("Scenarios per Taxonomy")
    ax.legend()
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, str(c), va="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "taxonomy_distribution.png")


def plot_exit_code_distribution(data):
    codes = [d["interaction"]["exit_code"] for d in data if "interaction" in d]
    if not codes:
        return
    code_counts = Counter(codes)
    top = code_counts.most_common(15)
    labels = [str(c) for c, _ in top]
    vals = [v for _, v in top]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals, color="#55A868")
    ax.set_xlabel("Exit Code")
    ax.set_ylabel("Frequency")
    ax.set_title("Interaction Exit Code Distribution (Top 15)")
    fig.tight_layout()
    save_fig(fig, "exit_code_distribution.png")


def plot_fix_exit_codes(data):
    """Exit code distributions for fix_A, fix_B, fix_C side by side."""
    fix_codes = defaultdict(list)
    for d in data:
        for cand in d.get("candidates", []):
            cid = cand.get("id", "?")
            ec = cand.get("execution_result", {}).get("exit_code")
            if ec is not None:
                fix_codes[cid].append(ec)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, fix_id in zip(axes, ["fix_A", "fix_B", "fix_C"]):
        codes = fix_codes.get(fix_id, [])
        if codes:
            cc = Counter(codes)
            top = cc.most_common(10)
            ax.bar([str(c) for c, _ in top], [v for _, v in top], color="#C44E52")
        ax.set_title(f"{fix_id} exit codes")
        ax.set_xlabel("Exit Code")
    axes[0].set_ylabel("Frequency")
    fig.suptitle("Candidate Fix Exit Code Distributions", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fix_exit_codes.png")


# ---------------------------------------------------------------------------
# 3. Data Quality Checks
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ["id", "taxonomy", "environment", "state_snapshot", "interaction", "candidates", "chosen_response", "rejected_response"]
INTERACTION_FIELDS = ["original_command", "error_output", "exit_code"]
CANDIDATE_FIELDS = ["id", "command", "type", "execution_result", "rank"]

def quality_checks(data):
    print("\n" + "=" * 60)
    print("3. DATA QUALITY CHECKS")
    print("=" * 60)
    issues = []

    # Missing top-level fields
    for i, d in enumerate(data):
        for f in REQUIRED_FIELDS:
            if f not in d:
                issues.append(f"Row {i}: missing field '{f}'")

    # Interaction checks
    zero_exit = 0
    for i, d in enumerate(data):
        inter = d.get("interaction", {})
        for f in INTERACTION_FIELDS:
            if f not in inter:
                issues.append(f"Row {i}: interaction missing '{f}'")
        if inter.get("exit_code", 1) == 0:
            zero_exit += 1

    # Candidate checks
    wrong_chosen = 0
    wrong_rejected = 0
    fixb_not_clean = 0
    fixa_no_warn = 0
    fixc_zero_exit = 0
    for i, d in enumerate(data):
        if d.get("chosen_response") != "fix_B":
            wrong_chosen += 1
        if d.get("rejected_response") != "fix_A":
            wrong_rejected += 1
        for cand in d.get("candidates", []):
            er = cand.get("execution_result", {})
            if cand.get("id") == "fix_B" and er.get("safety_check") != "clean":
                fixb_not_clean += 1
            if cand.get("id") == "fix_A":
                sc = er.get("safety_check", "")
                if not str(sc).startswith("warn"):
                    fixa_no_warn += 1
            if cand.get("id") == "fix_C" and er.get("exit_code", 1) == 0:
                fixc_zero_exit += 1

    # Duplicate IDs
    ids = [d.get("id") for d in data]
    dup_ids = [uid for uid, cnt in Counter(ids).items() if cnt > 1]

    n = len(data)
    checks = [
        ("Missing required fields", len(issues), 0),
        ("Interaction exit_code == 0 (should be >0)", zero_exit, 0),
        ("chosen_response != 'fix_B'", wrong_chosen, 0),
        ("rejected_response != 'fix_A'", wrong_rejected, 0),
        ("fix_B safety_check != 'clean'", fixb_not_clean, 0),
        ("fix_A missing 'warn:' prefix", fixa_no_warn, 0),
        ("fix_C exit_code == 0 (should fail)", fixc_zero_exit, 0),
        ("Duplicate IDs", len(dup_ids), 0),
    ]

    all_pass = True
    for label, actual, expected in checks:
        status = "✅ PASS" if actual == expected else "❌ FAIL"
        if actual != expected:
            all_pass = False
        print(f"  {status}  {label}: {actual}")

    if dup_ids:
        print(f"         Duplicate IDs: {dup_ids[:5]}{'...' if len(dup_ids)>5 else ''}")
    if issues:
        print(f"         First 5 missing-field issues:")
        for iss in issues[:5]:
            print(f"           - {iss}")

    return all_pass


# ---------------------------------------------------------------------------
# 4. Command Diversity Analysis
# ---------------------------------------------------------------------------

def extract_tool(cmd):
    """Extract the first word/tool from a command string."""
    cmd = cmd.strip()
    if cmd.startswith("sudo "):
        cmd = cmd[5:].strip()
    parts = cmd.split()
    return parts[0] if parts else "unknown"


def command_diversity(data):
    print("\n" + "=" * 60)
    print("4. COMMAND DIVERSITY")
    print("=" * 60)

    # Tools in original commands
    orig_tools = [extract_tool(d["interaction"]["original_command"]) for d in data if "interaction" in d]
    tool_counts = Counter(orig_tools)
    print(f"Unique tools in original_command: {len(tool_counts)}")
    print("Top 15 tools:")
    for tool, cnt in tool_counts.most_common(15):
        print(f"  {tool:<25} {cnt:>5}")

    # Tool usage per taxonomy
    tax_tools = defaultdict(Counter)
    for d in data:
        tax = d.get("taxonomy", "?")
        tool = extract_tool(d.get("interaction", {}).get("original_command", ""))
        tax_tools[tax][tool] += 1

    # Command length stats
    lengths = [len(d["interaction"]["original_command"]) for d in data if "interaction" in d]
    if lengths:
        print(f"\nCommand length — min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)/len(lengths):.0f}")

    # Plot tool distribution
    top_tools = [t for t, _ in tool_counts.most_common(15)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_tools[::-1], [tool_counts[t] for t in top_tools[::-1]], color="#8172B2")
    ax.set_xlabel("Frequency")
    ax.set_title("Top 15 Tools in original_command")
    fig.tight_layout()
    save_fig(fig, "tool_diversity.png")

    # Heatmap: taxonomy × tool
    all_taxes = EXPECTED_TAXONOMIES
    top10 = [t for t, _ in tool_counts.most_common(10)]
    matrix = []
    for tax in all_taxes:
        row = [tax_tools[tax].get(tool, 0) for tool in top10]
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(top10)))
    ax.set_xticklabels(top10, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(all_taxes)))
    ax.set_yticklabels([t[:30] for t in all_taxes], fontsize=8)
    ax.set_title("Tool Usage Heatmap (Taxonomy × Tool)")
    fig.colorbar(im, ax=ax, label="Count")
    # annotate cells
    for i in range(len(all_taxes)):
        for j in range(len(top10)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    save_fig(fig, "tool_taxonomy_heatmap.png")

    return tool_counts


# ---------------------------------------------------------------------------
# 5. Error Output Analysis
# ---------------------------------------------------------------------------

def error_output_analysis(data):
    print("\n" + "=" * 60)
    print("5. ERROR OUTPUT ANALYSIS")
    print("=" * 60)

    tax_lengths = defaultdict(list)
    for d in data:
        tax = d.get("taxonomy", "?")
        err = d.get("interaction", {}).get("error_output", "")
        tax_lengths[tax].append(len(err))

    for tax in EXPECTED_TAXONOMIES:
        lengths = tax_lengths.get(tax, [])
        if lengths:
            avg = sum(lengths) / len(lengths)
            print(f"  {tax[:45]:<47} avg_len={avg:.0f}  min={min(lengths)}  max={max(lengths)}")

    # Box plot of error message lengths per taxonomy
    fig, ax = plt.subplots(figsize=(10, 5))
    box_data = [tax_lengths.get(t, []) for t in EXPECTED_TAXONOMIES]
    labels = [t[:30] for t in EXPECTED_TAXONOMIES]
    if any(box_data):
        ax.boxplot(box_data, labels=labels, vert=False)
        ax.set_xlabel("Error Output Length (chars)")
        ax.set_title("Error Message Length Distribution by Taxonomy")
    fig.tight_layout()
    save_fig(fig, "error_length_boxplot.png")

    # Word frequency across all error outputs
    all_words = []
    for d in data:
        err = d.get("interaction", {}).get("error_output", "")
        words = re.findall(r"[a-zA-Z]{3,}", err.lower())
        all_words.extend(words)
    wc = Counter(all_words)
    stopwords = {"the", "and", "for", "not", "this", "that", "from", "with", "was", "are", "but", "has", "have"}
    wc = Counter({w: c for w, c in wc.items() if w not in stopwords})

    print(f"\nTop 20 words in error_output:")
    for w, c in wc.most_common(20):
        print(f"  {w:<20} {c:>5}")

    # Plot
    top_words = wc.most_common(20)
    if top_words:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh([w for w, _ in top_words][::-1], [c for _, c in top_words][::-1], color="#CCB974")
        ax.set_xlabel("Frequency")
        ax.set_title("Top 20 Words in Error Output")
        fig.tight_layout()
        save_fig(fig, "error_word_freq.png")


# ---------------------------------------------------------------------------
# 6. State Snapshot Analysis
# ---------------------------------------------------------------------------

def state_snapshot_analysis(data):
    print("\n" + "=" * 60)
    print("6. STATE SNAPSHOT ANALYSIS")
    print("=" * 60)

    key_values = defaultdict(list)
    parse_fails = 0

    for d in data:
        snap = d.get("state_snapshot", "")
        if isinstance(snap, str):
            try:
                obj = json.loads(snap)
            except (json.JSONDecodeError, TypeError):
                parse_fails += 1
                continue
        elif isinstance(snap, dict):
            obj = snap
        else:
            parse_fails += 1
            continue

        for k, v in obj.items():
            if isinstance(v, (int, float)):
                key_values[k].append(v)

    print(f"Parsed snapshots: {len(data) - parse_fails}/{len(data)}")
    if parse_fails:
        print(f"Parse failures: {parse_fails}")

    if not key_values:
        print("  No numeric keys found in state_snapshot.")
        return

    print(f"Numeric keys found: {list(key_values.keys())}")
    for k, vals in sorted(key_values.items()):
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {k:<30} n={len(vals):>5}  avg={avg:>8.1f}  min={min(vals):>8.1f}  max={max(vals):>8.1f}")

    # Plot histograms for top keys
    top_keys = sorted(key_values.keys(), key=lambda k: len(key_values[k]), reverse=True)[:6]
    if top_keys:
        ncols = min(3, len(top_keys))
        nrows = (len(top_keys) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        flat_axes = axes.flat  # safe 1-D iterator regardless of shape
        for idx, key in enumerate(top_keys):
            ax = flat_axes[idx]
            ax.hist(key_values[key], bins=20, color="#64B5F6", edgecolor="white")
            ax.set_title(key, fontsize=10)
            ax.set_xlabel("Value")
            ax.set_ylabel("Freq")
        # hide unused subplot slots
        for idx in range(len(top_keys), nrows * ncols):
            flat_axes[idx].set_visible(False)
        fig.suptitle("State Snapshot Numeric Distributions", fontsize=13)
        fig.tight_layout()
        save_fig(fig, "state_snapshot_histograms.png")


# ---------------------------------------------------------------------------
# 7. Duplicate Detection
# ---------------------------------------------------------------------------

def duplicate_detection(data):
    print("\n" + "=" * 60)
    print("7. DUPLICATE DETECTION")
    print("=" * 60)

    # Exact duplicate on (original_command, error_output)
    seen = Counter()
    for d in data:
        inter = d.get("interaction", {})
        key = (inter.get("original_command", ""), inter.get("error_output", ""))
        seen[key] += 1

    exact_dups = {k: v for k, v in seen.items() if v > 1}
    print(f"Exact duplicate (command + error_output) pairs: {len(exact_dups)}")
    if exact_dups:
        print("Top 5 most repeated:")
        for (cmd, _), cnt in sorted(exact_dups.items(), key=lambda x: -x[1])[:5]:
            print(f"  [{cnt}x] {cmd[:80]}")

    # Exact duplicate on original_command only
    cmd_counts = Counter(d.get("interaction", {}).get("original_command", "") for d in data)
    cmd_dups = {k: v for k, v in cmd_counts.items() if v > 1}
    print(f"\nDuplicate original_command (ignoring error): {len(cmd_dups)}")
    if cmd_dups:
        print("Top 5:")
        for cmd, cnt in sorted(cmd_dups.items(), key=lambda x: -x[1])[:5]:
            print(f"  [{cnt}x] {cmd[:80]}")


# ---------------------------------------------------------------------------
# Summary Report (text file)
# ---------------------------------------------------------------------------

def write_summary(data, tax_counts, all_pass, tool_counts):
    path = os.path.join(OUTPUT_DIR, "summary_report.txt")
    with open(path, "w") as f:
        f.write(f"Generated Errors Analysis Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Total scenarios: {len(data)}\n")
        total_target = TARGET_PER_TAXONOMY * len(EXPECTED_TAXONOMIES)
        f.write(f"Completeness: {len(data)}/{total_target} ({100*len(data)/total_target:.1f}%)\n\n")

        f.write("Per taxonomy:\n")
        for tax in EXPECTED_TAXONOMIES:
            c = tax_counts.get(tax, 0)
            f.write(f"  {tax}: {c}/{TARGET_PER_TAXONOMY}\n")

        f.write(f"\nQuality checks: {'ALL PASSED' if all_pass else 'SOME FAILED — see console output'}\n")
        f.write(f"Unique tools: {len(tool_counts)}\n")
        f.write(f"Top 5 tools: {', '.join(t for t, _ in tool_counts.most_common(5))}\n")

    print(f"\n  Summary report saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading data from: {DATA_PATH}")
    data = load_data(DATA_PATH)
    ensure_output_dir()

    tax_counts = dataset_summary(data)
    plot_taxonomy_distribution(tax_counts)
    plot_exit_code_distribution(data)
    plot_fix_exit_codes(data)
    all_pass = quality_checks(data)
    tool_counts = command_diversity(data)
    error_output_analysis(data)
    state_snapshot_analysis(data)
    duplicate_detection(data)
    write_summary(data, tax_counts, all_pass, tool_counts)

    print("\n" + "=" * 60)
    print("DONE — all plots saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
