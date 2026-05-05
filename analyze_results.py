#!/usr/bin/env python3
"""
Comprehensive analysis of baseline evaluation results.
Reads a results JSON file and produces a detailed metrics report.

Usage:
    python3 analyze_results.py
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_FILE = "SFT/results/sft_v1_evaluation_20260504_134740.json"
OUTPUT_DIR = "SFT/results"
# ==========================================


def analyze(data):
    total = len(data)
    report_lines = []

    def add(text=""):
        report_lines.append(text)

    add("=" * 70)
    add(f"  BASELINE EVALUATION REPORT")
    add(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"  Source: {RESULTS_FILE}")
    add("=" * 70)
    add()

    # -------------------------------------------------------
    # 1. OVERALL SUMMARY
    # -------------------------------------------------------
    add("1. OVERALL SUMMARY")
    add("-" * 40)
    add(f"  Total test samples:        {total}")

    has_llm_eval = sum(1 for x in data if "llm_evaluation" in x)
    add(f"  LLM-evaluated samples:     {has_llm_eval}/{total}")
    add()

    # -------------------------------------------------------
    # 2. EXACT MATCH & SUBSTRING ACCURACY (per rank)
    # -------------------------------------------------------
    add("2. EXACT MATCH & SUBSTRING ACCURACY")
    add("-" * 40)

    for rank in [1, 2, 3]:
        key = f"rank_{rank}"
        exact_count = 0
        substring_count = 0
        counted = 0

        for item in data:
            if key in item.get("matches", {}):
                counted += 1
                m = item["matches"][key]
                if m.get("exact", False):
                    exact_count += 1
                if m.get("is_substring", False):
                    substring_count += 1

        if counted > 0:
            add(f"  Rank {rank}:")
            add(f"    Exact Match:    {exact_count}/{counted} ({exact_count/counted*100:.1f}%)")
            add(f"    Substring:      {substring_count}/{counted} ({substring_count/counted*100:.1f}%)")
    add()

    # -------------------------------------------------------
    # 3. JACCARD SIMILARITY (per rank)
    # -------------------------------------------------------
    add("3. JACCARD SIMILARITY (per rank)")
    add("-" * 40)

    for rank in [1, 2, 3]:
        key = f"rank_{rank}"
        jaccards = []
        for item in data:
            if key in item.get("matches", {}):
                jaccards.append(item["matches"][key].get("jaccard", 0.0))

        if jaccards:
            avg = sum(jaccards) / len(jaccards)
            sorted_j = sorted(jaccards)
            median = sorted_j[len(sorted_j) // 2]
            top10 = sum(1 for j in jaccards if j >= 0.1) / len(jaccards) * 100
            top50 = sum(1 for j in jaccards if j >= 0.5) / len(jaccards) * 100
            top80 = sum(1 for j in jaccards if j >= 0.8) / len(jaccards) * 100
            perfect = sum(1 for j in jaccards if j == 1.0) / len(jaccards) * 100

            add(f"  Rank {rank}:")
            add(f"    Average Jaccard:     {avg*100:.1f}%")
            add(f"    Median Jaccard:      {median*100:.1f}%")
            add(f"    Min / Max:           {min(jaccards)*100:.1f}% / {max(jaccards)*100:.1f}%")
            add(f"    Samples >= 10%:      {top10:.1f}%")
            add(f"    Samples >= 50%:      {top50:.1f}%")
            add(f"    Samples >= 80%:      {top80:.1f}%")
            add(f"    Perfect (100%):      {perfect:.1f}%")
    add()

    # -------------------------------------------------------
    # 4. LLM JUDGE EVALUATION
    # -------------------------------------------------------
    add("4. LLM JUDGE EVALUATION")
    add("-" * 40)

    eval_items = [x for x in data if "llm_evaluation" in x]
    eval_total = len(eval_items)

    if eval_total == 0:
        add("  No LLM evaluations found.")
    else:
        # Matched rank distribution
        rank_counts = Counter()
        safe_count = 0
        unsafe_count = 0

        for item in eval_items:
            ev = item["llm_evaluation"]
            matched = ev.get("matched_rank")
            rank_counts[matched] += 1

            if ev.get("is_safe") is True:
                safe_count += 1
            elif ev.get("is_safe") is False:
                unsafe_count += 1

        add(f"  Total LLM-evaluated:       {eval_total}")
        add()
        add(f"  Semantic Match Distribution:")
        for rank in [1, 2, 3]:
            c = rank_counts.get(rank, 0)
            add(f"    Matched Rank {rank}:       {c}/{eval_total} ({c/eval_total*100:.1f}%)")
        no_match = rank_counts.get(None, 0) + rank_counts.get("null", 0)
        add(f"    No Match (null):        {no_match}/{eval_total} ({no_match/eval_total*100:.1f}%)")
        any_match = eval_total - no_match
        add(f"    Any Match (1+2+3):      {any_match}/{eval_total} ({any_match/eval_total*100:.1f}%)")
        add()
        add(f"  Safety Assessment:")
        add(f"    Safe predictions:       {safe_count}/{eval_total} ({safe_count/eval_total*100:.1f}%)")
        add(f"    Unsafe predictions:     {unsafe_count}/{eval_total} ({unsafe_count/eval_total*100:.1f}%)")
        add()

        # Similarity scores
        sim_scores = [ev.get("similarity_score") for x in eval_items 
                      for ev in [x["llm_evaluation"]] if ev.get("similarity_score") is not None]
        if sim_scores:
            avg_sim = sum(sim_scores) / len(sim_scores)
            sorted_sim = sorted(sim_scores)
            median_sim = sorted_sim[len(sorted_sim) // 2]
            add(f"  Similarity Score (0-10):")
            add(f"    Average:                {avg_sim:.1f}")
            add(f"    Median:                 {median_sim}")
            add(f"    Min / Max:              {min(sim_scores)} / {max(sim_scores)}")
            add(f"    Scores >= 7 (good):     {sum(1 for s in sim_scores if s >= 7)}/{len(sim_scores)} ({sum(1 for s in sim_scores if s >= 7)/len(sim_scores)*100:.1f}%)")
            add(f"    Scores >= 5 (partial):  {sum(1 for s in sim_scores if s >= 5)}/{len(sim_scores)} ({sum(1 for s in sim_scores if s >= 5)/len(sim_scores)*100:.1f}%)")
            add(f"    Scores <= 2 (bad):      {sum(1 for s in sim_scores if s <= 2)}/{len(sim_scores)} ({sum(1 for s in sim_scores if s <= 2)/len(sim_scores)*100:.1f}%)")
    add()

    # -------------------------------------------------------
    # 5. TAXONOMY BREAKDOWN
    # -------------------------------------------------------
    add("5. TAXONOMY BREAKDOWN")
    add("-" * 40)

    taxonomy_stats = defaultdict(lambda: {
        "total": 0,
        "exact_rank1": 0,
        "avg_jaccard_rank1": [],
        "llm_matched": 0,
        "llm_no_match": 0,
        "llm_safe": 0,
        "llm_unsafe": 0,
    })

    for item in data:
        tax = item.get("taxonomy", "Unknown")
        stats = taxonomy_stats[tax]
        stats["total"] += 1

        r1 = item.get("matches", {}).get("rank_1", {})
        if r1.get("exact", False):
            stats["exact_rank1"] += 1
        stats["avg_jaccard_rank1"].append(r1.get("jaccard", 0.0))

        if "llm_evaluation" in item:
            ev = item["llm_evaluation"]
            if ev.get("matched_rank") is not None:
                stats["llm_matched"] += 1
            else:
                stats["llm_no_match"] += 1
            if ev.get("is_safe") is True:
                stats["llm_safe"] += 1
            elif ev.get("is_safe") is False:
                stats["llm_unsafe"] += 1

    # Sort by total count descending
    sorted_taxonomies = sorted(taxonomy_stats.items(), key=lambda x: x[1]["total"], reverse=True)

    add(f"  {'Taxonomy':<45} {'N':>4} {'Exact':>6} {'Jaccard':>8} {'LLM Match':>10} {'Safe':>6}")
    add(f"  {'-'*45} {'---':>4} {'-----':>6} {'-------':>8} {'---------':>10} {'----':>6}")

    for tax, stats in sorted_taxonomies:
        n = stats["total"]
        exact = stats["exact_rank1"]
        avg_j = sum(stats["avg_jaccard_rank1"]) / len(stats["avg_jaccard_rank1"]) * 100 if stats["avg_jaccard_rank1"] else 0
        llm_m = stats["llm_matched"]
        safe = stats["llm_safe"]
        add(f"  {tax:<45} {n:>4} {exact:>5}  {avg_j:>6.1f}% {llm_m:>9}  {safe:>5}")

    add()

    # -------------------------------------------------------
    # 6. BEST & WORST EXAMPLES (by Rank 1 Jaccard)
    # -------------------------------------------------------
    add("6. TOP 5 BEST PREDICTIONS (by Rank 1 Jaccard)")
    add("-" * 40)

    scored = []
    for item in data:
        r1 = item.get("matches", {}).get("rank_1", {})
        scored.append((r1.get("jaccard", 0.0), item))

    scored.sort(key=lambda x: x[0], reverse=True)

    for j, item in scored[:5]:
        add(f"  ID: {item['id']}")
        add(f"    Taxonomy:   {item['taxonomy']}")
        add(f"    Predicted:  {item['predicted_command'][:80]}")
        r1 = item["matches"].get("rank_1", {})
        add(f"    Gold (R1):  {r1.get('command', 'N/A')[:80]}")
        add(f"    Jaccard:    {j*100:.1f}%")
        if "llm_evaluation" in item:
            add(f"    LLM Judge:  Rank {item['llm_evaluation'].get('matched_rank')} | Safe: {item['llm_evaluation'].get('is_safe')}")
        add()

    add("7. TOP 5 WORST PREDICTIONS (by Rank 1 Jaccard)")
    add("-" * 40)

    for j, item in scored[-5:]:
        add(f"  ID: {item['id']}")
        add(f"    Taxonomy:   {item['taxonomy']}")
        add(f"    Predicted:  {item['predicted_command'][:80]}")
        r1 = item["matches"].get("rank_1", {})
        add(f"    Gold (R1):  {r1.get('command', 'N/A')[:80]}")
        add(f"    Jaccard:    {j*100:.1f}%")
        if "llm_evaluation" in item:
            add(f"    LLM Judge:  Rank {item['llm_evaluation'].get('matched_rank')} | Safe: {item['llm_evaluation'].get('is_safe')}")
        add()

    # -------------------------------------------------------
    # 7. CROSS-METRIC AGREEMENT
    # -------------------------------------------------------
    add("8. CROSS-METRIC AGREEMENT (Jaccard vs LLM Judge)")
    add("-" * 40)

    # Cases where Jaccard >= 0.5 on Rank 1 but LLM says no match
    high_jaccard_no_llm = 0
    low_jaccard_llm_match = 0

    for item in data:
        r1 = item.get("matches", {}).get("rank_1", {})
        j = r1.get("jaccard", 0.0)
        if "llm_evaluation" in item:
            llm_rank = item["llm_evaluation"].get("matched_rank")
            if j >= 0.5 and llm_rank is None:
                high_jaccard_no_llm += 1
            if j < 0.2 and llm_rank == 1:
                low_jaccard_llm_match += 1

    add(f"  High Jaccard (>=50%) but LLM says No Match:  {high_jaccard_no_llm}")
    add(f"  Low Jaccard (<20%) but LLM says Rank 1:      {low_jaccard_llm_match}")
    add(f"  (These numbers show where string metrics disagree with semantic evaluation)")
    add()

    add("=" * 70)
    add("  END OF REPORT")
    add("=" * 70)

    return "\n".join(report_lines)


def generate_plots(data, output_dir, basename):
    """Generate publication-quality plots from evaluation data."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Consistent styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })
    colors = ['#2ecc71', '#e67e22', '#e74c3c', '#95a5a6']
    rank_colors = ['#3498db', '#e67e22', '#e74c3c']

    # ---- Plot 1: LLM Judge Semantic Match Distribution (Pie) ----
    eval_items = [x for x in data if "llm_evaluation" in x]
    if eval_items:
        rank_counts = Counter()
        for item in eval_items:
            matched = item["llm_evaluation"].get("matched_rank")
            rank_counts[matched] += 1

        labels = ['Rank 1\n(Best Practice)', 'Rank 2\n(Risky Bypass)', 'Rank 3\n(Fallback)', 'No Match']
        sizes = [rank_counts.get(1, 0), rank_counts.get(2, 0), rank_counts.get(3, 0), rank_counts.get(None, 0)]
        explode = (0.05, 0.05, 0.05, 0.1)

        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                           autopct='%1.1f%%', shadow=False, startangle=90,
                                           textprops={'fontsize': 11})
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax.set_title('LLM Judge: Semantic Match Distribution', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'1_llm_judge_match_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # ---- Plot 2: Jaccard Distribution Histogram (per rank) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for idx, rank in enumerate([1, 2, 3]):
        key = f"rank_{rank}"
        jaccards = [item["matches"][key].get("jaccard", 0.0)
                    for item in data if key in item.get("matches", {})]
        if jaccards:
            axes[idx].hist(jaccards, bins=20, color=rank_colors[idx], alpha=0.85, edgecolor='white')
            axes[idx].axvline(np.mean(jaccards), color='black', linestyle='--', linewidth=1.5,
                              label=f'Mean: {np.mean(jaccards)*100:.1f}%')
            axes[idx].set_title(f'Rank {rank}', fontweight='bold')
            axes[idx].set_xlabel('Jaccard Similarity')
            axes[idx].legend(fontsize=10)
    axes[0].set_ylabel('Number of Samples')
    fig.suptitle('Jaccard Similarity Distribution by Rank', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'2_jaccard_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Plot 3: Taxonomy Breakdown (Grouped Bar) ----
    taxonomy_stats = defaultdict(lambda: {"jaccards": [], "llm_matched": 0, "llm_total": 0, "safe": 0, "total": 0})
    for item in data:
        tax = item.get("taxonomy", "Unknown")
        stats = taxonomy_stats[tax]
        stats["total"] += 1
        r1 = item.get("matches", {}).get("rank_1", {})
        stats["jaccards"].append(r1.get("jaccard", 0.0))
        if "llm_evaluation" in item:
            stats["llm_total"] += 1
            if item["llm_evaluation"].get("matched_rank") is not None:
                stats["llm_matched"] += 1
            if item["llm_evaluation"].get("is_safe") is True:
                stats["safe"] += 1

    tax_names = []
    avg_jaccards = []
    llm_match_rates = []
    safe_rates = []

    for tax, stats in sorted(taxonomy_stats.items()):
        short_name = tax.split('/')[-1].strip() if '/' in tax else tax
        # Wrap long names
        if len(short_name) > 20:
            short_name = short_name[:18] + '...'
        tax_names.append(short_name)
        avg_jaccards.append(np.mean(stats["jaccards"]) * 100 if stats["jaccards"] else 0)
        llm_match_rates.append(stats["llm_matched"] / stats["llm_total"] * 100 if stats["llm_total"] > 0 else 0)
        safe_rates.append(stats["safe"] / stats["llm_total"] * 100 if stats["llm_total"] > 0 else 0)

    x = np.arange(len(tax_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, avg_jaccards, width, label='Avg Jaccard (R1)', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x, llm_match_rates, width, label='LLM Match Rate', color='#2ecc71', alpha=0.85)
    bars3 = ax.bar(x + width, safe_rates, width, label='Safe Rate', color='#e67e22', alpha=0.85)

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Metrics by Error Taxonomy', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tax_names, rotation=15, ha='right', fontsize=10)
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'3_taxonomy_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Plot 4: Safety Assessment (Donut) ----
    if eval_items:
        safe = sum(1 for x in eval_items if x["llm_evaluation"].get("is_safe") is True)
        unsafe = sum(1 for x in eval_items if x["llm_evaluation"].get("is_safe") is False)

        fig, ax = plt.subplots(figsize=(6, 6))
        sizes = [safe, unsafe]
        labels_s = [f'Safe\n({safe})', f'Unsafe\n({unsafe})']
        wedges, texts, autotexts = ax.pie(sizes, labels=labels_s, colors=['#2ecc71', '#e74c3c'],
                                           autopct='%1.1f%%', startangle=90, pctdistance=0.75,
                                           textprops={'fontsize': 12})
        centre_circle = plt.Circle((0, 0), 0.50, fc='white')
        ax.add_artist(centre_circle)
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax.set_title('Safety Assessment of Predictions', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'4_safety_assessment.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # ---- Plot 5: Exact Match / Substring / Jaccard >= 50% comparison (Bar) ----
    metrics_per_rank = {}
    for rank in [1, 2, 3]:
        key = f"rank_{rank}"
        exact = sum(1 for item in data if key in item.get("matches", {}) and item["matches"][key].get("exact", False))
        substr = sum(1 for item in data if key in item.get("matches", {}) and item["matches"][key].get("is_substring", False))
        j50 = sum(1 for item in data if key in item.get("matches", {}) and item["matches"][key].get("jaccard", 0) >= 0.5)
        metrics_per_rank[rank] = (exact, substr, j50)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    width = 0.22

    exact_vals = [metrics_per_rank[r][0] for r in [1, 2, 3]]
    substr_vals = [metrics_per_rank[r][1] for r in [1, 2, 3]]
    j50_vals = [metrics_per_rank[r][2] for r in [1, 2, 3]]

    ax.bar(x - width, exact_vals, width, label='Exact Match', color='#3498db')
    ax.bar(x, substr_vals, width, label='Substring Match', color='#9b59b6')
    ax.bar(x + width, j50_vals, width, label='Jaccard >= 50%', color='#e67e22')

    ax.set_ylabel('Number of Samples')
    ax.set_title('String-Based Metrics Comparison by Rank', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Rank 1\n(Best Practice)', 'Rank 2\n(Risky Bypass)', 'Rank 3\n(Fallback)'])
    ax.legend()

    # Add value labels on bars
    for bar_group in [ax.patches]:
        pass
    for rect in ax.patches:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'5_string_metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Plot 6: LLM Similarity Score Distribution ----
    sim_scores = [item["llm_evaluation"].get("similarity_score")
                  for item in data if "llm_evaluation" in item
                  and item["llm_evaluation"].get("similarity_score") is not None]
    if sim_scores:
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.arange(-0.5, 11.5, 1)  # bins centered on 0-10
        counts, _, patches = ax.hist(sim_scores, bins=bins, color='#3498db', alpha=0.85, edgecolor='white')

        # Color code: red for low, orange for medium, green for high
        for i, patch in enumerate(patches):
            if i <= 2:
                patch.set_facecolor('#e74c3c')  # red
            elif i <= 4:
                patch.set_facecolor('#e67e22')  # orange
            elif i <= 6:
                patch.set_facecolor('#f1c40f')  # yellow
            else:
                patch.set_facecolor('#2ecc71')  # green

        avg_sim = np.mean(sim_scores)
        ax.axvline(avg_sim, color='black', linestyle='--', linewidth=2, label=f'Mean: {avg_sim:.1f}')
        ax.set_xlabel('Similarity Score (0 = wrong, 10 = identical)')
        ax.set_ylabel('Number of Samples')
        ax.set_title('LLM Judge: Similarity Score Distribution', fontweight='bold')
        ax.set_xticks(range(11))
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'6_similarity_score_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Plots saved to: {plots_dir}/")


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"File not found: {RESULTS_FILE}")
        return

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    report = analyze(data)

    # Print to console
    print(report)

    # Save to file
    basename = os.path.splitext(os.path.basename(RESULTS_FILE))[0]
    output_file = os.path.join(OUTPUT_DIR, f"report_{basename}.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {output_file}")

    # Generate plots
    generate_plots(data, OUTPUT_DIR, basename)


if __name__ == "__main__":
    main()
