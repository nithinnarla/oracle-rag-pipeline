"""
ORACLE Stage 4 Ablation: full_text FK vs question-only FK

Tests whether scoring readability on the concatenated question+answer
(full_text, ORACLE's current corpus-build approach) versus the question
alone changes literacy-band routing correctness.

Background: Decision 1/3 in methodology_decisions.md flagged this as an
open risk, never validated. This script closes that gap empirically.

Method: for every query in the actual cross-dataset evaluation set,
recompute band assignment using classify_query() on the question text
alone (the real production classifier, not a reimplementation), and
compare against the routing decision that used the corpus's full_text
FK at build time.
"""
import sys
import os
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from literacy_classifier import classify_query

RESULTS_PATH = os.path.join(REPO_ROOT, "data", "processed", "cross_dataset_results.csv")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "stage4")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")


def run_ablation():
    results = pd.read_csv(RESULTS_PATH)
    actual = results[results["condition"] == "actual"].copy()

    actual["band_question_only"] = actual["query"].apply(lambda q: classify_query(q)["band"])
    actual["fk_question_only"] = actual["query"].apply(lambda q: classify_query(q)["fk_grade"])
    actual["margin_question_only"] = actual["query"].apply(lambda q: classify_query(q)["margin"])
    actual["band_match_question_only"] = actual["target_band"] == actual["band_question_only"]

    total = len(actual)
    agree = (actual["band_match"] == actual["band_match_question_only"]).sum()
    disagree = total - agree

    flipped = actual[actual["band_match"] != actual["band_match_question_only"]]
    became_wrong = flipped[flipped["band_match_question_only"] == False]
    became_correct = flipped[flipped["band_match_question_only"] == True]

    print(f"Total records evaluated: {total}")
    print(f"Agreement between full_text and question-only routing: {agree} ({agree/total*100:.1f}%)")
    print(f"Disagreement: {disagree} ({disagree/total*100:.1f}%)")
    print(f"  Correct under full_text, wrong under question-only: {len(became_wrong)}")
    print(f"  Wrong under full_text, correct under question-only: {len(became_correct)}")

    avg_margin_flipped = flipped["margin_question_only"].abs().mean()
    avg_margin_stable = actual.loc[~actual.index.isin(flipped.index), "margin_question_only"].abs().mean()
    print(f"\nMean threshold margin, flipped records: {avg_margin_flipped:.2f}")
    print(f"Mean threshold margin, stable records: {avg_margin_stable:.2f}")

    actual.to_csv(os.path.join(DOCS_DIR, "fk_ablation_results.csv"), index=False)
    print(f"\nSaved: docs/fk_ablation_results.csv")

    return actual, {
        "total": total, "agree": agree, "disagree": disagree,
        "became_wrong": len(became_wrong), "became_correct": len(became_correct),
        "avg_margin_flipped": avg_margin_flipped, "avg_margin_stable": avg_margin_stable
    }


def plot_ablation(actual, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Agree", "Correct to Wrong", "Wrong to Correct"]
    values = [stats["agree"], stats["became_wrong"], stats["became_correct"]]
    colors = ["#4C72B0", "#C44E52", "#55A868"]
    axes[0].bar(labels, values, color=colors)
    axes[0].set_title("full_text FK vs question-only FK\nRouting agreement")
    axes[0].set_ylabel("Record count")
    for i, v in enumerate(values):
        axes[0].text(i, v + 3, str(v), ha="center")

    flipped = actual[actual["band_match"] != actual["band_match_question_only"]]
    stable = actual.loc[~actual.index.isin(flipped.index)]
    axes[1].hist(stable["margin_question_only"].abs(), bins=20, alpha=0.6, label="Stable", color="#4C72B0")
    axes[1].hist(flipped["margin_question_only"].abs(), bins=20, alpha=0.6, label="Flipped", color="#C44E52")
    axes[1].set_title("Distance from nearest FK threshold\n(flipped records cluster near 0)")
    axes[1].set_xlabel("Margin (FK grade points)")
    axes[1].set_ylabel("Record count")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fk_ablation_full_text_vs_question_only.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    actual, stats = run_ablation()
    plot_ablation(actual, stats)


def plot_source_breakdown(actual):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flipped = actual[actual["band_match"] != actual["band_match_question_only"]]
    total_by_source = actual["source"].value_counts()
    flip_by_source = flipped["source"].value_counts()
    rate = (flip_by_source / total_by_source * 100).fillna(0.0).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C44E52" if v > 20 else "#DD8452" if v > 5 else "#4C72B0" for v in rate.values]
    bars = ax.bar(rate.index, rate.values, color=colors)
    ax.set_title("Routing disagreement rate by source\n(full_text FK vs question-only FK)")
    ax.set_ylabel("Flip rate (%)")
    ax.set_xlabel("Source")
    for bar, val in zip(bars, rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%", ha="center")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "fk_ablation_flip_rate_by_source.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    return rate


def run_publication_extensions(actual):
    """
    Three additions to move this from a documented finding to a
    publication-ready result: (1) test the length-mechanism hypothesis
    directly rather than infer it from two data points, (2) check
    whether flipped records actually show worse generation quality,
    closing the real question Decision 1/3 asked (retrieval quality,
    not just routing correctness), (3) confirm source concentration
    is statistically real, not small-sample noise.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pointbiserialr, mannwhitneyu, chi2_contingency

    flipped = actual["band_match"] != actual["band_match_question_only"]
    actual = actual.copy()
    actual["flipped"] = flipped
    actual["query_word_count"] = actual["query"].str.split().str.len()

    print("=== 1. Flip rate vs query length ===")
    corr, p_corr = pointbiserialr(actual["flipped"].astype(int), actual["query_word_count"])
    print(f"Point-biserial correlation (flipped vs word count): r={corr:.3f}, p={p_corr:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(actual.loc[~actual["flipped"], "query_word_count"],
               [0] * (~actual["flipped"]).sum(), alpha=0.3, s=20, color="#4C72B0", label="Stable")
    ax.scatter(actual.loc[actual["flipped"], "query_word_count"],
               [1] * actual["flipped"].sum(), alpha=0.5, s=20, color="#C44E52", label="Flipped")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Stable", "Flipped"])
    ax.set_xlabel("Query word count")
    ax.set_title(f"Flip status vs query length\n(r={corr:.3f}, p={p_corr:.4f})")
    ax.legend()
    plt.tight_layout()
    out1 = os.path.join(FIGURES_DIR, "fk_ablation_length_correlation.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Saved: {out1}")

    print("\n=== 2. Generation quality: flipped vs stable ===")
    flip_fk = actual.loc[actual["flipped"], "fk_reduction"].dropna()
    stable_fk = actual.loc[~actual["flipped"], "fk_reduction"].dropna()
    stat_fk, p_fk = mannwhitneyu(flip_fk, stable_fk, alternative="two-sided")
    print(f"FK reduction: flipped mean={flip_fk.mean():.3f} (n={len(flip_fk)}), "
          f"stable mean={stable_fk.mean():.3f} (n={len(stable_fk)}), "
          f"Mann-Whitney p={p_fk:.4f}")

    flip_rouge = actual.loc[actual["flipped"], "rouge_l"].dropna()
    stable_rouge = actual.loc[~actual["flipped"], "rouge_l"].dropna()
    stat_r, p_r = mannwhitneyu(flip_rouge, stable_rouge, alternative="two-sided")
    print(f"ROUGE-L: flipped mean={flip_rouge.mean():.3f} (n={len(flip_rouge)}), "
          f"stable mean={stable_rouge.mean():.3f} (n={len(stable_rouge)}), "
          f"Mann-Whitney p={p_r:.4f}")

    print("\n=== 3. Source concentration: chi-square test ===")
    contingency = pd.crosstab(actual["source"], actual["flipped"])
    chi2, p_chi, dof, expected = chi2_contingency(contingency)
    print(f"Chi-square: chi2={chi2:.2f}, dof={dof}, p={p_chi:.6f}")
    print(contingency)

    return {
        "length_corr": corr, "length_corr_p": p_corr,
        "fk_flip_mean": flip_fk.mean(), "fk_stable_mean": stable_fk.mean(), "fk_p": p_fk,
        "rouge_flip_mean": flip_rouge.mean(), "rouge_stable_mean": stable_rouge.mean(), "rouge_p": p_r,
        "chi2": chi2, "chi2_p": p_chi
    }
