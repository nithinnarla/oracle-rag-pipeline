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
