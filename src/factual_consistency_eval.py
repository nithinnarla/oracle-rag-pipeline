"""
ORACLE - Stage 4: Factual Consistency Evaluation (Adapted from PlainQAFact)
Phase 4 - Stage 4: Evaluation metrics (Decision 7: PlainQAFact + APPLS)

IMPORTANT - READ BEFORE CITING: This is an ADAPTED evaluation, not the
official PlainQAFact metric (You & Guo, 2025, arXiv 2503.08890, accepted JBI).
The official pipeline requires Llama 3.1 8B Instruct locally (40GB+ GPU
memory) and a separate LERC scoring model - infeasible on this machine
(Apple Silicon, no CUDA). Installing the official `plainqafact` pip package
was also rejected: its dependency tree (torch 2.13.0, transformers 4.44.2,
pyserini, faiss-cpu, nmslib, spacy) conflicts with versions already verified
working elsewhere in this project (torch 2.10.0, transformers 4.57.6, used
by mbert_classifier.py in the sibling HyDMIS repo) and was never intended to
share an environment with an existing project per the official repo's own
isolated-conda-env setup instructions.

This script borrows PlainQAFact's real conceptual approach - classify
sentence type, extract a claim, verify it against the source, score
consistency - but substitutes GPT-4o-mini for every model in the original
pipeline (their fine-tuned classifier, Llama 3.1 8B for answer extraction,
BART for question generation, and their QA/LERC scoring models). Results
from this script are NOT directly comparable to PlainQAFact's published
benchmarks and must be labeled as an adapted evaluation, not the official
metric, wherever reported.

UPDATE (Aug 6 2026): the official PlainQAFact metric has since been run
separately on a cloud GPU (RunPod, RTX PRO 6000), on the same 20 source_text/
generated_summary pairs this script evaluates. Result: internal_mean~0.65,
external_mean~0.26, overall_mean~0.33 - substantially different from this
script's ~0.96, NOT because either is broken, but because the two measure
factual consistency against different ground truth (this script: the source
abstract; official PlainQAFact: external knowledge-base retrieval, which
struggles to find fact-specific matches for ORACLE's clinical-trial-specific
claims). See methodology_decisions.md Decision 13 for the complete
investigation, systematic evidence (234 claims), and what this means for
reporting both scores in the paper - they are not interchangeable and
should not be presented as one validating the other.

Pipeline/infrastructure script - no notebook (single quantitative
evaluation, matches methodology_decisions.md documentation pattern).
"""

import os
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "lay_summarizer_results.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "factual_consistency_results.csv")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "stage4")
MODEL = "gpt-4o-mini"

client = OpenAI()

CONSISTENCY_PROMPT = """You are evaluating factual consistency between a scientific abstract (source) and a simplified plain-language summary (generated), adapting the PlainQAFact methodology (You & Guo, 2025).

For the SUMMARY below, do the following:
1. Identify each distinct factual claim in the summary.
2. For each claim, classify it as either:
   - SIMPLIFICATION: directly restates something present in the source
   - ELABORATION: adds context/explanation not explicitly in the source (e.g. background, examples)
3. For each claim, judge whether it is factually CONSISTENT or INCONSISTENT with the source (for ELABORATION claims, judge based on general medical knowledge, since no external retrieval is used in this adapted version).
4. Return a JSON object with this exact structure:
{{
  "claims": [
    {{"claim": "...", "type": "SIMPLIFICATION or ELABORATION", "consistent": true or false}}
  ],
  "simplification_consistent": <count of consistent SIMPLIFICATION claims>,
  "simplification_total": <count of all SIMPLIFICATION claims>,
  "elaboration_consistent": <count of consistent ELABORATION claims>,
  "elaboration_total": <count of all ELABORATION claims>
}}

SOURCE (scientific abstract):
{source}

SUMMARY (plain language):
{summary}

Return ONLY the JSON object, no other text."""


def score_pair(source_text: str, generated_summary: str) -> dict:
    """Score one source/summary pair using the adapted methodology."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": CONSISTENCY_PROMPT.format(
                source=source_text, summary=generated_summary
            )}],
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())

        s_total = parsed.get("simplification_total", 0)
        s_consistent = parsed.get("simplification_consistent", 0)
        e_total = parsed.get("elaboration_total", 0)
        e_consistent = parsed.get("elaboration_consistent", 0)

        s_score = s_consistent / s_total if s_total > 0 else None
        e_score = e_consistent / e_total if e_total > 0 else None

        n_s = s_total
        n_e = e_total
        if n_s + n_e > 0:
            weighted_scores = []
            if s_score is not None:
                weighted_scores.append(s_score * n_s)
            if e_score is not None:
                weighted_scores.append(e_score * n_e)
            overall = sum(weighted_scores) / (n_s + n_e)
        else:
            overall = None

        return {
            "n_claims": len(parsed.get("claims", [])),
            "simplification_score": s_score,
            "elaboration_score": e_score,
            "overall_score": overall,
            "n_simplification": n_s,
            "n_elaboration": n_e,
            "consistency_error": None,
        }
    except Exception as e:
        return {
            "n_claims": None, "simplification_score": None, "elaboration_score": None,
            "overall_score": None, "n_simplification": None, "n_elaboration": None,
            "consistency_error": str(e),
        }


def run_factual_consistency_eval():
    print("ORACLE - Factual Consistency Evaluation (Adapted from PlainQAFact)")
    print("=" * 65)
    print("  NOTE: this is an ADAPTED evaluation (GPT-4o-mini throughout),")
    print("  not the official PlainQAFact metric - see script docstring.")
    print()

    df = pd.read_csv(INPUT_PATH)
    df = df[df["error"].isna()].copy()
    print(f"  Loaded {len(df)} valid source/summary pairs")

    results = []
    for i, row in df.iterrows():
        print(f"  Scoring pair {i+1}/{len(df)}...")
        result = score_pair(row["source_text"], row["generated_summary"])
        results.append(result)

    results_df = pd.DataFrame(results)
    combined = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    combined.to_csv(OUTPUT_PATH, index=False)

    n_errors = combined["consistency_error"].notna().sum()
    valid = combined[combined["consistency_error"].isna()]

    print(f"\n--- Results ---")
    print(f"  Scored: {len(valid)}/{len(df)} ({n_errors} errors)")
    if len(valid) > 0:
        print(f"  Mean overall consistency score: {valid['overall_score'].mean():.3f}")
        print(f"  Mean simplification-claim score: {valid['simplification_score'].mean():.3f}")
        if valid['elaboration_score'].notna().any():
            print(f"  Mean elaboration-claim score: {valid['elaboration_score'].mean():.3f}")
        else:
            print(f"  No elaboration claims detected across this sample")
    print(f"  Saved: {OUTPUT_PATH}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    data_to_plot = [
        valid["simplification_score"].dropna().values,
        valid["elaboration_score"].dropna().values,
    ]
    bp = ax.boxplot(data_to_plot, labels=["Simplification", "Elaboration"],
                     patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], ["#4a90d9", "#e08214"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, d in enumerate(data_to_plot, start=1):
        x = np.random.normal(i, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.6, color="black", s=25, zorder=3)
    ax.set_ylabel("Factual Consistency Score")
    ax.set_title("Factual Consistency by Claim Type\n(Adapted PlainQAFact Methodology - GPT-4o-mini)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "factual_consistency_by_claim_type.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig saved - factual_consistency_by_claim_type.png")

    print("\n--- Factual Consistency Evaluation complete ---")
    print("  Reminder: adapted evaluation, not official PlainQAFact --")
    print("  see cloud GPU setup plan for the validated metric.")

    return combined


if __name__ == "__main__":
    run_factual_consistency_eval()
