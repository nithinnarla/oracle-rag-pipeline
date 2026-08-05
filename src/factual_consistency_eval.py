"""
ORACLE -- Stage 4: Factual Consistency Evaluation (Adapted from PlainQAFact)
Phase 4 -- Stage 4: Evaluation metrics (Decision 7: PlainQAFact + APPLS)

IMPORTANT -- READ BEFORE CITING: This is an ADAPTED evaluation, not the
official PlainQAFact metric (You & Guo, 2026, arXiv 2503.08890, JBI).
The official pipeline requires Llama 3.1 8B Instruct locally (40GB+ GPU
memory) and a separate LERC scoring model -- infeasible on this machine
(Apple Silicon, no CUDA). Installing the official `plainqafact` pip package
was also rejected: its dependency tree (torch 2.13.0, transformers 4.44.2,
pyserini, faiss-cpu, nmslib, spacy) conflicts with versions already verified
working elsewhere in this project (torch 2.10.0, transformers 4.57.6, used
by mbert_classifier.py in the sibling HyDMIS repo) and was never intended to
share an environment with an existing project per the official repo's own
isolated-conda-env setup instructions.

This script borrows PlainQAFact's real conceptual approach -- classify
sentence type, extract a claim, verify it against the source, score
consistency -- but substitutes GPT-4o-mini for every model in the original
pipeline (their fine-tuned classifier, Llama 3.1 8B for answer extraction,
BART for question generation, and their QA/LERC scoring models). Results
from this script are NOT directly comparable to PlainQAFact's published
benchmarks and must be labeled as an adapted evaluation, not the official
metric, wherever reported.

PLAN: the official PlainQAFact metric will be run separately, in an isolated
cloud GPU environment (see docs/plainqafact_cloud_setup.md), on the same
source_text/generated_summary pairs this script evaluates. The two scores
will be compared as an internal validation check, not treated as identical.

Pipeline/infrastructure script -- no notebook (single quantitative
evaluation, matches methodology_decisions.md documentation pattern).
"""

import os
import json
import warnings
import pandas as pd
from openai import OpenAI

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "lay_summarizer_results.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "factual_consistency_results.csv")
MODEL = "gpt-4o-mini"

client = OpenAI()

CONSISTENCY_PROMPT = """You are evaluating factual consistency between a scientific abstract (source) and a simplified plain-language summary (generated), adapting the PlainQAFact methodology (You & Guo, 2026).

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
    print("ORACLE -- Factual Consistency Evaluation (Adapted from PlainQAFact)")
    print("=" * 65)
    print("  NOTE: this is an ADAPTED evaluation (GPT-4o-mini throughout),")
    print("  not the official PlainQAFact metric -- see script docstring.")
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

    print("\n--- Factual Consistency Evaluation complete ---")
    print("  Reminder: adapted evaluation, not official PlainQAFact --")
    print("  see cloud GPU setup plan for the validated metric.")

    return combined


if __name__ == "__main__":
    run_factual_consistency_eval()
