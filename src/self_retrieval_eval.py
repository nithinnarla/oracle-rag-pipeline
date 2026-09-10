"""
ORACLE, Self-Retrieval Evaluation
Phase 4, Stage 2 extension: Precision@1 / Recall@K / MRR via self-supervised ground truth

Each record's own question is used as a query; the record's own record_id is
the ground-truth relevant document, since it is definitionally the source the
question was drawn from. Retrieval is run within each record's own true
literacy band (band_override), isolating retrieval quality from routing
accuracy, which is measured separately (Stage 2 retrieval_evaluation.py).

This is a self-retrieval task: it measures whether the embedding space can
recover a record's own source given its own question, the easiest possible
retrieval task by construction. High scores confirm basic embedding-space
integrity; they are not a claim about hard, realistic-query retrieval quality.

Input: oracle_corpus.csv, retrieval_pipeline.retrieve()
Output: results CSV to data/processed/self_retrieval_results.csv,
        1 figure to figures/stage2/self_retrieval_precision_by_band.png
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from retrieval_pipeline import retrieve
from dpr_encoder import get_dpr_query_encoder

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'oracle_corpus.csv')
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures', 'stage2')
RESULTS_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'self_retrieval_results.csv')
os.makedirs(FIGURES_DIR, exist_ok=True)

BAND_ORDER = ['low', 'medium', 'high', 'clinical']
BAND_COLORS = {'low': '#2ecc71', 'medium': '#3498db', 'high': '#e67e22', 'clinical': '#e74c3c'}

# Set to None to run the full corpus (slower); set an int to stratified-sample
# per band first for a fast correctness check before scaling up.
N_SAMPLE_PER_BAND = 300


def run_self_retrieval_eval():
    print("ORACLE, Self-Retrieval Evaluation")
    print("=" * 50)

    df = pd.read_csv(CORPUS_PATH)
    df = df[df['literacy_band'].isin(BAND_ORDER)].copy()

    print(f"\nLoading DPR query encoder once (reused across all queries)...")
    q_tokenizer, q_model = get_dpr_query_encoder()

    all_results = []

    for band in BAND_ORDER:
        band_df = df[df['literacy_band'] == band].copy()
        band_size = len(band_df)

        if N_SAMPLE_PER_BAND is not None and band_size > N_SAMPLE_PER_BAND:
            eval_df = band_df.sample(n=N_SAMPLE_PER_BAND, random_state=42)
        else:
            eval_df = band_df

        print(f"\n--- Band: {band} ({len(eval_df)}/{band_size} records evaluated, "
              f"top_k={band_size}) ---")
        start = time.time()

        for _, row in eval_df.iterrows():
            true_id = str(row['record_id'])
            question = str(row['question'])
            if len(question.strip()) == 0:
                continue

            result = retrieve(
                question,
                top_k=band_size,
                band_override=band,
                q_tokenizer=q_tokenizer,
                q_model=q_model,
            )

            rank = None
            for doc in result['retrieved']:
                if doc['record_id'] == true_id:
                    rank = doc['rank']
                    break

            all_results.append({
                'record_id': true_id,
                'source': row['source'],
                'literacy_band': band,
                'rank': rank,
                'precision_at_1': 1 if rank == 1 else 0,
                'recall_at_5': 1 if rank is not None and rank <= 5 else 0,
                'recall_at_10': 1 if rank is not None and rank <= 10 else 0,
                'reciprocal_rank': (1.0 / rank) if rank is not None else 0.0,
            })

        elapsed = time.time() - start
        print(f"    {len(eval_df)} queries evaluated in {elapsed:.1f}s "
              f"({elapsed/max(len(eval_df),1):.3f}s/query)")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved: {RESULTS_PATH}")

    print("\n--- Summary by Band ---")
    summary = results_df.groupby('literacy_band').agg(
        n=('record_id', 'count'),
        precision_at_1=('precision_at_1', 'mean'),
        recall_at_5=('recall_at_5', 'mean'),
        recall_at_10=('recall_at_10', 'mean'),
        mrr=('reciprocal_rank', 'mean'),
    ).reindex(BAND_ORDER)
    print(summary.round(4).to_string())

    overall = results_df.agg(
        precision_at_1=('precision_at_1', 'mean'),
        recall_at_5=('recall_at_5', 'mean'),
        recall_at_10=('recall_at_10', 'mean'),
        mrr=('reciprocal_rank', 'mean'),
    )
    print(f"\n--- Overall (n={len(results_df)}) ---")
    print(overall.round(4).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(BAND_ORDER))
    w = 0.25
    ax.bar(x - w, summary['precision_at_1'], w, label='Precision@1',
           color=[BAND_COLORS[b] for b in BAND_ORDER], alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.bar(x, summary['recall_at_10'], w, label='Recall@10',
           color=[BAND_COLORS[b] for b in BAND_ORDER], alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.bar(x + w, summary['mrr'], w, label='MRR',
           color=[BAND_COLORS[b] for b in BAND_ORDER], alpha=0.3, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_ORDER)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Self-Retrieval Evaluation by Literacy Band\n'
                  '(within-band retrieval, own-question-to-own-source ground truth)')
    ax.legend()
    plt.tight_layout()
    outpath = os.path.join(FIGURES_DIR, 'self_retrieval_precision_by_band.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {outpath}")

    return results_df


if __name__ == "__main__":
    run_self_retrieval_eval()
