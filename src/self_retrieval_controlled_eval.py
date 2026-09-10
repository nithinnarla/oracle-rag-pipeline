"""
ORACLE, Self-Retrieval Evaluation — Pool-Size-Controlled Comparison
Phase 4, Stage 2 extension, controlled ablation

Complements self_retrieval_eval.py (as-deployed, true band-pool sizes) with
a controlled comparison that fixes the retrieval candidate pool to N records
per query (matching the smallest band, low=4,264), so cross-band differences
in Precision@1/MRR cannot be explained by pool-size alone. Each query's own
true source record is always included in its fixed-size candidate pool, so
retrieval difficulty is held constant while genuine embedding-space quality
differences by band remain visible.

Answers: does low band's higher as-deployed P@1 reflect genuinely better
retrieval quality, or is it fully explained by having fewer candidates?
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
from retrieval_pipeline import get_band_embeddings, encode_query, cosine_similarity_batch
from dpr_encoder import get_dpr_query_encoder

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'oracle_corpus.csv')
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures', 'stage2')
RESULTS_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'self_retrieval_controlled_results.csv')
os.makedirs(FIGURES_DIR, exist_ok=True)

BAND_ORDER = ['low', 'medium', 'high', 'clinical']
BAND_COLORS = {'low': '#2ecc71', 'medium': '#3498db', 'high': '#e67e22', 'clinical': '#e74c3c'}

N_QUERIES_PER_BAND = 300
FIXED_POOL_SIZE = 4264  # matches low band's true size, the smallest band


def run_controlled_eval():
    print("ORACLE, Self-Retrieval Evaluation - Pool-Size-Controlled")
    print("=" * 60)
    print(f"Fixed candidate pool size per query: {FIXED_POOL_SIZE}")

    df = pd.read_csv(CORPUS_PATH)
    df = df[df['literacy_band'].isin(BAND_ORDER)].copy()

    q_tokenizer, q_model = get_dpr_query_encoder()

    all_results = []
    rng = np.random.RandomState(42)

    for band in BAND_ORDER:
        band_df = df[df['literacy_band'] == band].copy()
        embeddings, doc_ids = get_band_embeddings(band)
        doc_ids = np.array([str(d) for d in doc_ids])
        band_size = len(doc_ids)

        eval_df = band_df.sample(n=min(N_QUERIES_PER_BAND, len(band_df)), random_state=42)

        print(f"\n--- Band: {band} (true pool={band_size}, "
              f"fixed eval pool={min(FIXED_POOL_SIZE, band_size)}) ---")
        start = time.time()

        for _, row in eval_df.iterrows():
            true_id = str(row['record_id'])
            question = str(row['question'])
            if len(question.strip()) == 0:
                continue

            true_idx_arr = np.where(doc_ids == true_id)[0]
            if len(true_idx_arr) == 0:
                continue
            true_idx = true_idx_arr[0]

            pool_size = min(FIXED_POOL_SIZE, band_size)
            other_indices = np.array([i for i in range(band_size) if i != true_idx])
            n_others_needed = pool_size - 1
            if len(other_indices) >= n_others_needed:
                chosen_others = rng.choice(other_indices, size=n_others_needed, replace=False)
            else:
                chosen_others = other_indices
            pool_indices = np.concatenate([[true_idx], chosen_others])
            rng.shuffle(pool_indices)

            pool_embeddings = embeddings[pool_indices]
            pool_doc_ids = doc_ids[pool_indices]

            query_vec = encode_query(question, q_tokenizer, q_model)
            scores = cosine_similarity_batch(query_vec, pool_embeddings)
            ranked_order = np.argsort(scores)[::-1]

            rank = None
            for r, idx in enumerate(ranked_order):
                if pool_doc_ids[idx] == true_id:
                    rank = r + 1
                    break

            all_results.append({
                'record_id': true_id,
                'source': row['source'],
                'literacy_band': band,
                'pool_size': pool_size,
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

    print("\n--- Controlled Summary by Band (fixed pool=4,264) ---")
    summary = results_df.groupby('literacy_band').agg(
        n=('record_id', 'count'),
        precision_at_1=('precision_at_1', 'mean'),
        recall_at_5=('recall_at_5', 'mean'),
        recall_at_10=('recall_at_10', 'mean'),
        mrr=('reciprocal_rank', 'mean'),
    ).reindex(BAND_ORDER)
    print(summary.round(4).to_string())

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
    ax.set_title(f'Self-Retrieval, Pool-Size-Controlled (N={FIXED_POOL_SIZE} per query)\n'
                  'Isolates band-quality effects from candidate-pool-size effects')
    ax.legend()
    plt.tight_layout()
    outpath = os.path.join(FIGURES_DIR, 'self_retrieval_controlled_by_band.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {outpath}")

    return results_df


if __name__ == "__main__":
    run_controlled_eval()
