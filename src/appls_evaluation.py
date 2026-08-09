"""
ORACLE APPLS Option 2 - Metric Sensitivity Evaluation
Tests whether ORACLE's metrics (FK, ROUGE-L, BERTScore) are sensitive
to informativeness (delete_sentence), coherence (coherent), and
simplification (simplification) perturbations on ORACLE's own PLABA test split.

Decision 15: Empirical upgrade of Decision 14 citation-based justification.
3 of 4 APPLS criteria covered empirically. Faithfulness covered by Decision 13
PlainQAFact evaluation.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textstat
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures', 'stage4')
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'appls')

def compute_fk(texts):
    return [textstat.flesch_kincaid_grade(t) for t in texts]

def compute_rouge_l(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return [scorer.score(r, h)['rougeL'].fmeasure for r, h in zip(references, hypotheses)]

def bin_scores(pct, scores, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_means_pct = []
    bin_means_score = []
    for i in range(n_bins):
        mask = [(pct[j] >= bins[i]) and (pct[j] < bins[i+1]) for j in range(len(pct))]
        if any(mask):
            bin_means_pct.append(np.mean([pct[j] for j in range(len(pct)) if mask[j]]))
            bin_means_score.append(np.mean([scores[j] for j in range(len(scores)) if mask[j]]))
    return bin_means_pct, bin_means_score

def run_evaluation():
    results = []

    tasks = [
        ('delete_sentence_plaba_test_perturbation.csv', 'perturbed_sentence_percentage', 'delete_sentence', 'Informativeness'),
        ('coherent_plaba_test_perturbation.csv', 'perturbed_percentage', 'coherent', 'Coherence'),
        ('simplification_plaba_test_perturbation.csv', 'perturbed_chunk_percentage', 'simplification', 'Simplification'),
    ]

    all_data = {}
    for fname, pct_col, task, criterion in tasks:
        print(f"\nProcessing {task} ({criterion})...")
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        refs = df['reference_text'].tolist()
        perturbed = df['perturbed_text'].tolist()
        pct = df[pct_col].tolist()

        print("  Computing FK grade...")
        fk_scores = compute_fk(perturbed)
        fk_corr, fk_p = spearmanr(pct, fk_scores)

        print("  Computing ROUGE-L...")
        rouge_scores = compute_rouge_l(refs, perturbed)
        rouge_corr, rouge_p = spearmanr(pct, rouge_scores)

        print("  Computing BERTScore (sample 100)...")
        sample_idx = list(range(min(100, len(refs))))
        P, R, F1 = bert_score(
            [perturbed[i] for i in sample_idx],
            [refs[i] for i in sample_idx],
            lang='en', verbose=False
        )
        bert_scores = F1.numpy().tolist()
        pct_sample = [pct[i] for i in sample_idx]
        bert_corr, bert_p = spearmanr(pct_sample, bert_scores)

        print(f"  FK:        r={fk_corr:.3f}, p={fk_p:.3f}")
        print(f"  ROUGE-L:   r={rouge_corr:.3f}, p={rouge_p:.3f}")
        print(f"  BERTScore: r={bert_corr:.3f}, p={bert_p:.3f}")

        all_data[task] = {
            'pct': pct, 'fk': fk_scores, 'rouge': rouge_scores,
            'pct_sample': pct_sample, 'bert': bert_scores,
            'criterion': criterion, 'n': len(df)
        }

        results.append({
            'task': task, 'criterion': criterion,
            'fk_corr': fk_corr, 'fk_p': fk_p,
            'rouge_corr': rouge_corr, 'rouge_p': rouge_p,
            'bert_corr': bert_corr, 'bert_p': bert_p,
            'n_rows': len(df)
        })

    task_labels = {
        'delete_sentence': 'Informativeness - Sentence Deletion',
        'coherent': 'Coherence - Sentence Reordering',
        'simplification': 'Simplification - Lexical Substitution',
    }
    task_list = list(task_labels.items())

    # Figure 1: ROUGE-L and BERTScore sensitivity curves (3x2 grid)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('ORACLE APPLS - ROUGE-L and BERTScore Sensitivity to Perturbations', fontsize=13, fontweight='bold')

    metric_pairs = [('rouge', 'ROUGE-L'), ('bert', 'BERTScore')]

    for row, (task, task_label) in enumerate(task_list):
        for col, (metric_key, metric_name) in enumerate(metric_pairs):
            ax = axes[row][col]
            d = all_data[task]
            pct = d['pct_sample'] if metric_key == 'bert' else d['pct']
            scores = d[metric_key]
            bin_x, bin_y = bin_scores(pct, scores)
            ax.scatter(pct, scores, alpha=0.1, s=5, color='steelblue')
            ax.plot(bin_x, bin_y, 'r-', linewidth=2, label='Bin mean')
            r_val = results[[r['task'] for r in results].index(task)][f'{metric_key}_corr']
            ax.set_title(f'{metric_name} - {task_label}\n(Spearman r={r_val:.3f})', fontsize=10)
            ax.set_xlabel('Perturbation %', fontsize=9)
            ax.set_ylabel(metric_name, fontsize=9)
            ax.legend(fontsize=8)

    plt.tight_layout()
    outpath = os.path.join(FIGURES_DIR, 'appls_metric_sensitivity.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nFigure saved: {outpath}')

    # Figure 2: FK Grade sensitivity curves (1x3 grid)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('ORACLE APPLS - FK Grade Sensitivity to Perturbations', fontsize=13, fontweight='bold')

    for col, (task, task_label) in enumerate(task_list):
        ax = axes[col]
        d = all_data[task]
        bin_x, bin_y = bin_scores(d['pct'], d['fk'])
        ax.scatter(d['pct'], d['fk'], alpha=0.1, s=5, color='orange')
        ax.plot(bin_x, bin_y, 'r-', linewidth=2, label='Bin mean')
        r_val = results[[r['task'] for r in results].index(task)]['fk_corr']
        sensitive = 'SENSITIVE' if abs(r_val) > 0.3 else 'not sensitive'
        ax.set_title(f'FK Grade - {task_label}\n(Spearman r={r_val:.3f}, {sensitive})', fontsize=10)
        ax.set_xlabel('Perturbation %', fontsize=9)
        ax.set_ylabel('FK Grade', fontsize=9)
        ax.legend(fontsize=8)

    plt.tight_layout()
    outpath = os.path.join(FIGURES_DIR, 'appls_fk_sensitivity.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure saved: {outpath}')

    df_results = pd.DataFrame(results)
    print('\n=== SUMMARY ===')
    print(df_results[['task','criterion','fk_corr','rouge_corr','bert_corr']].to_string(index=False))
    df_results.to_csv(os.path.join(REPO_ROOT, 'docs', 'appls_oracle_results.csv'), index=False)
    return df_results

if __name__ == '__main__':
    run_evaluation()
