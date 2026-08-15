"""
ORACLE Cross-Dataset Evaluation - Figure Generation
Phase 4, Stage 4: Cross-dataset evaluation figures

Five figures:
1. cross_dataset_fk_comparison.png - FK source vs generated per source (2x3 grid)
2. cross_dataset_routing_rouge.png - Band routing accuracy + ROUGE-L side by side
3. cross_dataset_fk_by_band.png - FK reduction by literacy band heatmap
4. cross_dataset_routing_impact.png - wrong-routing vs upper-bound, 3 metrics,
   Wilcoxon significance annotated (FK reduction significant, ROUGE-L and
   BERTScore not significant, since correct retrieved context is held fixed
   across both conditions and only the band prompt differs)
5. cross_dataset_misroute_significance.png - same wrong-band vs upper-bound
   comparison as a summary table, computed live rather than hardcoded

Stage 4 figures saved to figures/stage4/
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures', 'stage4')
os.makedirs(FIGURES_DIR, exist_ok=True)
DATA_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'cross_dataset_results.csv')

def run_figures():
    df = pd.read_csv(DATA_PATH)
    actual = df[df['condition'] == 'actual'].copy()

    summary = actual.groupby('source').agg(
        n=('query', 'count'),
        band_accuracy=('band_match', 'mean'),
        mean_fk_source=('fk_source', 'mean'),
        mean_fk_generated=('fk_generated', 'mean'),
        mean_fk_reduction=('fk_reduction', 'mean'),
        mean_rouge_l=('rouge_l', 'mean')
    ).round(3).reset_index()

    sources = summary['source'].tolist()
    bands = ['low', 'medium', 'high', 'clinical']

    # Figure 1: FK source vs generated per source (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('ORACLE Cross-Dataset - FK Grade: Source vs Generated per Dataset', fontsize=13, fontweight='bold')
    axes = axes.flatten()

    for idx, source in enumerate(sources):
        ax = axes[idx]
        src_data = actual[actual['source'] == source]
        fk_src = src_data['fk_source'].mean()
        fk_gen = src_data['fk_generated'].mean()
        fk_red = src_data['fk_reduction'].mean()
        n = len(src_data)
        bars = ax.bar(['Source FK', 'Generated FK'], [fk_src, fk_gen],
                      color=['steelblue', 'coral'], width=0.5)
        ax.set_title(f'{source} (n={n})\nFK change: {fk_red:+.2f}', fontsize=10, fontweight='bold')
        ax.set_ylabel('FK Grade', fontsize=9)
        ax.axhline(y=8, color='green', linestyle='--', linewidth=1, alpha=0.7)
        for bar, val in zip(bars, [fk_src, fk_gen]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_dataset_fk_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: cross_dataset_fk_comparison.png')

    # Figure 2: Band routing accuracy + ROUGE-L side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ORACLE Cross-Dataset - Routing Accuracy and ROUGE-L per Dataset', fontsize=13, fontweight='bold')

    colors = ['green' if v >= 0.7 else 'orange' if v >= 0.4 else 'red'
              for v in summary['band_accuracy']]
    ax1.bar(sources, summary['band_accuracy'], color=colors)
    ax1.set_title('Literacy Band Routing Accuracy', fontsize=11)
    ax1.set_xlabel('Dataset', fontsize=9)
    ax1.set_ylabel('Routing Accuracy', fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax1.tick_params(axis='x', rotation=15)
    for i, v in enumerate(summary['band_accuracy']):
        ax1.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=9)

    ax2.bar(sources, summary['mean_rouge_l'], color='steelblue')
    ax2.set_title('ROUGE-L Score', fontsize=11)
    ax2.set_xlabel('Dataset', fontsize=9)
    ax2.set_ylabel('Mean ROUGE-L', fontsize=9)
    ax2.tick_params(axis='x', rotation=15)
    for i, v in enumerate(summary['mean_rouge_l']):
        ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_dataset_routing_rouge.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: cross_dataset_routing_rouge.png')

    # Figure 3: FK reduction by literacy band heatmap
    pivot = actual.groupby(['source', 'target_band'])['fk_reduction'].mean().unstack(fill_value=np.nan).round(2)
    pivot = pivot.reindex(columns=[b for b in bands if b in pivot.columns])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax,
                linewidths=0.5, mask=pivot.isna(),
                cbar_kws={'label': 'FK Reduction (positive = simplified)'})
    ax.set_title('ORACLE Cross-Dataset - FK Reduction by Dataset and Literacy Band\n(positive = generated text simpler than source)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Literacy Band', fontsize=10)
    ax.set_ylabel('Dataset', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_dataset_fk_by_band.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: cross_dataset_fk_by_band.png')

    # Figure 4: wrong-routing vs upper-bound, 3 metrics, paired Wilcoxon significance
    wrong = actual[actual['band_match'] == False].copy()
    upper = df[df['condition'] == 'upper_bound'].copy()
    wrong_p = wrong.set_index(['source', 'query'])
    upper_p = upper.set_index(['source', 'query'])
    paired = wrong_p.join(upper_p, lsuffix='_wrong', rsuffix='_upper', how='inner')

    metrics = [
        ('fk_reduction', 'FK Reduction', False),
        ('rouge_l', 'ROUGE-L', True),
        ('bertscore', 'BERTScore', True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'ORACLE - Wrong-Band vs Upper-Bound (Correct Band) Generation, Same Retrieved Context (n={len(paired)} paired queries)',
        fontsize=12, fontweight='bold'
    )

    for ax, (col, label, is_bounded) in zip(axes, metrics):
        a = paired[col + '_wrong'].dropna()
        b = paired[col + '_upper'].dropna()
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        stat, p = wilcoxon(a, b)
        sig = 'significant' if p < 0.05 else 'not significant'
        bars = ax.bar(['Wrong band', 'Upper bound\n(correct band)'],
                       [a.mean(), b.mean()], color=['red', 'royalblue'], width=0.5)
        ax.set_title(f'{label}\nWilcoxon p={p:.3f} ({sig})', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Mean {label}', fontsize=9)
        if not is_bounded:
            ax.axhline(y=0, color='black', linewidth=0.8)
        for bar, val in zip(bars, [a.mean(), b.mean()]):
            ypos = val - 0.15 if (val < 0 and not is_bounded) else val + (0.005 if is_bounded else 0.05)
            ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.3f}',
                    ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_dataset_routing_impact.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: cross_dataset_routing_impact.png')

    print('\nAll 4 figures saved to figures/stage4/')


def plot_misroute_significance():
    """
    Wrong-band vs upper-bound (correct-band) generation, paired Wilcoxon
    test, computed live from the current cross_dataset_results.csv.
    Replaces the earlier plot_significance_progression(), which reported
    hardcoded p=0.032 (n=38) -> p=0.0094 (n=71) as a "strengthening"
    trend; that snapshot pair was not reproducible from this file, and
    at the current n the FK-reduction result has moved back toward the
    threshold (p~0.049) rather than continuing to strengthen. Reporting
    the real current numbers instead of a stale, unreproducible pair.
    """
    df = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'processed', 'cross_dataset_results.csv'))
    actual = df[df['condition'] == 'actual']
    wrong = actual[actual['band_match'] == False].copy()
    upper = df[df['condition'] == 'upper_bound'].copy()
    wrong_p = wrong.set_index(['source', 'query'])
    upper_p = upper.set_index(['source', 'query'])
    paired = wrong_p.join(upper_p, lsuffix='_wrong', rsuffix='_upper', how='inner')

    metrics = [
        ('fk_reduction', 'FK Reduction'),
        ('rouge_l', 'ROUGE-L'),
        ('bertscore', 'BERTScore'),
    ]

    results = []
    for col, label in metrics:
        a = paired[col + '_wrong'].dropna()
        b = paired[col + '_upper'].dropna()
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        stat, p = wilcoxon(a, b)
        results.append({
            'metric': label, 'n': len(a),
            'wrong_mean': a.mean(), 'upper_mean': b.mean(), 'p': p
        })
        sig = 'significant' if p < 0.05 else 'not significant'
        print(f"{label}: n={len(a)}, wrong_mean={a.mean():.4f}, "
              f"upper_mean={b.mean():.4f}, p={p:.5f} ({sig})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    table_data = [[r['metric'], r['n'], f"{r['wrong_mean']:.4f}",
                   f"{r['upper_mean']:.4f}", f"{r['p']:.4f}",
                   'sig.' if r['p'] < 0.05 else 'n.s.'] for r in results]
    col_labels = ['Metric', 'n', 'Wrong band', 'Correct band', 'p (Wilcoxon)', '']
    tbl = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2)
    ax.set_title('Misrouted vs Correct-Band Generation\n(current data, n=%d paired queries)' % len(paired),
                 fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'cross_dataset_misroute_significance.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')
    return results


if __name__ == '__main__':
    run_figures()
    plot_misroute_significance()
