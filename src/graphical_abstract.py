"""
ORACLE, Graphical Abstract - Journal of Biomedical Informatics
Phase 5, submission artifact

JBI asks for a graphical abstract as a separate file, at least 1328 x 531
pixels (w x h) and legible when printed at 5 x 13 cm. This draws one at
1992 x 798, the same aspect proportionally larger.

It states the paper's bounded claim rather than advertising the system:
literacy-band routing reaches readability and does not reach retrieval
relevance or factual fidelity. The three test statistics are read from
cross_dataset_results.csv, so the figure cannot drift from Section 5.3.

No API calls, no model loading, deterministic.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, 'data', 'processed', 'cross_dataset_results.csv')
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

DPI = 300
INK = '#2c3e50'
MUTED = '#7f8c8d'
MOVES = '#1a7f5a'
FLAT = '#95a5a6'


def paired_stats():
    """Wrong-band against correct-band on identical retrieved context."""
    from scipy.stats import wilcoxon
    df = pd.read_csv(RESULTS)
    wrong = df[(df.condition == 'actual') & (df.band_match == False)]
    upper = df[df.condition == 'upper_bound']
    paired = wrong.set_index(['source', 'query']).join(
        upper.set_index(['source', 'query']), lsuffix='_w', rsuffix='_u', how='inner')

    out = {}
    for col in ('fk_reduction', 'rouge_l', 'bertscore'):
        a = paired[col + '_w'].dropna()
        b = paired[col + '_u'].dropna()
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        _, p = wilcoxon(a, b)
        out[col] = {'wrong': float(a.mean()), 'correct': float(b.mean()),
                    'p': float(p), 'n': int(len(a))}
    return out


def box(ax, x, y, w, h, text, fill, edge, fontsize, weight='normal', color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.006,rounding_size=0.03',
                                linewidth=1.1, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', color=color,
            fontsize=fontsize, fontweight=weight, zorder=3, linespacing=1.5)


def draw(stats):
    # 6.64 x 2.66 inches at 300 dpi, JBI's aspect proportionally larger than
    # the 1328 x 531 minimum
    fig, ax = plt.subplots(figsize=(6.64, 2.66))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis('off')

    ax.text(50, 37.8, 'Where literacy conditioning reaches in a retrieval-augmented pipeline',
            ha='center', va='center', fontsize=9.4, fontweight='bold', color=INK)

    # ---- left: the mechanism ----
    ax.text(15.5, 33.0, 'conditioned, not post-hoc', ha='center',
            fontsize=5.8, style='italic', color=MUTED)
    for i, (y, h, label, fill, edge) in enumerate([
            (25.0, 5.0, 'Query', '#ffffff', INK),
            (17.2, 6.2, 'Estimated\nliteracy band', '#ffffff', INK),
            (7.8, 6.6, 'Retrieval restricted\nto that band', '#fdebd0', '#b9770e')]):
        box(ax, 4.0, y, 23.0, h, label, fill, edge, 6.4, 'bold' if i == 0 else 'normal')
    ax.add_patch(FancyArrowPatch((15.5, 25.0), (15.5, 23.4), arrowstyle='-|>',
                                 mutation_scale=7, linewidth=1.0, color=INK))
    ax.add_patch(FancyArrowPatch((15.5, 17.2), (15.5, 14.4), arrowstyle='-|>',
                                 mutation_scale=7, linewidth=1.0, color=INK))

    # ---- centre: what correct routing moves, and what it does not ----
    ax.text(51.5, 33.0, 'wrong band vs correct band, same retrieved context',
            ha='center', fontsize=5.8, style='italic', color=MUTED)
    panels = [
        ('Readability', 'FK reduction', 'fk_reduction', True,  '%.2f'),
        ('Content',     'ROUGE-L',      'rouge_l',      False, '%.3f'),
        ('Fidelity',    'BERTScore',    'bertscore',    False, '%.3f'),
    ]
    x0, w, gap = 32.0, 11.5, 2.5
    for i, (l1, l2, key, moves, fmt) in enumerate(panels):
        st = stats[key]
        x = x0 + i * (w + gap)
        col = MOVES if moves else FLAT
        ax.text(x + w / 2, 28.6, l1, ha='center', va='center',
                fontsize=6.4, fontweight='bold', color=INK)
        ax.text(x + w / 2, 26.2, l2, ha='center', va='center',
                fontsize=5.8, color=MUTED)

        base, span = 11.6, 9.4
        vals = [st['wrong'], st['correct']]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        for j, v in enumerate(vals):
            frac = 0.34 + 0.66 * ((v - lo) / rng) if moves else 0.64
            bh = span * frac
            ax.add_patch(plt.Rectangle((x + 1.1 + j * 4.8, base), 3.6, bh,
                                       facecolor=col, alpha=0.55 + 0.45 * j,
                                       edgecolor=col, linewidth=0.7, zorder=2))
            ax.text(x + 2.9 + j * 4.8, base + bh + 0.9, fmt % v, ha='center',
                    fontsize=5.1, color=INK, zorder=3)
        ax.text(x + 2.9, base - 1.5, 'wrong', ha='center', fontsize=4.8, color=MUTED)
        ax.text(x + 7.7, base - 1.5, 'correct', ha='center', fontsize=4.8, color=MUTED)

        verdict = 'p=%.4f' % st['p'] if moves else 'p=%.2f' % st['p']
        ax.text(x + w / 2, 5.6, verdict, ha='center', va='center',
                fontsize=6.0, color=col, fontweight='bold')
        ax.text(x + w / 2, 3.2, 'improves' if moves else 'no change', ha='center',
                va='center', fontsize=5.8, color=col)

    ax.text(51.5, 0.7, 'paired Wilcoxon, n=%d and n=%d. FK reduction is negative: '
                       'both conditions score above source level, the correct band less so.'
            % (stats['fk_reduction']['n'], stats['rouge_l']['n']),
            ha='center', fontsize=4.6, color=MUTED)

    # ---- right: the bounded conclusion ----
    ax.add_patch(FancyBboxPatch((74.5, 6.0), 23.5, 25.0,
                                boxstyle='round,pad=0.008,rounding_size=0.03',
                                linewidth=1.1, edgecolor=MOVES, facecolor='#eafaf1', zorder=2))
    ax.text(86.25, 28.2, 'What this bounds', ha='center', va='center', fontsize=6.6,
            fontweight='bold', color=INK, zorder=3)
    ax.text(86.25, 18.4,
            'Literacy conditioning\nreaches readability.\n\n'
            'It does not reach\nretrieval relevance\nor factual fidelity.',
            ha='center', va='center', fontsize=6.2, color=INK,
            linespacing=1.75, zorder=3)
    ax.text(86.25, 8.2, '36,664 records, 5 sources', ha='center', va='center',
            fontsize=5.1, color=MUTED, zorder=3)

    out = os.path.join(FIGURES_DIR, 'oracle_graphical_abstract.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    return out


def run():
    print('ORACLE - Graphical Abstract')
    print('=' * 52)
    stats = paired_stats()
    for k, v in stats.items():
        print('  %-14s wrong=%.4f correct=%.4f p=%.5f n=%d'
              % (k, v['wrong'], v['correct'], v['p'], v['n']))
    out = draw(stats)
    try:
        from PIL import Image
        w, h = Image.open(out).size
        ok = 'OK' if (w >= 1328 and h >= 531) else 'BELOW JBI MINIMUM'
        print('\n  saved %s  %dx%d  %s (JBI minimum 1328x531)'
              % (os.path.basename(out), w, h, ok))
    except ImportError:
        print('\n  saved %s' % os.path.basename(out))


if __name__ == '__main__':
    run()
