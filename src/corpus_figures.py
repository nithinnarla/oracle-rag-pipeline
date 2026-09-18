"""
ORACLE, Corpus Composition Figures - Section 5.1
Phase 5, corpus-level figures for the paper

Draws the two corpus-level figures Section 5.1 describes. Both previously
had no generating script: corpus_composition.png existed in figures/ with
no source, and the FK distribution slot was filled by a retrieval-eval
figure (eval_fk_distribution.png) that plots retrieved documents from the
20-query set, not corpus records, so it did not match its caption.

Reads oracle_corpus.csv only. No API calls, no model loading, deterministic.

Figures:
1. corpus_composition.png      - records per source, matching Table 1
2. corpus_fk_distribution.png  - FK grade across all records with the four
                                 band boundaries marked, the basis for
                                 literacy-band assignment
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'oracle_corpus.csv')
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

DPI = 300

# display names, ordered largest to smallest so the bar chart reads top-down
SOURCE_LABELS = [
    ('medmcqa', 'MedMCQA', '#1f77b4'),
    ('medqa', 'MedQA', '#159e60'),
    ('mirage', 'MIRAGE', '#d68910'),
    ('pubmedqa', 'PubMedQA', '#8067c4'),
    ('plaba', 'PLABA', '#c0392b'),
]

# band cut points as literacy_classifier.py actually applies them: the intervals
# are half-open on continuous FK, low <= 6, 6 < medium <= 10, 10 < high <= 14,
# clinical > 14. Counts are read off the literacy_band column rather than
# re-derived here, so they agree with Table 2 by construction.
BAND_BOUNDS = [(6, 'low'), (10, 'medium'), (14, 'high')]
BAND_COLORS = {'low': '#8fd9b6', 'medium': '#85c1e9',
               'high': '#f5cba7', 'clinical': '#f1948a'}
BAND_ORDER = ['low', 'medium', 'high', 'clinical']


def plot_corpus_composition(df):
    counts = [(label, int((df.source == key).sum()), color)
              for key, label, color in SOURCE_LABELS]
    total = len(df)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [c[0] for c in counts][::-1]
    values = [c[1] for c in counts][::-1]
    colors = [c[2] for c in counts][::-1]

    bars = ax.barh(labels, values, color=colors)
    for bar, n in zip(bars, values):
        ax.text(bar.get_width() + total * 0.008, bar.get_y() + bar.get_height() / 2,
                '%s (%.1f%%)' % (format(n, ','), 100 * n / total),
                va='center', fontsize=11)

    ax.set_title('ORACLE Retrieval Corpus Composition\n'
                 'Total: %s records across %d sources' % (format(total, ','), len(counts)),
                 fontsize=13)
    ax.set_xlabel('Records', fontsize=11)
    ax.set_xlim(0, max(values) * 1.22)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'corpus_composition.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  saved %s' % os.path.basename(out))
    for label, n, _ in counts:
        print('    %-10s %7s  %5.1f%%' % (label, format(n, ','), 100 * n / total))


def plot_corpus_fk_distribution(df):
    fk = pd.to_numeric(df.fk_grade, errors='coerce')
    band_counts = df.literacy_band.value_counts()
    fk = fk.dropna()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    # clip the display range so a handful of extreme scores do not flatten the bulk
    ax.hist(fk.clip(-2, 30), bins=64, color='#5a7fa8', edgecolor='white', linewidth=0.3)

    prev = ax.get_xlim()[0]
    for bound, band in BAND_BOUNDS:
        ax.axvspan(prev, bound, color=BAND_COLORS[band], alpha=0.22)
        prev = bound
    ax.axvspan(prev, ax.get_xlim()[1], color=BAND_COLORS['clinical'], alpha=0.22)
    for bound, _ in BAND_BOUNDS:
        ax.axvline(bound, color='#444444', linestyle='--', linewidth=1)

    ymax = ax.get_ylim()[1]
    for centre, band in [(2, 'low'), (8.2, 'medium'), (12.2, 'high'), (22, 'clinical')]:
        n = int(band_counts.get(band, 0))
        ax.text(centre, ymax * 0.93, '%s\n%s' % (band, format(n, ',')),
                ha='center', fontsize=10, fontweight='bold')

    ax.set_title('Flesch-Kincaid Grade Distribution Across All Corpus Records\n'
                 '%s records, shaded by literacy band (cut points at FK 6, 10, 14)'
                 % format(len(fk), ','), fontsize=13)
    ax.set_xlabel('Flesch-Kincaid Grade Level', fontsize=11)
    ax.set_ylabel('Records', fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'corpus_fk_distribution.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  saved %s' % os.path.basename(out))
    print('    FK median %.2f, mean %.2f, min %.2f, max %.2f'
          % (fk.median(), fk.mean(), fk.min(), fk.max()))


def run_figures():
    print('ORACLE - Corpus Composition Figures')
    print('=' * 52)
    df = pd.read_csv(CORPUS_PATH, usecols=['source', 'literacy_band', 'fk_grade'])
    print('  corpus: %s records' % format(len(df), ','))
    plot_corpus_composition(df)
    plot_corpus_fk_distribution(df)
    print('\nBoth figures saved to figures/ at %d dpi' % DPI)


if __name__ == '__main__':
    run_figures()
