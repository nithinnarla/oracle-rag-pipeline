"""
ORACLE, Framework Architecture Diagram - Section 3.1
Phase 5, schematic for the paper

Section 3.1 describes the four-stage pipeline in prose with no figure. This
draws that description: ingestion with literacy scoring, retrieval restricted
to a band-specific candidate pool by hard band selection, band-conditioned
generation, and readability plus content-fidelity evaluation.

Band pool sizes are read from the corpus rather than hard-coded, so the
diagram cannot drift from Table 2.

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
CORPUS_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'oracle_corpus.csv')
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

DPI = 300
BAND_ORDER = ['low', 'medium', 'high', 'clinical']
BAND_COLORS = {'low': '#8fd9b6', 'medium': '#85c1e9',
               'high': '#f5cba7', 'clinical': '#f1948a'}
BAND_RANGE = {'low': 'FK <= 6', 'medium': '6 < FK <= 10',
              'high': '10 < FK <= 14', 'clinical': 'FK > 14'}

STAGE_FILL = '#f4f6f8'
STAGE_EDGE = '#5d6d7e'


def box(ax, x, y, w, h, text, fill, edge='#34495e', fontsize=8.5, weight='normal', radius=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.004,rounding_size=%s' % radius,
                                linewidth=0.9, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, zorder=3, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, style='-|>', color='#34495e', lw=1.1, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, linewidth=lw, color=color,
                                 connectionstyle='arc3,rad=%s' % rad, zorder=4))


def draw(band_counts, total):
    fig, ax = plt.subplots(figsize=(13.8, 7.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    # stage panels, with their headings placed above rather than inside so
    # nothing can collide with the first box in a column
    stages = [
        (1.5, 22.0, 'Stage 1\nIngestion and literacy scoring'),
        (25.0, 22.0, 'Stage 2\nBand-conditioned retrieval'),
        (48.5, 22.0, 'Stage 3\nBand-conditioned generation'),
        (72.0, 26.5, 'Stage 4\nEvaluation'),
    ]
    for x, w, label in stages:
        ax.add_patch(FancyBboxPatch((x, 2.0), w, 48.0,
                                    boxstyle='round,pad=0.01,rounding_size=0.02',
                                    linewidth=1.0, edgecolor=STAGE_EDGE,
                                    facecolor=STAGE_FILL, alpha=0.6, zorder=1))
        ax.text(x + w / 2, 53.6, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='#2c3e50',
                zorder=3, linespacing=1.6)

    # ---- Stage 1: corpus, FK scoring, band pools ----
    box(ax, 3.5, 43.0, 18.0, 5.0,
        'Retrieval corpus\n%s records, 5 sources' % format(total, ','),
        '#ffffff', weight='bold')
    box(ax, 3.5, 36.0, 18.0, 4.5, 'Flesch-Kincaid grade\nscored per document', '#ffffff')
    arrow(ax, 12.5, 43.0, 12.5, 40.5)
    ax.text(12.5, 33.7, 'assigned to one of four bands', ha='center',
            fontsize=8, style='italic', color='#566573')
    y = 28.0
    for band in BAND_ORDER:
        box(ax, 3.5, y, 18.0, 3.6,
            '%s   %s\n%s records' % (band, BAND_RANGE[band],
                                     format(band_counts[band], ',')),
            BAND_COLORS[band], fontsize=7.8)
        y -= 4.4
    arrow(ax, 12.5, 36.0, 12.5, 31.6)

    # ---- Stage 2: query, estimated band, hard selection, retrieval ----
    box(ax, 26.5, 43.5, 20.0, 4.2, 'User query', '#ffffff', weight='bold')
    box(ax, 26.5, 36.5, 20.0, 4.8, 'Literacy band estimated\nfrom the query', '#ffffff')
    arrow(ax, 36.5, 43.5, 36.5, 41.3)
    box(ax, 26.5, 28.0, 20.0, 5.6,
        'Hard band selection\ncandidate pool restricted\nto that band only',
        '#fdebd0', edge='#b9770e')
    arrow(ax, 36.5, 36.5, 36.5, 33.6)
    box(ax, 26.5, 19.5, 20.0, 5.6,
        'DPR dense retrieval\ndpr-question_encoder\ncosine ranking within pool', '#ffffff')
    arrow(ax, 36.5, 28.0, 36.5, 25.1)
    box(ax, 26.5, 12.5, 20.0, 4.2, 'Top-k retrieved\ndocuments', '#ffffff')
    arrow(ax, 36.5, 19.5, 36.5, 16.7)

    # band pools supply the restricted candidate pool
    arrow(ax, 21.5, 20.0, 26.5, 30.8, rad=-0.24, color='#b9770e')
    ax.text(23.2, 25.0, 'band\npools', fontsize=7.5, style='italic',
            color='#b9770e', ha='center', linespacing=1.4)

    # ---- Stage 3: band prompt, assembly, generation ----
    box(ax, 50.0, 36.5, 20.0, 5.6,
        'Band-specific system\nprompt template\none per literacy band',
        '#d6eaf8', edge='#2471a3')
    box(ax, 50.0, 28.0, 20.0, 5.6,
        'Prompt assembly\nband instruction +\nretrieved documents', '#ffffff')
    arrow(ax, 60.0, 36.5, 60.0, 33.6)
    box(ax, 50.0, 19.5, 20.0, 5.6, 'Generation\ngpt-4o-mini', '#ffffff', weight='bold')
    arrow(ax, 60.0, 28.0, 60.0, 25.1)
    box(ax, 50.0, 12.5, 20.0, 4.2, 'Literacy-adapted\nresponse', '#ffffff')
    arrow(ax, 60.0, 19.5, 60.0, 16.7)
    arrow(ax, 46.5, 14.6, 50.0, 29.5, rad=0.22)

    # ---- Stage 4: evaluation axes ----
    ev = [
        ('Retrieval', 'Precision@1, Recall@K, MRR\nself-retrieval ground truth', '#eaf2f8'),
        ('Readability', 'Flesch-Kincaid, SMOG', '#e8f8f5'),
        ('Content fidelity', 'ROUGE-L, BERTScore\nAPPLS perturbation validation', '#fef9e7'),
        ('Factual consistency', 'adapted GPT-4o-mini method\nand official PlainQAFact', '#fdedec'),
    ]
    y = 40.0
    for title, detail, fill in ev:
        box(ax, 73.5, y, 23.5, 7.0, '%s\n%s' % (title, detail), fill, fontsize=7.8)
        y -= 9.0
    arrow(ax, 70.0, 14.6, 73.5, 23.0, rad=0.2)

    ax.text(85.25, 6.6,
            'Routing correctness is the independent variable: the same\n'
            'retrieved context is generated under the wrong band and\n'
            'under the correct band (Section 5.3)',
            ha='center', va='center', fontsize=7.6, style='italic',
            color='#566573', linespacing=1.5)

    ax.set_title('ORACLE: literacy-conditioned retrieval-augmented generation pipeline',
                 fontsize=12.5, fontweight='bold', pad=16)

    out = os.path.join(FIGURES_DIR, 'oracle_architecture.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  saved %s' % os.path.basename(out))
    return out


def run_figure():
    print('ORACLE - Framework Architecture Diagram')
    print('=' * 52)
    df = pd.read_csv(CORPUS_PATH, usecols=['literacy_band'])
    counts = df.literacy_band.value_counts()
    band_counts = {b: int(counts.get(b, 0)) for b in BAND_ORDER}
    print('  band pool sizes read from corpus: %s' % band_counts)
    draw(band_counts, len(df))
    print('\nSaved to figures/ at %d dpi' % DPI)


if __name__ == '__main__':
    run_figure()
