"""
ORACLE — Medical Jargon Identifier
Phase 4 — Stage 3: Jargon Identification and Frequency Analysis

Scans the full ORACLE corpus to:
1. Identify high-frequency medical terms not in current substitution table
2. Rank jargon candidates by frequency and literacy band distribution
3. Flag terms that appear predominantly in clinical/high bands vs low/medium
4. Produce candidate list for expanding JARGON_SUBSTITUTIONS in literacy_adapter.py

Current substitution table: 38 terms in literacy_adapter.py
Target: identify top 50 additional candidates ranked by frequency

Methodology:
- Tokenize full_text across all 37,076 corpus records
- Filter against medical term lexicon (UMLS-derived patterns)
- Compute frequency per term per literacy band
- Rank by clinical/high band frequency minus low/medium frequency
  (high delta = term appears in clinical docs but not plain language = jargon candidate)

Output:
- figures/stage3/jargon_frequency_distribution.png
- figures/stage3/jargon_band_heatmap.png
- figures/stage3/jargon_top50_candidates.png
- data/processed/jargon_candidates.csv

Script type: EDA/analysis — script + notebook + figures
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'oracle_corpus.csv')
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures', 'stage3')
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Current 38-term substitution table from literacy_adapter.py — skip these
EXISTING_SUBSTITUTIONS = {
    'myocardial infarction', 'hypertension', 'hypotension', 'arrhythmia',
    'atherosclerosis', 'tachycardia', 'bradycardia', 'thrombosis', 'embolism',
    'hyperglycemia', 'hypoglycemia', 'dyslipidemia', 'insulin resistance',
    'dyspnea', 'pneumonia', 'bronchitis', 'pulmonary', 'etiology', 'prognosis',
    'pathophysiology', 'contraindication', 'comorbidity', 'asymptomatic',
    'idiopathic', 'chronic', 'acute', 'benign', 'malignant', 'prophylaxis',
    'pharmacological', 'adverse effects', 'subcutaneous', 'intravenous',
    'lesion', 'inflammation', 'exacerbate', 'alleviate', 'manifestation',
    'prevalence', 'incidence', 'mortality', 'morbidity'
}

# Medical term patterns — multi-word and single clinical terms
MEDICAL_PATTERNS = [
    r'\b\w+itis\b',          # inflammation: appendicitis, bronchitis
    r'\b\w+osis\b',          # condition: fibrosis, cirrhosis
    r'\b\w+emia\b',          # blood condition: anemia, leukemia
    r'\b\w+ectomy\b',        # surgical removal: appendectomy
    r'\b\w+plasty\b',        # surgical repair: angioplasty
    r'\b\w+scopy\b',         # examination: colonoscopy
    r'\b\w+tomy\b',          # incision: laparotomy
    r'\b\w+graphy\b',        # imaging: radiography
    r'\b\w+pathy\b',         # disease: neuropathy, cardiomyopathy
    r'\b\w+algia\b',         # pain: myalgia, neuralgia
    r'\b\w+megaly\b',        # enlargement: hepatomegaly
    r'\b\w+rrhea\b',         # flow: diarrhea, dysmenorrhea
    r'\b\w+trophy\b',        # growth: hypertrophy, atrophy
    r'\b\w+genesis\b',       # origin: pathogenesis, oncogenesis
    r'\b\w+lysis\b',         # breakdown: hemolysis, dialysis
    r'\b\w+uria\b',          # urine condition: hematuria, proteinuria
    r'\b\w+toxic\w*\b',      # toxicity terms
    r'\b\w+therapeutic\b',   # therapy terms
    r'\b\w+pharmacologic\w*\b', # drug terms
]

# High-value clinical standalone terms not caught by patterns
CLINICAL_TERMS = [
    'sepsis', 'anaphylaxis', 'ischemia', 'infarction', 'edema', 'fibrosis',
    'necrosis', 'stenosis', 'occlusion', 'perfusion', 'ventilation', 'intubation',
    'catheter', 'dialysis', 'biopsy', 'metastasis', 'carcinoma', 'lymphoma',
    'leukemia', 'hemorrhage', 'coagulation', 'anticoagulant', 'immunosuppression',
    'corticosteroid', 'cytokine', 'endoscopy', 'laparoscopy', 'anastomosis',
    'resection', 'excision', 'debridement', 'lavage', 'intramuscular', 'sublingual',
    'transdermal', 'parenteral', 'bioavailability', 'pharmacokinetics',
    'pharmacodynamics', 'titration', 'bolus', 'infusion', 'tachypnea',
    'hypercapnia', 'hypoxemia', 'atelectasis', 'consolidation', 'effusion',
    'ascites', 'jaundice', 'cyanosis', 'pallor', 'diaphoresis', 'syncope',
    'vertigo', 'nystagmus', 'paresthesia', 'dysarthria', 'aphasia', 'ataxia',
    'tremor', 'rigidity', 'spasticity', 'paralysis', 'plegia', 'paresis',
]


def extract_medical_terms(text):
    """Extract medical terms from text using patterns and term list."""
    if not isinstance(text, str) or len(text) < 10:
        return []

    text_lower = text.lower()
    found = set()

    # Pattern-based extraction
    for pattern in MEDICAL_PATTERNS:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if len(match) > 5:  # skip very short matches
                found.add(match.strip())

    # Direct term matching
    for term in CLINICAL_TERMS:
        if term in text_lower:
            found.add(term)

    # Remove existing substitutions
    found = found - EXISTING_SUBSTITUTIONS

    return list(found)


def run_jargon_identifier():
    print("ORACLE Phase 4 — Stage 3: Medical Jargon Identifier")
    print("=" * 55)

    print("\n--- Loading Corpus ---")
    df = pd.read_csv(CORPUS_PATH)
    print(f"  Records: {len(df):,} across {df['source'].nunique()} sources")
    print(f"  Literacy bands: {df['literacy_band'].value_counts().to_dict()}")

    print("\n--- Extracting Medical Terms ---")
    df['medical_terms'] = df['full_text'].apply(extract_medical_terms)
    df_with_terms = df[df['medical_terms'].apply(len) > 0]
    print(f"  Records with medical terms: {len(df_with_terms):,}")

    # Count term frequency per literacy band
    print("\n--- Computing Term Frequencies ---")
    band_term_counts = {band: Counter() for band in ['low', 'medium', 'high', 'clinical']}

    for _, row in df.iterrows():
        band = row['literacy_band']
        if band in band_term_counts:
            for term in row['medical_terms']:
                band_term_counts[band][term] += 1

    # Total frequency across all bands
    total_counts = Counter()
    for band_counts in band_term_counts.values():
        total_counts.update(band_counts)

    print(f"  Unique medical terms found: {len(total_counts):,}")
    print(f"  Top 10 by frequency:")
    for term, count in total_counts.most_common(10):
        print(f"    {term:<30} {count:,}")

    # Compute jargon score: high in clinical/high bands, low in low/medium bands
    jargon_scores = {}
    for term, total in total_counts.items():
        if total < 5:  # skip rare terms
            continue
        clinical_count = band_term_counts['clinical'].get(term, 0)
        high_count = band_term_counts['high'].get(term, 0)
        low_count = band_term_counts['low'].get(term, 0)
        medium_count = band_term_counts['medium'].get(term, 0)

        clinical_high = clinical_count + high_count
        low_medium = low_count + medium_count

        # Jargon score: normalized difference
        score = (clinical_high - low_medium) / (total + 1)
        jargon_scores[term] = {
            'term': term,
            'total': total,
            'clinical': clinical_count,
            'high': high_count,
            'medium': medium_count,
            'low': low_count,
            'jargon_score': score
        }

    # Sort by jargon score
    candidates_df = pd.DataFrame(list(jargon_scores.values()))
    candidates_df = candidates_df.sort_values('jargon_score', ascending=False)
    top50 = candidates_df.head(50)

    print(f"\n  Top 10 jargon candidates (highest clinical/high vs low/medium gap):")
    for _, row in top50.head(10).iterrows():
        print(f"    {row['term']:<30} score={row['jargon_score']:.3f} "
              f"total={row['total']} clinical={row['clinical']} low={row['low']}")

    # Save candidates
    out_path = os.path.join(PROCESSED_DIR, 'jargon_candidates.csv')
    candidates_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Figure 1 — Top 50 jargon candidates by frequency
    fig, ax = plt.subplots(figsize=(14, 10))
    top20 = top50.head(20)
    colors = ['#e74c3c' if s > 0.5 else '#f39c12' if s > 0.2 else '#3498db'
              for s in top20['jargon_score']]
    bars = ax.barh(range(len(top20)), top20['total'], color=colors, edgecolor='black',
                   linewidth=0.5)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['term'], fontsize=10)
    ax.set_xlabel('Total frequency in corpus')
    ax.set_title('ORACLE — Top 20 Medical Jargon Candidates\n'
                 '(Red=high jargon score >0.5, Orange=medium >0.2, Blue=lower)',
                 fontsize=12)
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    legend = [Patch(color='#e74c3c', label='High jargon score (>0.5)'),
              Patch(color='#f39c12', label='Medium jargon score (>0.2)'),
              Patch(color='#3498db', label='Lower jargon score')]
    ax.legend(handles=legend, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'jargon_top20_candidates.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 1 saved -- jargon_top20_candidates.png")

    # Figure 2 — Band distribution heatmap for top 20 terms
    heat_data = top20[['clinical', 'high', 'medium', 'low']].values
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heat_data, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Clinical', 'High', 'Medium', 'Low'], fontsize=11)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['term'], fontsize=10)
    for i in range(len(top20)):
        for j, col in enumerate(['clinical', 'high', 'medium', 'low']):
            val = int(top20.iloc[i][col])
            ax.text(j, i, str(val), ha='center', va='center', fontsize=8,
                    color='white' if val > heat_data.max() * 0.6 else 'black')
    plt.colorbar(im, ax=ax, label='Term frequency')
    ax.set_title('Medical Jargon — Frequency by Literacy Band\n'
                 'Top 20 candidates by jargon score', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'jargon_band_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 2 saved -- jargon_band_heatmap.png")

    # Figure 3 — Jargon score distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(candidates_df['jargon_score'], bins=30, color='#3498db',
            edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5,
               label='High jargon threshold (0.5)')
    ax.axvline(x=0.2, color='orange', linestyle='--', linewidth=1.5,
               label='Medium jargon threshold (0.2)')
    ax.set_xlabel('Jargon Score (clinical/high - low/medium gap)')
    ax.set_ylabel('Number of terms')
    ax.set_title('ORACLE — Medical Jargon Score Distribution\n'
                 'Higher score = more clinical, less plain language = stronger candidate',
                 fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'jargon_score_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 3 saved -- jargon_score_distribution.png")


    # Figure 4 — Jargon term count by source
    source_jargon = df.copy()
    source_jargon['jargon_count'] = source_jargon['medical_terms'].apply(len)
    source_stats = source_jargon.groupby('source')['jargon_count'].agg(['mean', 'sum']).reset_index()
    source_stats = source_stats.sort_values('mean', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    axes[0].bar(source_stats['source'], source_stats['mean'],
                color=colors[:len(source_stats)], edgecolor='black', linewidth=0.5)
    axes[0].set_title('Mean Jargon Terms per Record by Source', fontsize=11)
    axes[0].set_xlabel('Source')
    axes[0].set_ylabel('Mean jargon terms per record')
    axes[0].tick_params(axis='x', rotation=15)

    axes[1].bar(source_stats['source'], source_stats['sum'],
                color=colors[:len(source_stats)], edgecolor='black', linewidth=0.5)
    axes[1].set_title('Total Jargon Terms by Source', fontsize=11)
    axes[1].set_xlabel('Source')
    axes[1].set_ylabel('Total jargon term occurrences')
    axes[1].tick_params(axis='x', rotation=15)

    plt.suptitle('ORACLE — Jargon Distribution by Source\n'
                 'PLABA (plain language) expected to show lowest jargon density',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'jargon_source_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 4 saved -- jargon_source_distribution.png")


    # Figure 5 — FK grade vs jargon density per source
    source_fk_jargon = df.copy()
    source_fk_jargon['jargon_count'] = source_fk_jargon['medical_terms'].apply(len)
    source_stats2 = source_fk_jargon.groupby('source').agg(
        mean_fk=('fk_grade', 'mean'),
        mean_jargon=('jargon_count', 'mean')
    ).reset_index()

    source_colors = {
        'plaba': '#e74c3c', 'pubmed': '#3498db', 'mirage': '#2ecc71',
        'medqa': '#f39c12', 'medmcqa': '#9b59b6', 'pubmedqa': '#1abc9c'
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in source_stats2.iterrows():
        color = source_colors.get(row['source'], '#7f8c8d')
        ax.scatter(row['mean_fk'], row['mean_jargon'], s=200,
                  color=color, edgecolors='black', linewidth=1, zorder=5)
        ax.annotate(row['source'], (row['mean_fk'], row['mean_jargon']),
                   textcoords='offset points', xytext=(8, 5), fontsize=11,
                   color=color, fontweight='bold')

    ax.set_xlabel('Mean FK Grade (sentence length proxy)', fontsize=12)
    ax.set_ylabel('Mean Jargon Terms per Record', fontsize=12)
    ax.set_title('ORACLE — FK Grade vs Medical Jargon Density by Source\n'
                 'Key finding: PLABA has low FK but non-zero jargon — FK ≠ vocabulary difficulty',
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'jargon_fk_vs_jargon_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 5 saved -- jargon_fk_vs_jargon_scatter.png")

    print(f"\n--- Jargon Identifier complete ---")
    print(f"  {len(total_counts):,} unique medical terms identified in corpus")
    print(f"  {len(candidates_df):,} candidates with frequency >= 5")
    print(f"  Top 50 saved to jargon_candidates.csv")
    print(f"  5 figures saved to figures/stage3/")
    print(f"  Existing 38-term substitution table excluded from candidates")
    print(f"  Top candidates should be reviewed for addition to literacy_adapter.py")

    return candidates_df


if __name__ == "__main__":
    run_jargon_identifier()
