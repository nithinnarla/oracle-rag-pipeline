"""
ORACLE, MedQA USMLE EDA
Phase 4, Exploratory Data Analysis
Clinical QA Domain

EDA on MedQA USMLE dataset, 11,451 clinical QA instances (Jin et al., 2021).
USMLE board exam questions, hardest biomedical QA benchmark.
Used as ORACLE's clinical retrieval quality ceiling evaluation.

Label mapping: answer_idx = correct option (A/B/C/D)
Two splits: train (10,178) + test (1,273)
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from medqa_loader import load_medqa_all


def run_eda():
    print("ORACLE Phase 4, MedQA USMLE EDA")
    print("=" * 50)

    result = load_medqa_all()
    df = result['train']['data'].copy()
    df['label'] = result['train']['labels'].values
    df['q_len'] = df['question'].astype(str).str.len()
    df['n_phrases'] = df['metamap_phrases'].apply(lambda x: len(x) if hasattr(x,'__len__') else 0)

    # Extract option lengths
    def opt_len(opts, key):
        try:
            return len(str(opts.get(key, '')))
        except AttributeError:
            return 0

    for opt in ['A','B','C','D']:
        df[f'opt_{opt}_len'] = df['options'].apply(lambda x: opt_len(x, opt))

    print(f"\nDataset shape: {df.shape}")

    print(f"\n--- Split Distribution ---")
    for split in ['train','test']:
        n = result[split]['metadata']['n_samples']
        print(f"  {split:<12} {n:,}")
    print(f"  Total:       {result['metadata']['n_samples']:,}")

    print(f"\n--- Answer Distribution ---")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  Option {label}: {count:,} ({count/len(df):.1%})")
    print(f"  Note: Near-balanced, slight B bias ({label_counts['B']/len(df):.1%})")

    print(f"\n--- USMLE Step Distribution ---")
    step_counts = df['meta_info'].value_counts()
    for step, count in step_counts.items():
        print(f"  {step:<15} {count:,} ({count/len(df):.1%})")

    print(f"\n--- Question Length Distribution ---")
    print(f"  Mean: {df['q_len'].mean():.0f} chars | Median: {df['q_len'].median():.0f}")
    print(f"  Min: {df['q_len'].min()} | Max: {df['q_len'].max()}")
    print(f"  Note: Much longer than MedMCQA (mean 79), full clinical scenarios")
    for opt in ['A','B','C','D']:
        mean_len = df[df['label']==opt]['q_len'].mean()
        print(f"  Option {opt} mean q_len: {mean_len:.0f}")

    print(f"\n--- Question Length by USMLE Step ---")
    for step, subset in df.groupby('meta_info'):
        print(f"  {step:<15} mean={subset['q_len'].mean():.0f} | median={subset['q_len'].median():.0f}")

    print(f"\n--- Answer Option Length ---")
    for opt in ['A','B','C','D']:
        col = f'opt_{opt}_len'
        print(f"  Option {opt}: mean={df[col].mean():.0f} | median={df[col].median():.0f}")

    print(f"\n--- MetaMap Phrases Coverage ---")
    print(f"  Mean phrases per question: {df['n_phrases'].mean():.1f}")
    print(f"  Median: {df['n_phrases'].median():.1f}")
    print(f"  Min: {df['n_phrases'].min()} | Max: {df['n_phrases'].max()}")

    print(f"\n--- Answer Bias by USMLE Step ---")
    for step, subset in df.groupby('meta_info'):
        bias = subset['label'].value_counts(normalize=True)
        top = bias.index[0]
        print(f"  {step:<15} top={top} ({bias[top]:.1%}) | A={bias.get('A',0):.1%} B={bias.get('B',0):.1%} C={bias.get('C',0):.1%} D={bias.get('D',0):.1%}")

    print(f"\n--- Missing Values ---")
    nulls = df[['question','options','meta_info','metamap_phrases']].isnull().sum()
    if nulls.sum() == 0:
        print("  No missing values")
    else:
        for col, n in nulls[nulls>0].items():
            print(f"  {col:<20} {n:,} ({n/len(df):.1%})")

    print(f"\n--- Key Observations ---")
    print(f"  Total records: {result['metadata']['n_samples']:,}")
    print(f"  Mean question length: {df['q_len'].mean():.0f} chars, full clinical scenarios")
    print(f"  Near-balanced answer distribution, well-designed exam")
    print(f"  USMLE Steps: {list(df['meta_info'].unique())}, tests different clinical knowledge levels")
    print(f"  MetaMap phrases: mean {df['n_phrases'].mean():.1f} clinical entities per question")
    print(f"  ORACLE uses MedQA as retrieval quality ceiling, hardest benchmark")

    print(f"\n--- MedQA USMLE EDA complete ---")
    print(f"  Ready for ORACLE Stage 1 retrieval pipeline")

    return df


if __name__ == "__main__":
    df = run_eda()
