"""
ORACLE — MIRAGE Benchmark EDA
Phase 4 — Exploratory Data Analysis
RAG Evaluation Domain

EDA on MIRAGE benchmark — 7,663 medical QA questions (Xiong et al., 2024).
Compiled from 5 source datasets for RAG-specific evaluation.
Used as ORACLE's primary RAG pipeline evaluation framework.

Composition:
  medqa:    1,273 questions (USMLE clinical)
  medmcqa:  4,183 questions (Indian medical licensing)
  pubmedqa:   500 questions (biomedical research, yes/no/maybe)
  bioasq:     618 questions (biomedical, binary yes/no)
  mmlu:     1,089 questions (medical knowledge breadth)

Answer format varies by source:
  medqa/medmcqa/mmlu: 4-option ABCD
  pubmedqa: 3-option yes/no/maybe (A/B/C)
  bioasq: 2-option yes/no (A/B)
"""

import json
import pandas as pd
import numpy as np
import urllib.request
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from mirage_loader import load_mirage


def run_eda():
    print("ORACLE Phase 4 — MIRAGE Benchmark EDA")
    print("=" * 50)

    result = load_mirage()

    # Build unified dataframe
    frames = []
    for ds_name, ds_data in result.items():
        if ds_name == 'metadata':
            continue
        df_split = ds_data['data'].copy()
        df_split['source'] = ds_name
        frames.append(df_split)
    df = pd.concat(frames, ignore_index=True)

    # Derived fields
    df['q_len'] = df['question'].astype(str).str.len()
    df['n_options'] = df['options'].apply(lambda x: len(x) if isinstance(x, dict) else 0)
    df['answer_type'] = df['n_options'].map({2: 'binary', 3: 'three-way', 4: 'four-way'})

    print(f"\nDataset shape: {df.shape}")

    print(f"\n--- Split Composition ---")
    source_counts = df['source'].value_counts()
    for src in ['medqa', 'medmcqa', 'pubmedqa', 'bioasq', 'mmlu']:
        count = source_counts.get(src, 0)
        pct = count / len(df)
        print(f"  {src:<12} {count:,} ({pct:.1%})")
    print(f"  Total:       {len(df):,}")
    print(f"  Note: medmcqa dominates at {source_counts.get('medmcqa',0)/len(df):.1%} of benchmark")

    print(f"\n--- Answer Format by Source ---")
    for src, subset in df.groupby('source'):
        n_opts = subset['n_options'].mode()[0]
        fmt = {2: 'binary (A/B)', 3: 'three-way (A/B/C)', 4: 'four-way (A/B/C/D)'}.get(n_opts, 'unknown')
        print(f"  {src:<12} {fmt}")

    print(f"\n--- Overall Answer Distribution ---")
    answer_counts = df['answer'].value_counts().sort_index()
    for ans, count in answer_counts.items():
        print(f"  Option {ans}: {count:,} ({count/len(df):.1%})")

    print(f"\n--- Answer Distribution by Source ---")
    for src, subset in df.groupby('source'):
        bias = subset['answer'].value_counts(normalize=True)
        top = bias.index[0]
        bias_str = ' | '.join([f"{k}={v:.1%}" for k, v in bias.items()])
        print(f"  {src:<12} top={top} ({bias[top]:.1%}) | {bias_str}")

    print(f"\n--- Question Length by Source ---")
    for src, subset in df.groupby('source'):
        print(f"  {src:<12} mean={subset['q_len'].mean():.0f} | median={subset['q_len'].median():.0f} | min={subset['q_len'].min()} | max={subset['q_len'].max()}")
    print(f"  Overall:     mean={df['q_len'].mean():.0f} | median={df['q_len'].median():.0f}")

    print(f"\n--- Option Count Distribution ---")
    opt_counts = df['n_options'].value_counts().sort_index()
    for n, count in opt_counts.items():
        fmt = {2: 'binary', 3: 'three-way', 4: 'four-way'}.get(n, 'other')
        print(f"  {n} options ({fmt:<12}): {count:,} ({count/len(df):.1%})")

    print(f"\n--- PMID Coverage ---")
    pmid_sources = ['bioasq', 'pubmedqa']
    for src in pmid_sources:
        subset = df[df['source'] == src]
        if 'PMID' in subset.columns:
            has_pmid = subset['PMID'].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
            print(f"  {src:<12} PMID coverage: {has_pmid:,}/{len(subset):,} ({has_pmid/len(subset):.1%})")
    non_pmid = [s for s in df['source'].unique() if s not in pmid_sources]
    print(f"  {', '.join(non_pmid)}: no PMID — closed-book QA format")

    print(f"\n--- RAG Difficulty Proxy (Question Length) ---")
    print(f"  Shortest questions = less retrieval context needed")
    print(f"  Longest questions = richer clinical scenarios, more retrieval anchors")
    for src, subset in df.groupby('source'):
        long_q = (subset['q_len'] > df['q_len'].quantile(0.75)).sum()
        print(f"  {src:<12} questions in top quartile of length: {long_q:,} ({long_q/len(subset):.1%})")





    print(f"\n--- Word Count by Source ---")
    df['q_words'] = df['question'].astype(str).str.split().str.len()
    for src, subset in df.groupby('source'):
        print(f"  {src:<12} mean={subset['q_words'].mean():.1f} | median={subset['q_words'].median():.0f} | min={subset['q_words'].min()} | max={subset['q_words'].max()}")
    print(f"  Overall:     mean={df['q_words'].mean():.1f} | median={df['q_words'].median():.0f}")
    medqa_words = df[df['source']=='medqa']['q_words'].mean()
    bioasq_words = df[df['source']=='bioasq']['q_words'].mean()
    print(f"  Note: medqa mean {medqa_words:.0f} words vs bioasq mean {bioasq_words:.0f} words — clinical scenarios vs factual queries")

    print(f"\n--- Option Text Length by Source ---")
    for src, subset in df.groupby('source'):
        opt_lens = []
        for opts in subset['options']:
            if isinstance(opts, dict):
                opt_lens.extend([len(str(v)) for v in opts.values()])
        if opt_lens:
            import numpy as np
            print(f"  {src:<12} mean_option_len={np.mean(opt_lens):.0f} | median={np.median(opt_lens):.0f} | min={min(opt_lens)} | max={max(opt_lens)}")
    print(f"  Note: bioasq/pubmedqa options are 2-3 chars (yes/no/maybe) vs medqa/mmlu options are 21-33 chars mean (full clinical statements)")

    print(f"\n--- Missing Values ---")
    nulls = df[['question', 'options', 'answer', 'source']].isnull().sum()
    if nulls.sum() == 0:
        print("  No missing values")
    else:
        for col, n in nulls[nulls > 0].items():
            print(f"  {col:<20} {n:,} ({n/len(df):.1%})")

    print(f"\n--- Key Observations ---")
    print(f"  Total records: {len(df):,} across 5 source datasets")
    print(f"  medmcqa dominates: {source_counts.get('medmcqa',0):,} questions ({source_counts.get('medmcqa',0)/len(df):.1%})")
    print(f"  Three answer formats: binary (bioasq), three-way (pubmedqa), four-way (medqa/medmcqa/mmlu)")
    print(f"  PMID available for bioasq + pubmedqa — enables retrieval ground truth validation")
    print(f"  Mean question length: {df['q_len'].mean():.0f} chars — varies widely by source")
    print(f"  ORACLE uses MIRAGE as end-to-end RAG evaluation — tests full pipeline not just retrieval")
    print(f"  RAG-specific design: question-only retrieval setting, no options during retrieval phase")

    print(f"\n--- MIRAGE EDA complete ---")
    print(f"  Ready for ORACLE Stage 1 retrieval pipeline evaluation")

    return df


if __name__ == "__main__":
    df = run_eda()
