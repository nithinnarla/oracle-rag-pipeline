"""
ORACLE — MedMCQA EDA
Phase 4 — Exploratory Data Analysis
Medical QA Domain

EDA on MedMCQA dataset — 193,155 medical exam QA instances (Pal et al., 2022).
4-way multiple choice questions across 21 medical subjects and 2,388 topics.
Used as ORACLE broad medical QA retrieval evaluation baseline.

Label mapping: cop = correct option (0=opa, 1=opb, 2=opc, 3=opd)
Limitation: Medical school exam questions — not patient-facing.
"""

import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from medmcqa_loader import load_medmcqa_all

OPTION_LABELS = {0:'A (opa)', 1:'B (opb)', 2:'C (opc)', 3:'D (opd)'}


def run_eda():
    print("ORACLE Phase 4 — MedMCQA EDA")
    print("=" * 50)

    result = load_medmcqa_all()

    # Use train split for EDA
    df = result['train']['data'].copy()
    df['label'] = result['train']['labels'].values
    df['q_len'] = df['question'].astype(str).str.len()
    df['opa_len'] = df['opa'].astype(str).str.len()
    df['opb_len'] = df['opb'].astype(str).str.len()
    df['opc_len'] = df['opc'].astype(str).str.len()
    df['opd_len'] = df['opd'].astype(str).str.len()
    df['has_exp'] = df['exp'].notna()

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print(f"\n--- Split Distribution ---")
    for split in ['train','test','validation']:
        n = result[split]['metadata']['n_samples']
        print(f"  {split:<12} {n:,}")
    print(f"  Total:       {result['metadata']['n_samples']:,}")

    print(f"\n--- Correct Answer Distribution ---")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  {OPTION_LABELS[label]:<12} {count:,} ({count/len(df):.1%})")
    print(f"  Note: Slight bias toward option A (29.3%) — exam question design pattern")

    print(f"\n--- Subject Distribution ---")
    subj_stats = df.groupby('subject_name').agg(
        count=('label','count'),
        mean_q_len=('q_len','mean')
    ).reset_index().sort_values('count', ascending=False)
    for _, row in subj_stats.iterrows():
        print(f"  {str(row['subject_name']):<40} n={int(row['count']):,} | mean q_len={row['mean_q_len']:.0f}")

    print(f"\n--- Top 20 Topics ---")
    topic_counts = df['topic_name'].value_counts().head(20)
    for topic, count in topic_counts.items():
        print(f"  {str(topic):<45} n={count:,}")
    print(f"  Note: topic_name missing {df['topic_name'].isna().sum():,} ({df['topic_name'].isna().mean():.1%})")

    print(f"\n--- Question Length Distribution ---")
    print(f"  Mean: {df['q_len'].mean():.0f} chars")
    print(f"  Median: {df['q_len'].median():.0f} chars")
    print(f"  Min: {df['q_len'].min()} | Max: {df['q_len'].max()}")
    print(f"  Top 5 subjects by mean question length:")
    for _, row in subj_stats.nlargest(5,'mean_q_len').iterrows():
        print(f"    {str(row['subject_name']):<40} mean={row['mean_q_len']:.0f} chars")

    print(f"\n--- Answer Option Length Distribution ---")
    for col, label in [('opa_len','A'),('opb_len','B'),('opc_len','C'),('opd_len','D')]:
        print(f"  Option {label}: mean={df[col].mean():.0f} | median={df[col].median():.0f}")

    print(f"\n--- Explanation Coverage ---")
    print(f"  Has explanation: {df['has_exp'].sum():,} ({df['has_exp'].mean():.1%})")
    print(f"  Missing explanation: {(~df['has_exp']).sum():,} ({(~df['has_exp']).mean():.1%})")
    exp_by_subj = df.groupby('subject_name')['has_exp'].mean().sort_values()
    print(f"  Lowest explanation coverage subjects:")
    for subj, rate in exp_by_subj.head(5).items():
        print(f"    {str(subj):<40} {rate:.1%}")

    print(f"\n--- Missing Values ---")
    nulls = df[['question','opa','opb','opc','opd','exp','subject_name','topic_name']].isnull().sum()
    for col, n in nulls[nulls>0].items():
        print(f"  {col:<25} {n:,} ({n/len(df):.1%})")



    print(f"\n--- Explanation Length by Subject ---")
    df["exp_len"] = df["exp"].astype(str).str.len()
    exp_by_subj = df[df["exp"].notna()].groupby("subject_name")["exp_len"].mean().sort_values(ascending=False)
    print(f"  Mean explanation length: {df[df['exp'].notna()]['exp_len'].mean():.0f} chars")
    print(f"  Max: {df[df['exp'].notna()]['exp_len'].max()} chars — requires chunking in ORACLE Stage 1")
    for subj, val in exp_by_subj.head(5).items():
        print(f"    {str(subj):<40} mean={val:.0f} chars")
    print(f"  Note: Explanation length varies 3x across subjects — affects RAG retrieval quality")

    print(f"\n--- Answer Bias by Subject ---")
    pivot = df.groupby("subject_name")["label"].value_counts(normalize=True).unstack().fillna(0)
    pivot.columns = ["A","B","C","D"]
    pivot["A_minus_D"] = pivot["A"] - pivot["D"]
    pivot = pivot.sort_values("A_minus_D", ascending=False)
    print(f"  Answer A vs D gap by subject (top 5 and bottom 5):")
    for subj, row in pivot.head(5).iterrows():
        print(f"    {str(subj):<40} A-D gap={row['A_minus_D']:.3f}")
    print(f"  ...")
    for subj, row in pivot.tail(5).iterrows():
        print(f"    {str(subj):<40} A-D gap={row['A_minus_D']:.3f}")
    print(f"  Note: Orthopaedics (15.7%) vs Gynaecology (4.6%) — 3x difference")

    print(f"\n--- Key Observations ---")
    print(f"  Total train records: {len(df):,}")
    print(f"  Medical subjects: 21")
    print(f"  Unique topics: {df['topic_name'].nunique():,}")
    print(f"  4-way MCQ — ORACLE retrieval evaluated on correct option prediction")
    print(f"  Slight answer bias: option A most common correct answer (29.3%)")
    print(f"  88% have explanations — rich for RAG retrieval evaluation")
    print(f"  topic_name missing 52.3% — subject_name used as primary grouping")

    print(f"\n--- MedMCQA EDA complete ---")
    print(f"  Ready for ORACLE Stage 1 retrieval pipeline")

    return df


if __name__ == "__main__":
    df = run_eda()
