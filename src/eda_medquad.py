"""
ORACLE — MedQuAD EDA
Phase 4 — Exploratory Data Analysis
Patient-Facing Health Information Domain

EDA on MedQuAD — 47,441 medical QA pairs (Ben Abacha & Demner-Fushman, 2019).
Source: HuggingFace lavita/MedQuAD — 12 NIH websites.
Used as ORACLE's primary patient-facing health information accessibility evaluation.

Key finding: 65.4% of records have no answer (31,034 null answers).
Missingness is structural — ADAM (17,348) and MPlusDrugs (12,889) have 0% coverage.
Usable for RAG evaluation: 16,407 records with answers (34.6% of total).

Label: answer (free text, mean=1,303 chars for non-null)
Features: document_source, category, question_type, question_focus, question
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset


def run_eda():
    print("ORACLE Phase 4 — MedQuAD EDA")
    print("=" * 50)

    dataset = load_dataset("lavita/MedQuAD")
    df = dataset['train'].to_pandas()

    # Derived fields
    df['has_answer'] = df['answer'].notna()
    df['q_len'] = df['question'].astype(str).str.len()
    df['ans_len'] = df['answer'].astype(str).str.len()
    df_ans = df[df['has_answer']].copy()
    df_ans['ans_len'] = df_ans['answer'].astype(str).str.len()

    print(f"\nDataset shape: {df.shape}")

    print(f"\n--- Dataset Overview ---")
    print(f"  Total records:     {len(df):,}")
    print(f"  With answers:      {df['has_answer'].sum():,} ({df['has_answer'].mean():.1%})")
    print(f"  Without answers:   {(~df['has_answer']).sum():,} ({(~df['has_answer']).mean():.1%})")
    print(f"  NIH sources:       {df['document_source'].nunique()}")
    print(f"  Question types:    {df['question_type'].nunique()}")
    print(f"  Unique conditions: {df['question_focus'].nunique():,}")

    print(f"\n--- Answer Coverage by Source ---")
    for src, subset in df.groupby('document_source'):
        has = subset['has_answer'].sum()
        pct = has / len(subset)
        status = "FULL" if pct == 1.0 else ("NONE" if pct == 0.0 else "PARTIAL")
        print(f"  {src:<30} {has:,}/{len(subset):,} ({pct:.1%}) [{status}]")

    print(f"\n--- Structural Missingness Finding ---")
    full_sources = [s for s, g in df.groupby('document_source') if g['has_answer'].mean() == 1.0]
    none_sources = [s for s, g in df.groupby('document_source') if g['has_answer'].mean() == 0.0]
    print(f"  Full coverage sources ({len(full_sources)}): {', '.join(full_sources)}")
    print(f"  Zero coverage sources ({len(none_sources)}): {', '.join(none_sources)}")
    zero_records = df[df['document_source'].isin(none_sources)].shape[0]
    print(f"  Zero-coverage records: {zero_records:,} ({zero_records/len(df):.1%} of total)")
    print(f"  Note: ADAM + MPlusDrugs + MPlusHerbsSupplements = {zero_records:,} records with NO answers (plus 5 GARD records = {df['has_answer'].eq(False).sum():,} total)")
    print(f"  RAG-usable subset: {df['has_answer'].sum():,} records from {len(full_sources)} sources")

    print(f"\n--- Answer Coverage by Category ---")
    for cat, subset in df.groupby('category', dropna=False):
        has = subset['has_answer'].sum()
        print(f"  {str(cat):<15} {has:,}/{len(subset):,} ({has/len(subset):.1%})")
    print(f"  Note: Drug category = 0% coverage. Disease = 4.2%. NaN category = 100%.")
    print(f"  Finding: UMLS metadata presence inversely correlated with answer availability")

    print(f"\n--- Question Type Distribution (Top 15) ---")
    qt_counts = df['question_type'].value_counts().head(15)
    for qt, count in qt_counts.items():
        has = df[df['question_type'] == qt]['has_answer'].sum()
        print(f"  {qt:<45} total={count:,} | with_answer={has:,} ({has/count:.1%})")

    print(f"\n--- Question Length Distribution ---")
    print(f"  Mean: {df['q_len'].mean():.0f} chars | Median: {df['q_len'].median():.0f}")
    print(f"  Min: {df['q_len'].min()} | Max: {df['q_len'].max()}")
    for src, subset in df.groupby('document_source'):
        print(f"  {src:<30} mean={subset['q_len'].mean():.0f} | median={subset['q_len'].median():.0f}")

    print(f"\n--- Answer Length Distribution (Non-Null Only) ---")
    print(f"  Usable records: {len(df_ans):,}")
    print(f"  Mean: {df_ans['ans_len'].mean():.0f} chars | Median: {df_ans['ans_len'].median():.0f}")
    print(f"  Min: {df_ans['ans_len'].min()} | Max: {df_ans['ans_len'].max()}")
    print(f"  Note: Mean answer length {df_ans['ans_len'].mean():.0f} chars — long narrative answers")
    print(f"  ORACLE chunking strategy must handle long answers")

    print(f"\n--- Answer Coverage by Question Type (Top 10 by coverage) ---")
    qt_coverage = df.groupby('question_type').apply(
        lambda x: x['has_answer'].mean()).sort_values(ascending=False).head(10)
    for qt, pct in qt_coverage.items():
        count = df[df['question_type'] == qt].shape[0]
        print(f"  {qt:<45} {pct:.1%} ({count:,} total)")

    print(f"\n--- Top Question Focus Areas ---")
    top_focus = df['question_focus'].value_counts().head(10)
    for focus, count in top_focus.items():
        print(f"  {focus:<35} {count:,}")


    print(f"\n--- Answer Length by Source (Non-Null Only) ---")
    for src, subset in df_ans.groupby('document_source'):
        print(f"  {src:<30} mean={subset['ans_len'].mean():.0f} | median={subset['ans_len'].median():.0f} | n={len(subset):,}")
    print(f"  Overall:                       mean={df_ans['ans_len'].mean():.0f} | median={df_ans['ans_len'].median():.0f} | n={len(df_ans):,}")
    print(f"  Note: answer length varies significantly by NIH source — affects ORACLE chunking strategy")

    print(f"\n--- Missing Values ---")
    key_cols = ['document_source', 'category', 'question_type', 'question', 'answer', 'question_focus']
    for col in key_cols:
        nulls = df[col].isnull().sum()
        print(f"  {col:<25} {nulls:,} ({nulls/len(df):.1%})")
    print(f"  Note: answer nulls are structural (ADAM/MPlusDrugs/MPlusHerbsSupplements + 5 GARD records), not random missingness")

    print(f"\n--- Key Observations ---")
    print(f"  Total records: {len(df):,} across {df['document_source'].nunique()} NIH sources")
    print(f"  RAG-usable: {df['has_answer'].sum():,} ({df['has_answer'].mean():.1%}) — structural missingness not random")
    print(f"  ADAM ({df[df['document_source']=='ADAM'].shape[0]:,}) + MPlusDrugs ({df[df['document_source']=='MPlusDrugs'].shape[0]:,}) + MPlusHerbsSupplements ({df[df['document_source']=='MPlusHerbsSupplements'].shape[0]:,}) = zero answer coverage + 5 GARD records = {(~df['has_answer']).sum():,} total missing")
    print(f"  Drug category: 0% answer coverage — all drug QA pairs lack answers")
    print(f"  Mean answer length: {df_ans['ans_len'].mean():.0f} chars — long narrative patient-facing answers")
    print(f"  39 question types — most granular taxonomy across all ORACLE datasets")
    print(f"  ORACLE uses MedQuAD as patient-facing accessibility benchmark — tests lay language retrieval")

    print(f"\n--- MedQuAD EDA complete ---")
    print(f"  RAG-usable subset: {df['has_answer'].sum():,} records ready for ORACLE evaluation")

    return df


if __name__ == "__main__":
    df = run_eda()
