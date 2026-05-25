"""
ORACLE — MedMCQA Dataset Loader
Optimized Retrieval Augmented Generation for Personalized Health Information Accessibility

MedMCQA — 193,155 medical QA instances (Pal et al., 2022)
Source: HuggingFace — openlifescienceai/medmcqa

Splits:
- train:      182,822 instances
- test:         6,150 instances
- validation:   4,183 instances

Why MedMCQA for ORACLE:
Large-scale medical school exam questions covering 2,400+ healthcare topics
across 21 medical subjects. Used as ORACLE's broad medical QA retrieval
evaluation benchmark — tests whether literacy-conditioned retrieval surfaces
relevant medical knowledge before accessibility adaptation.

Limitation: Medical school exam questions written for clinical professionals.
ORACLE uses MedMCQA as retrieval quality baseline, not accessibility evaluation.
Accessibility evaluation handled by Consumer Health QA and PLABA.
"""

import pandas as pd
from datasets import load_dataset


def load_medmcqa(split: str = "train") -> dict:
    """
    Load MedMCQA split for ORACLE retrieval evaluation.

    Args:
        split: 'train', 'test', or 'validation'

    Returns:
        dict with keys: data (DataFrame), labels (Series), metadata (dict)
    """
    print(f"Loading MedMCQA {split} split...")
    dataset = load_dataset("openlifescienceai/medmcqa")
    df = dataset[split].to_pandas()

    print(f"  Records:     {len(df):,}")
    print(f"  Subjects:    {df['subject_name'].nunique()} unique medical subjects")

    X = df[['id', 'question', 'opa', 'opb', 'opc', 'opd', 'exp', 'subject_name', 'topic_name']].copy()
    y = df['cop'].copy()

    metadata = {
        'name': f'MedMCQA-{split}',
        'n_samples': len(X),
        'purpose': 'Medical QA retrieval evaluation baseline',
        'paper': 'Pal et al. (2022) — MedMCQA, CHIL'
    }

    return {'data': X, 'labels': y, 'metadata': metadata}


def load_medmcqa_all() -> dict:
    """
    Load all MedMCQA splits — 193,155 total records.
    """
    print("Loading MedMCQA — all splits...")
    dataset = load_dataset("openlifescienceai/medmcqa")

    splits = {}
    total = 0
    for split in ['train', 'test', 'validation']:
        df = dataset[split].to_pandas()
        splits[split] = {
            'data': df[['id', 'question', 'opa', 'opb', 'opc', 'opd', 'exp', 'subject_name', 'topic_name']].copy(),
            'labels': df['cop'].copy(),
            'metadata': {
                'name': f'MedMCQA-{split}',
                'n_samples': len(df)
            }
        }
        total += len(df)
        print(f"  {split}: {len(df):,} records")

    print(f"\nMedMCQA loaded: {total:,} total records")

    splits['metadata'] = {
        'name': 'MedMCQA',
        'n_samples': total,
        'source': 'HuggingFace — openlifescienceai/medmcqa',
        'paper': 'Pal et al. (2022) — MedMCQA, CHIL'
    }

    return splits


if __name__ == "__main__":
    result = load_medmcqa_all()
    print(f"\n  Total verified: {result['metadata']['n_samples']:,} records")
