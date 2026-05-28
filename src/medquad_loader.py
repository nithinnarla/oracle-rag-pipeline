"""
ORACLE — MedQuAD Dataset Loader
Optimized Retrieval Augmented Generation for Personalized Health Information Accessibility

MedQuAD — 47,441 medical QA pairs (Ben Abacha & Demner-Fushman, 2019)
Source: HuggingFace — lavita/MedQuAD

Dataset note: Initially planned as "Consumer Health QA (Ben Abacha et al., 2020)"
During verification May 2026, the cited dataset could not be located as a
standalone downloadable resource. MedQuAD is the appropriate replacement —
same research group (NLM/NIH), same underlying infrastructure, more comprehensive
coverage with 47,441 QA pairs from 12 NIH websites.

Why MedQuAD for ORACLE:
47,441 patient-facing QA pairs curated from 12 NIH websites including
MedlinePlus, NIDDK, NCI, and others. Questions written by health consumers,
answers written for health consumers. The most relevant dataset for ORACLE's
core research question — does literacy-conditioned retrieval improve outcomes
for patients seeking health information online.

This is the critical accessibility evaluation dataset. Unlike PubMedQA, MedMCQA,
and MedQA which test clinical professional knowledge, MedQuAD tests the exact
patient-facing health information access scenario ORACLE is designed to improve.
"""

import pandas as pd
from datasets import load_dataset


def load_medquad() -> dict:
    """
    Load MedQuAD — 47,441 patient-facing medical QA pairs from 12 NIH websites.
    Source: HuggingFace lavita/MedQuAD
    """
    print("Loading MedQuAD...")
    dataset = load_dataset("lavita/MedQuAD")
    df = dataset['train'].to_pandas()

    print(f"  Records:     {len(df):,}")
    print(f"  Sources:     {df['document_source'].nunique()} NIH websites")
    print(f"  Categories:  {df['category'].nunique()} unique categories")
    print(f"  Top sources: {df['document_source'].value_counts().head(5).to_dict()}")

    X = df[['document_id', 'document_source', 'category',
            'question_focus', 'question_type', 'question']].copy()
    y = df['answer'].copy()

    metadata = {
        'name': 'MedQuAD',
        'n_samples': len(X),
        'source': 'HuggingFace — lavita/MedQuAD',
        'purpose': 'Patient-facing health information accessibility evaluation',
        'note': 'Replaces Consumer Health QA — same NLM/NIH research group, more comprehensive',
        'paper': 'Ben Abacha & Demner-Fushman (2019) — MedQuAD, BMC Bioinformatics'
    }

    print(f"\nMedQuAD loaded: {len(X):,} patient-facing QA pairs")
    print(f"  Purpose: {metadata['purpose']}")

    return {'data': X, 'labels': y, 'metadata': metadata}


if __name__ == "__main__":
    result = load_medquad()
    print(f"\n  Total verified: {result['metadata']['n_samples']:,} records")
