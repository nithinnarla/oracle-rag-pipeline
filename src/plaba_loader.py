"""
ORACLE — PLABA Dataset Loader
Optimized Retrieval Augmented Generation for Personalized Health Information Accessibility

PLABA — 921 plain language adaptation instances (Attal et al., 2023)
Source: OSF Repository — osf.io/rnpmf

Splits:
- train: 635 instances
- val:   138 instances
- test:  148 instances

Note: Paper reports 750 abstracts — actual row count is 921 because
multiple adaptation versions exist per abstract. Each row represents
one expert-created plain language adaptation of one PubMed abstract.

Why PLABA for ORACLE:
Gold standard plain language adaptation dataset — 75 health topics,
10 PubMed abstracts per topic, expert-created sentence-level adaptations
from NLM annotators. The only dataset with paired professional and
plain language versions of biomedical abstracts at sentence level.

Used as ORACLE's primary plain language evaluation dataset — directly
measures whether literacy-conditioned generation produces output
comparable to expert human plain language adaptation.

Source: OSF repository accessed via API (direct download URL was
returning 500 errors; resolved via osf.io API file enumeration).
"""

import urllib.request
import pandas as pd
import io


PLABA_URLS = {
    'train': 'https://osf.io/download/g3t5x/',
    'val': 'https://osf.io/download/qa3hd/',
    'test': 'https://osf.io/download/6ksbm/'
}


def load_plaba_split(split: str = 'train') -> dict:
    """
    Load a single PLABA split.

    Args:
        split: 'train', 'val', or 'test'

    Returns:
        dict with keys: data (DataFrame), metadata (dict)
    """
    print(f"Loading PLABA {split} split...")
    response = urllib.request.urlopen(PLABA_URLS[split], timeout=60)
    df = pd.read_csv(io.BytesIO(response.read()))

    print(f"  Records:  {len(df):,}")
    print(f"  Topics:   {df['question'].nunique()} health topics")

    X = df[['question', 'pmid', 'input_text', 'Question_Type']].copy()
    y = df['target_text'].copy()

    return {
        'data': X,
        'labels': y,
        'metadata': {
            'name': f'PLABA-{split}',
            'n_samples': len(df),
            'purpose': 'Plain language adaptation evaluation — gold standard',
            'paper': 'Attal et al. (2023) — PLABA, Scientific Data'
        }
    }


def load_plaba_all() -> dict:
    """
    Load all PLABA splits — 921 total instances.
    """
    print("Loading PLABA — all splits...")
    splits = {}
    total = 0

    for split in ['train', 'val', 'test']:
        result = load_plaba_split(split)
        splits[split] = result
        total += result['metadata']['n_samples']

    print(f"\nPLABA loaded: {total:,} total instances")
    print(f"  Note: 750 unique abstracts across 75 topics — 921 rows due to multiple adaptation versions")

    splits['metadata'] = {
        'name': 'PLABA',
        'n_samples': total,
        'source': 'OSF — osf.io/rnpmf',
        'paper': 'Attal et al. (2023) — PLABA, Scientific Data'
    }

    return splits


if __name__ == "__main__":
    result = load_plaba_all()
    print(f"\n  Total verified: {result['metadata']['n_samples']:,} records")
