"""
ORACLE, MedQA USMLE Dataset Loader
Optimized Retrieval Augmented Generation for Personalized Health Information Accessibility

MedQA USMLE, 11,451 clinical QA instances (Jin et al., 2021)
Source: HuggingFace, GBaker/MedQA-USMLE-4-options

Splits:
- train: 10,178 instances
- test:   1,273 instances

Note: HuggingFace 4-options version contains 11,451 records.
Original paper reported 12,723, difference due to 4-option filtering
of the full dataset which includes 5-option questions.

Why MedQA USMLE for ORACLE:
USMLE board exam questions require genuine clinical reasoning, the hardest
biomedical QA benchmark. Used as ORACLE's clinical retrieval quality ceiling
evaluation. If literacy-conditioned retrieval maintains USMLE performance
while improving accessibility, that validates the architectural approach.

Limitation: USMLE tests clinicians not patients. Used for retrieval
quality evaluation only, not accessibility evaluation.
"""

from datasets import load_dataset


def load_medqa(split: str = "train") -> dict:
    print(f"Loading MedQA USMLE {split} split...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    df = dataset[split].to_pandas()
    print(f"  Records:     {len(df):,}")
    print(f"  Features:    {list(df.columns)}")
    X = df[['question', 'options', 'meta_info', 'metamap_phrases']].copy()
    y = df['answer_idx'].copy()
    metadata = {
        'name': f'MedQA-USMLE-{split}',
        'n_samples': len(X),
        'purpose': 'Clinical QA retrieval evaluation, USMLE standard',
        'paper': 'Jin et al. (2021), MedQA, Applied Sciences'
    }
    return {'data': X, 'labels': y, 'metadata': metadata}


def load_medqa_all() -> dict:
    print("Loading MedQA USMLE, all splits...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    splits = {}
    total = 0
    for split in ['train', 'test']:
        df = dataset[split].to_pandas()
        splits[split] = {
            'data': df[['question', 'options', 'meta_info', 'metamap_phrases']].copy(),
            'labels': df['answer_idx'].copy(),
            'metadata': {
                'name': f'MedQA-USMLE-{split}',
                'n_samples': len(df)
            }
        }
        total += len(df)
        print(f"  {split}: {len(df):,} records")
    print(f"\nMedQA USMLE loaded: {total:,} total records")
    print(f"  Note: 4-options version, original paper reported 12,723")
    splits['metadata'] = {
        'name': 'MedQA-USMLE',
        'n_samples': total,
        'source': 'HuggingFace, GBaker/MedQA-USMLE-4-options',
        'paper': 'Jin et al. (2021), MedQA, Applied Sciences'
    }
    return splits


if __name__ == "__main__":
    result = load_medqa_all()
    print(f"\n  Total verified: {result['metadata']['n_samples']:,} records")
