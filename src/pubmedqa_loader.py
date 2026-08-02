"""
ORACLE — PubMedQA Dataset Loader
Optimized Retrieval Augmented Generation for Personalized Health Information Accessibility

PubMedQA — 273,518 biomedical QA instances (Jin et al., 2019)
Source: HuggingFace — qiaojin/PubMedQA

Three configurations:
- pqa_labeled:     1,000 expert-annotated instances — gold standard evaluation
- pqa_unlabeled:  61,249 instances — retrieval corpus expansion
- pqa_artificial: 211,269 instances — machine-generated training data

Why PubMedQA for ORACLE:
PubMedQA is the standard biomedical QA benchmark. Questions are answerable
from provided PubMed abstracts with yes/no/maybe labels. ORACLE uses all
three splits — labeled for evaluation, unlabeled and artificial for retrieval
corpus and training pipeline.

Limitation: Questions written by researchers for researchers — not patient-facing.
ORACLE uses PubMedQA to establish retrieval quality baseline before evaluating
accessibility on Consumer Health QA and PLABA.
"""

from datasets import load_dataset


def load_pubmedqa_labeled() -> dict:
    print("Loading PubMedQA labeled split...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    df = dataset['train'].to_pandas()
    print(f"  Records:     {len(df):,}")
    print(f"  Label dist:  {df['final_decision'].value_counts().to_dict()}")
    return {
        'data': df[['pubid', 'question', 'context', 'long_answer']].copy(),
        'labels': df['final_decision'].copy(),
        'metadata': {
            'name': 'PubMedQA-labeled',
            'n_samples': len(df),
            'purpose': 'Gold standard evaluation — expert annotated',
            'paper': 'Jin et al. (2019) — PubMedQA, EMNLP'
        }
    }


def load_pubmedqa_unlabeled() -> dict:
    print("Loading PubMedQA unlabeled split...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")
    df = dataset['train'].to_pandas()
    print(f"  Records:     {len(df):,}")
    return {
        'data': df[['pubid', 'question', 'context', 'long_answer']].copy(),
        'metadata': {
            'name': 'PubMedQA-unlabeled',
            'n_samples': len(df),
            'purpose': 'Retrieval corpus expansion',
            'paper': 'Jin et al. (2019) — PubMedQA, EMNLP'
        }
    }


def load_pubmedqa_artificial() -> dict:
    print("Loading PubMedQA artificial split...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
    df = dataset['train'].to_pandas()
    print(f"  Records:     {len(df):,}")
    print(f"  Label dist:  {df['final_decision'].value_counts().to_dict()}")
    return {
        'data': df[['pubid', 'question', 'context', 'long_answer']].copy(),
        'labels': df['final_decision'].copy(),
        'metadata': {
            'name': 'PubMedQA-artificial',
            'n_samples': len(df),
            'purpose': 'Training pipeline — machine generated labels',
            'paper': 'Jin et al. (2019) — PubMedQA, EMNLP'
        }
    }


def load_pubmedqa_all() -> dict:
    labeled = load_pubmedqa_labeled()
    unlabeled = load_pubmedqa_unlabeled()
    artificial = load_pubmedqa_artificial()

    total = (labeled['metadata']['n_samples'] +
             unlabeled['metadata']['n_samples'] +
             artificial['metadata']['n_samples'])

    print(f"\nPubMedQA loaded: {total:,} total records")
    print(f"  Labeled:     {labeled['metadata']['n_samples']:,} (evaluation)")
    print(f"  Unlabeled:   {unlabeled['metadata']['n_samples']:,} (retrieval corpus)")
    print(f"  Artificial:  {artificial['metadata']['n_samples']:,} (training)")

    return {
        'labeled': labeled,
        'unlabeled': unlabeled,
        'artificial': artificial,
        'metadata': {
            'name': 'PubMedQA',
            'n_samples': total,
            'source': 'HuggingFace — qiaojin/PubMedQA',
            'paper': 'Jin et al. (2019) — PubMedQA, EMNLP'
        }
    }


if __name__ == "__main__":
    result = load_pubmedqa_all()
    print(f"\n  Total verified: {result['metadata']['n_samples']:,} records")
