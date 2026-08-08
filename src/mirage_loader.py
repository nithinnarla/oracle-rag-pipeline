"""
ORACLE, MIRAGE Benchmark Loader
Optimized Retrieval Augmented Generation for Personalized Health Information Accessibility

MIRAGE, 7,663 medical QA questions (Xiong et al., 2024)
Source: GitHub, Teddy-XiongGZ/MIRAGE (benchmark.json)

Composition:
- medqa:    1,273 questions
- medmcqa:  4,183 questions
- pubmedqa:   500 questions
- bioasq:     618 questions
- mmlu:     1,089 questions

Why MIRAGE for ORACLE:
MIRAGE is the first comprehensive RAG-specific evaluation benchmark for medicine.
Unlike individual QA datasets, MIRAGE evaluates the full RAG pipeline.
retrieval quality, generation accuracy, and faithfulness together.
Used as ORACLE's primary RAG evaluation framework, directly tests whether
literacy-conditioned retrieval improves medical RAG performance.

Note: MIRAGE compiles subsets from existing datasets (MedQA, MedMCQA,
PubMedQA, BioASQ, MMLU-Med). Questions are curated for RAG evaluation
specifically, question-only retrieval setting, no answer options provided
during retrieval phase.
"""

import json
import pandas as pd
import urllib.request


MIRAGE_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"


def load_mirage() -> dict:
    """
    Load MIRAGE benchmark, 7,663 medical QA questions across 5 datasets.
    Source: GitHub Teddy-XiongGZ/MIRAGE benchmark.json
    """
    print("Loading MIRAGE benchmark...")
    print(f"  Source: {MIRAGE_URL}")

    response = urllib.request.urlopen(MIRAGE_URL, timeout=60)
    data = json.loads(response.read())

    splits = {}
    total = 0

    for dataset_name, questions in data.items():
        count = len(questions)
        total += count
        rows = []
        for qid, qdata in questions.items():
            row = {'question_id': qid, 'dataset': dataset_name}
            row.update(qdata)
            rows.append(row)
        splits[dataset_name] = {
            'data': pd.DataFrame(rows),
            'metadata': {
                'name': f'MIRAGE-{dataset_name}',
                'n_samples': count
            }
        }
        print(f"  {dataset_name}: {count:,} questions")

    print(f"\nMIRAGE loaded: {total:,} total questions")

    splits['metadata'] = {
        'name': 'MIRAGE',
        'n_samples': total,
        'source': 'GitHub, Teddy-XiongGZ/MIRAGE',
        'paper': 'Xiong et al. (2024), MIRAGE: Benchmarking RAG for Medicine, ACL Findings'
    }

    return splits


if __name__ == "__main__":
    result = load_mirage()
    print(f"\n  Total verified: {result['metadata']['n_samples']:,} records")
