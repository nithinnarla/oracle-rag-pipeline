"""
ORACLE — Retrieval Pipeline
Phase 4 — Stage 2: Literacy-Conditioned Dense Retrieval

End-to-end retrieval pipeline for literacy-conditioned health RAG.
Given a user query, routes to the correct literacy band index and
retrieves top-k relevant documents using DPR dense retrieval.

Architecture:
1. Classify query literacy band (FK-based rule routing)
2. Encode query using DPR query encoder (768-dim)
3. Compute cosine similarity against band-specific corpus embeddings
4. Return top-k documents with routing metadata for Stage 3

Stage 3 requirement (per methodology_decisions.md Decision 11):
- Routing metadata (band, fk_grade, margin) passed to Stage 3
- Stage 3 must not assume routing is correct
- Stage 3 literacy correction layer operates regardless of routing band

Input: user query (string), top_k (int, default 5)
Output: dict with retrieved documents + routing metadata

Pipeline/infrastructure script — no notebook.
"""

import numpy as np
import os
import sys
import warnings
import logging
import pandas as pd
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(__file__))
from literacy_classifier import classify_query, get_band_embeddings
from dpr_encoder import get_dpr_query_encoder, encode_query

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(REPO_ROOT, "data", "processed", "oracle_corpus.csv")


# Load corpus once at module level
_CORPUS = None

def _get_corpus():
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = pd.read_csv(CORPUS_PATH).set_index('record_id')
    return _CORPUS


def cosine_similarity_batch(query_vec: np.ndarray,
                             corpus_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query vector and corpus matrix.
    query_vec: (768,)
    corpus_vecs: (N, 768)
    Returns: (N,) similarity scores
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    corpus_norms = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-10)
    return corpus_norms @ query_norm


def retrieve(query: str,
             top_k: int = 5,
             band_override: str = None,
             q_tokenizer=None,
             q_model=None) -> dict:
    """
    Retrieve top-k relevant documents for a query.

    Args:
        query: user query string
        top_k: number of documents to retrieve
        band_override: force retrieval from specific band (for Stage 3 correction)

    Returns dict with:
        - query: original query
        - routing: band, fk_grade, margin, confidence
        - retrieved: list of top-k documents with scores
        - metadata: for Stage 3 literacy correction
    """
    # Step 1 — Classify query literacy band
    routing = classify_query(query)
    band = band_override if band_override else routing['band']

    # Step 2 — Load band embeddings
    embeddings, doc_ids = get_band_embeddings(band)

    # Step 3 — Encode query (load encoder if not pre-loaded)
    if q_tokenizer is None or q_model is None:
        q_tokenizer, q_model = get_dpr_query_encoder()
    query_vec = encode_query(query, q_tokenizer, q_model)

    # Step 4 — Compute similarity
    scores = cosine_similarity_batch(query_vec, embeddings)

    # Step 5 — Get top-k indices
    top_k_actual = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:top_k_actual]

    # Step 6 — Load corpus for document text (cached at module level)
    corpus_indexed = _get_corpus()

    # Step 7 — Build results
    retrieved = []
    for rank, idx in enumerate(top_indices):
        doc_id = str(doc_ids[idx])
        score = float(scores[idx])
        try:
            row = corpus_indexed.loc[doc_id]
            retrieved.append({
                'rank': rank + 1,
                'record_id': doc_id,
                'score': round(score, 4),
                'source': str(row['source']),
                'literacy_band': str(row['literacy_band']),
                'fk_grade': float(row['fk_grade']) if pd.notna(row['fk_grade']) else None,
                'full_text': str(row['full_text'])[:500],
                'question': str(row.get('question', ''))[:200]
            })
        except KeyError:
            retrieved.append({
                'rank': rank + 1,
                'record_id': doc_id,
                'score': round(score, 4),
                'full_text': 'Document not found in corpus'
            })

    return {
        'query': query,
        'routing': {
            'band': band,
            'band_override': band_override,
            'fk_grade': routing.get('fk_grade'),
            'margin': routing.get('margin'),
            'confidence': routing.get('confidence', 'fk_based'),
            'note': routing.get('note', '')
        },
        'retrieved': retrieved,
        'metadata': {
            'top_k': top_k_actual,
            'corpus_band_size': len(embeddings),
            'stage3_note': (
                'Routing band is FK-based best-effort approximation. '
                'Stage 3 must apply literacy correction regardless of routing band. '
                'band_override parameter available for Stage 3 correction.'
            )
        }
    }


def run_retrieval_pipeline():
    """Validate retrieval pipeline on sample health queries."""
    print("ORACLE Phase 4 — Retrieval Pipeline")
    print("=" * 45)

    print("\n--- Pipeline Components ---")
    print("  1. literacy_classifier.classify_query() — FK-based band routing")
    print("  2. literacy_classifier.get_band_embeddings() — load band index")
    print("  3. dpr_encoder.encode_query() — DPR query encoding (768-dim)")
    print("  4. cosine_similarity_batch() — similarity search")
    print("  5. corpus lookup — retrieve document text + metadata")

    print("\n--- Loading Query Encoder ---")
    q_tokenizer, q_model = get_dpr_query_encoder()
    print("  DPR query encoder loaded ✓")

    print("\n--- Sample Retrievals ---")
    test_queries = [
        ("What medicine should I take for a headache?", "low", 3),
        ("How does insulin regulate blood sugar levels?", "medium", 3),
        ("What are the contraindications of metformin in CKD?", "high", 3),
    ]

    for query, expected_band, k in test_queries:
        print(f"\n  Query: '{query}'")
        result = retrieve(query, top_k=k, q_tokenizer=q_tokenizer, q_model=q_model)
        routing = result['routing']
        print(f"  Routing: band={routing['band']} (expected={expected_band}) "
              f"FK={routing['fk_grade']} margin={routing.get('margin', 'N/A')}")
        print(f"  Retrieved {len(result['retrieved'])} documents:")
        for doc in result['retrieved']:
            print(f"    [{doc['rank']}] score={doc['score']:.4f} "
                  f"source={doc.get('source','?')} "
                  f"band={doc.get('literacy_band','?')} "
                  f"text={doc['full_text'][:60]}...")

    print(f"\n--- Retrieval Pipeline complete ---")
    print(f"  retrieve() ready for Stage 3 health literacy adaptation")
    print(f"  routing metadata passed to Stage 3 for literacy correction")
    print(f"  band_override parameter available for Stage 3 correction")

    return result


if __name__ == "__main__":
    run_retrieval_pipeline()
