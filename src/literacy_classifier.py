"""
ORACLE — Literacy Query Router
Phase 4 — Stage 2: Literacy-Conditioned Dense Retrieval

Routes user queries to the correct DPR embedding index based on
Flesch-Kincaid readability grade of the query text.

Architecture decision: Rule-based FK thresholds rather than ML classifier.
ML classifier trained on corpus documents achieved 100% accuracy due to
circular feature-label relationship — FK grade used both to create literacy
band labels and as the primary classifier feature. Rule-based routing is
more honest, interpretable, and generalizes correctly to user queries.

Band thresholds (from text_preprocessor.py):
- low:      FK ≤ 6  (plain language, patient-facing)
- medium:   FK 7-10 (general public)
- high:     FK 11-14 (health professional)
- clinical: FK 15+  (clinical professional)

Input: user query text (string)
Output: literacy band prediction + FK grade + routing to correct embeddings

See methodology_decisions.md Decision 11 for full reasoning.
"""

import numpy as np
import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import textstat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.path.join(REPO_ROOT, "data", "processed", "embeddings")

BAND_ORDER = ["low", "medium", "high", "clinical"]

# FK thresholds matching text_preprocessor.py
BAND_THRESHOLDS = {
    "low": (float('-inf'), 6),
    "medium": (6, 10),
    "high": (10, 14),
    "clinical": (14, float('inf'))
}


def classify_query(text: str) -> dict:
    """
    Classify a query text into literacy band using FK grade thresholds.
    Returns band, fk_grade, fre_score, word_count, and confidence proxy.
    """
    if not text or len(text.split()) < 3:
        return {
            'band': 'medium',
            'fk_grade': 10.0,
            'fre_score': 50.0,
            'word_count': len(text.split()) if text else 0,
            'confidence': 'low_word_count'
        }

    try:
        fk = textstat.flesch_kincaid_grade(text)
        fre = textstat.flesch_reading_ease(text)
        wc = len(text.split())
    except Exception:
        return {'band': 'medium', 'fk_grade': 10.0, 'fre_score': 50.0,
                'word_count': 0, 'confidence': 'error'}

    # Apply thresholds
    if fk <= 6:
        band = 'low'
    elif fk <= 10:
        band = 'medium'
    elif fk <= 14:
        band = 'high'
    else:
        band = 'clinical'

    # Distance from nearest threshold as confidence proxy
    if band == 'low':
        margin = 6 - fk
    elif band == 'medium':
        margin = min(fk - 6, 10 - fk)
    elif band == 'high':
        margin = min(fk - 10, 14 - fk)
    else:
        margin = fk - 14

    return {
        'band': band,
        'fk_grade': round(fk, 2),
        'fre_score': round(fre, 2),
        'word_count': wc,
        'margin': round(margin, 2)
    }


def get_band_embeddings(band: str) -> tuple:
    """
    Load DPR embeddings and IDs for the specified literacy band.
    Returns (embeddings, ids) as numpy arrays.
    """
    emb_path = os.path.join(EMBEDDINGS_DIR, f"embeddings_{band}.npy")
    ids_path = os.path.join(EMBEDDINGS_DIR, f"band_ids_{band}.npy")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Embeddings not found for band '{band}': {emb_path}\n"
            "Run dpr_encoder.py first."
        )

    embeddings = np.load(emb_path)
    ids = np.load(ids_path, allow_pickle=True)
    return embeddings, ids


def run_literacy_classifier():
    """Validate rule-based router on sample queries and corpus statistics."""
    print("ORACLE Phase 4 — Literacy Query Router (Rule-based FK Thresholds)")
    print("=" * 65)

    print("\n--- Architecture Decision ---")
    print("  Rule-based FK thresholds — not ML classifier")
    print("  Reason: ML classifier on corpus achieves 100% accuracy due to")
    print("  circular feature-label relationship (FK used to create labels)")
    print("  Rule-based routing is correct, interpretable, and generalizes")
    print("  See methodology_decisions.md Decision 11")

    print("\n--- Band Thresholds ---")
    for band, (lo, hi) in BAND_THRESHOLDS.items():
        lo_str = f"{lo}" if lo != float('-inf') else "-inf"
        hi_str = f"{hi}" if hi != float('inf') else "+inf"
        print(f"  {band:<10} FK grade: ({lo_str}, {hi_str}]")

    print("\n--- Corpus Band Distribution Verification ---")
    import pandas as pd
    corpus_path = os.path.join(REPO_ROOT, "data", "processed", "oracle_corpus.csv")
    df = pd.read_csv(corpus_path)
    print(df['literacy_band'].value_counts().to_string())
    print(f"  Total: {len(df):,} records")

    print("\n--- Embeddings Availability ---")
    for band in BAND_ORDER:
        emb_path = os.path.join(EMBEDDINGS_DIR, f"embeddings_{band}.npy")
        ids_path = os.path.join(EMBEDDINGS_DIR, f"band_ids_{band}.npy")
        if os.path.exists(emb_path):
            emb = np.load(emb_path)
            print(f"  {band:<10} embeddings: {emb.shape} ✓")
        else:
            print(f"  {band:<10} embeddings: MISSING — run dpr_encoder.py")

    print("\n--- Sample Query Classification ---")
    test_queries = [
        ("Take this pill with water every morning.", "low"),
        ("You should check your blood sugar levels daily.", "medium"),
        ("Metformin is first-line therapy for type 2 diabetes management.", "high"),
        ("The pathophysiology of T2DM involves insulin resistance and beta-cell dysfunction.", "clinical"),
        ("What causes high blood pressure?", "low"),
        ("How does chemotherapy affect the immune system?", "medium"),
        ("Describe the mechanism of ACE inhibitors in hypertension.", "high"),
        ("What is the role of the renin-angiotensin-aldosterone system in CKD progression?", "clinical"),
    ]

    correct = 0
    for query, expected in test_queries:
        result = classify_query(query)
        pred = result['band']
        match = "✓" if pred == expected else "✗"
        correct += (pred == expected)
        print(f"  {match} [{expected}→{pred}] FK={result['fk_grade']:5.1f} "
              f"margin={result.get('margin', 0):4.1f}: {query[:55]}...")

    print(f"\n  Sample accuracy: {correct}/{len(test_queries)} "
          f"({correct/len(test_queries):.1%})")

    print("\n--- FK Grade Distribution by Band ---")
    for band in BAND_ORDER:
        band_df = df[df['literacy_band'] == band]
        fk = band_df['fk_grade'].dropna()
        print(f"  {band:<10} n={len(band_df):,} "
              f"FK mean={fk.mean():.1f} "
              f"std={fk.std():.1f} "
              f"min={fk.min():.1f} "
              f"max={fk.max():.1f}")

    print(f"\n--- Literacy Router complete ---")
    print(f"  classify_query() ready for retrieval_pipeline.py")
    print(f"  get_band_embeddings() ready for retrieval_pipeline.py")
    print(f"  Rule-based routing: no model file needed")


if __name__ == "__main__":
    run_literacy_classifier()
