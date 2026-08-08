"""
ORACLE, DPR Encoder
Phase 4, Stage 1: Dense Passage Retrieval Encoding

Encodes ORACLE retrieval corpus using Dense Passage Retrieval (DPR).
Produces document embeddings per literacy band for literacy-conditioned retrieval.

Architecture:
- Context encoder: facebook/dpr-ctx_encoder-single-nq-base
- Query encoder: facebook/dpr-question_encoder-single-nq-base
- Embeddings saved as numpy arrays per literacy band

Input: data/processed/oracle_corpus.csv (37,076 records)
Output: data/processed/embeddings/
  - corpus_embeddings.npy, full corpus embeddings (37076, 768)
  - corpus_ids.npy, record_id alignment
  - embeddings_low.npy, low literacy band embeddings
  - embeddings_medium.npy, medium literacy band embeddings
  - embeddings_high.npy, high literacy band embeddings
  - band_ids_low.npy / band_ids_medium.npy / band_ids_high.npy

Note: DPR context encoder produces 768-dim embeddings.
Note: Full corpus encoding takes 10-20 minutes on CPU, run once and cache.
Note: Literacy band embeddings are subsets of full corpus for per-band retrieval.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# Path setup
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(REPO_ROOT, "data", "processed", "oracle_corpus.csv")
EMBEDDINGS_DIR = os.path.join(REPO_ROOT, "data", "processed", "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


def load_corpus() -> pd.DataFrame:
    """Load ORACLE corpus from preprocessed CSV."""
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(
            f"Corpus not found at {CORPUS_PATH}. "
            "Run text_preprocessor.py first."
        )
    df = pd.read_csv(CORPUS_PATH)
    df["full_text"] = df["full_text"].fillna("").astype(str)
    df = df[df["full_text"].str.len() > 10].reset_index(drop=True)
    return df


def get_dpr_context_encoder():
    """Load DPR context encoder and tokenizer."""
    try:
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
        print("  Loading DPR context encoder: facebook/dpr-ctx_encoder-single-nq-base")
        tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        model = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        model.eval()
        return tokenizer, model
    except ImportError:
        raise ImportError(
            "transformers library required. "
            "Install with: pip install transformers torch"
        )


def get_dpr_query_encoder():
    """Load DPR query encoder and tokenizer."""
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
    print("  Loading DPR query encoder: facebook/dpr-question_encoder-single-nq-base")
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    model = DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    model.eval()
    return tokenizer, model


def encode_texts(texts: list, tokenizer, model, batch_size: int = 32) -> np.ndarray:
    """
    Encode list of texts using DPR context encoder.
    Returns numpy array of shape (n_texts, 768).
    """
    import torch
    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1

        if batch_num % 50 == 0 or batch_num == 1:
            print(f"    Encoding batch {batch_num}/{n_batches} "
                  f"({min(i + batch_size, len(texts))}/{len(texts)} texts)")

        inputs = tokenizer(
            batch,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.pooler_output.numpy()

        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


def encode_query(query: str, tokenizer, model) -> np.ndarray:
    """
    Encode a single query using DPR query encoder.
    Returns numpy array of shape (768,).
    """
    import torch
    inputs = tokenizer(
        query,
        max_length=256,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.pooler_output.numpy()[0]
    return embedding


def run_dpr_encoder():
    """Main encoding pipeline."""
    print("ORACLE Phase 4, DPR Encoder")
    print("=" * 50)

    print("\n--- Loading Corpus ---")
    df = load_corpus()
    print(f"  Corpus: {len(df):,} records")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Literacy band distribution:")
    print(df["literacy_band"].value_counts().to_string(header=False))
    print(f"  Sources:")
    print(df["source"].value_counts().to_string(header=False))

    print("\n--- Loading DPR Encoders ---")
    ctx_tokenizer, ctx_model = get_dpr_context_encoder()
    q_tokenizer, q_model = get_dpr_query_encoder()
    print("  Context encoder loaded ")
    print("  Query encoder loaded ")

    print("\n--- Encoding Full Corpus ---")
    print(f"  Encoding {len(df):,} documents (batch_size=32)...")
    print(f"  Expected time: 10-20 minutes on CPU")
    texts = df["full_text"].tolist()
    corpus_embeddings = encode_texts(texts, ctx_tokenizer, ctx_model, batch_size=32)
    corpus_ids = df["record_id"].values
    print(f"  Corpus embeddings shape: {corpus_embeddings.shape}")

    print("\n--- Saving Full Corpus Embeddings ---")
    np.save(os.path.join(EMBEDDINGS_DIR, "corpus_embeddings.npy"), corpus_embeddings)
    np.save(os.path.join(EMBEDDINGS_DIR, "corpus_ids.npy"), corpus_ids)
    print(f"  Saved corpus_embeddings.npy {corpus_embeddings.shape}")
    print(f"  Saved corpus_ids.npy {corpus_ids.shape}")

    print("\n--- Encoding Per Literacy Band ---")
    for band in ["low", "medium", "high", "clinical"]:
        band_mask = df["literacy_band"] == band
        band_df = df[band_mask].reset_index(drop=True)
        if len(band_df) == 0:
            print(f"  {band}: 0 records, skipping")
            continue
        print(f"  Encoding {band} band: {len(band_df):,} records")
        band_texts = band_df["full_text"].tolist()
        band_embeddings = encode_texts(band_texts, ctx_tokenizer, ctx_model, batch_size=32)
        band_ids = band_df["record_id"].values
        np.save(os.path.join(EMBEDDINGS_DIR, f"embeddings_{band}.npy"), band_embeddings)
        np.save(os.path.join(EMBEDDINGS_DIR, f"band_ids_{band}.npy"), band_ids)
        print(f"  Saved embeddings_{band}.npy {band_embeddings.shape}")
        print(f"  Saved band_ids_{band}.npy {band_ids.shape}")

    print("\n--- Verifying Embedding Quality ---")
    # Spot check: encode a sample query and compute similarity
    sample_query = "What is the recommended treatment for urinary tract infection in pregnancy?"
    q_embedding = encode_query(sample_query, q_tokenizer, q_model)
    similarities = np.dot(corpus_embeddings, q_embedding)
    top_idx = np.argsort(similarities)[::-1][:3]
    print(f"  Sample query: '{sample_query[:60]}...'")
    print(f"  Top 3 retrieved documents:")
    for rank, idx in enumerate(top_idx):
        print(f"    {rank+1}. [{df.iloc[idx]['source']}] "
              f"{df.iloc[idx]['full_text'][:80]}...")

    print("\n--- DPR Encoder complete ---")
    print(f"  Full corpus: {corpus_embeddings.shape} embeddings saved")
    print(f"  Per-band embeddings saved for low/medium/high literacy")
    print(f"  Embeddings dir: {EMBEDDINGS_DIR}")
    print(f"  Ready for literacy_classifier.py and retrieval_pipeline.py")

    return corpus_embeddings, corpus_ids


if __name__ == "__main__":
    run_dpr_encoder()
