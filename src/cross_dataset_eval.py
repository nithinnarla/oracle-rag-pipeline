"""
ORACLE Cross-Dataset Evaluation
Phase 4, Stage 4: Systematic evaluation across all corpus sources

Samples queries from each source dataset, runs the full literacy-conditioned
RAG pipeline (retrieve + generate), and measures generation quality per source.

Two conditions per query:
- actual: generation uses the classifier's routing_band (real pipeline behavior)
- upper_bound: for misrouted queries only, generation is re-run using the
  correct target_band, isolating how much of the FK/quality gap is caused
  by routing error versus generation quality itself

Metrics per query: FK grade, SMOG, ROUGE-L, BERTScore, FK reduction vs source.

Sources evaluated: medmcqa, medqa, mirage, plaba, pubmed, pubmedqa
Sample size: up to 5 queries per source per band

Decision 16: Cross-dataset evaluation design
"""
import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import textstat
from rouge_score import rouge_scorer as rouge_scorer_module
from bert_score import score as bert_score
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from retrieval_pipeline import retrieve
from generation_pipeline import generate_response, score_readability

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'oracle_corpus.csv')
OUTPUT_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'cross_dataset_results.csv')

SAMPLES_PER_SOURCE_PER_BAND = 5
RANDOM_SEED = 42

def compute_rouge_l(reference, hypothesis):
    scorer = rouge_scorer_module.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, hypothesis)['rougeL'].fmeasure

def run_generation(query, band, retrieved_docs, source, target_band, fk_source, answer, condition):
    gen_result = generate_response(query, band, retrieved_docs)
    generated = gen_result.get('text', '')
    error = gen_result.get('error', '')
    readability = score_readability(generated)
    fk_generated = readability.get('fk_grade')
    fk_reduction = (fk_source - fk_generated) if fk_source is not None and fk_generated is not None else None
    rouge_l = compute_rouge_l(answer, generated) if answer and generated else None
    tokens = gen_result.get('usage', {}).get('total_tokens', 0)
    return {
        'source': source,
        'target_band': target_band,
        'used_band': band,
        'band_match': band == target_band,
        'condition': condition,
        'query': query[:200],
        'generated_text': generated,
        'answer_text': answer[:500],
        'fk_source': fk_source,
        'fk_generated': fk_generated,
        'smog_generated': readability.get('smog_grade'),
        'fk_reduction': fk_reduction,
        'rouge_l': rouge_l,
        'tokens': tokens,
        'error': error
    }

def run_cross_dataset_eval():
    print("Loading corpus...")
    df = pd.read_csv(CORPUS_PATH)
    print(f"Corpus: {len(df)} records, {df['source'].nunique()} sources")

    sources = df['source'].unique()
    bands = ['low', 'medium', 'high', 'clinical']
    results = []

    for source in sources:
        source_df = df[df['source'] == source]
        print(f"\n--- Source: {source} ({len(source_df)} records) ---")

        for band in bands:
            band_df = source_df[source_df['literacy_band'] == band]
            if len(band_df) == 0:
                print(f"  Band {band}: no records, skipping")
                continue

            sample = band_df.sample(
                n=min(SAMPLES_PER_SOURCE_PER_BAND, len(band_df)),
                random_state=RANDOM_SEED
            )
            print(f"  Band {band}: {len(sample)} queries")

            for _, row in sample.iterrows():
                if source == 'plaba':
                    query = str(row['full_text']) if pd.notna(row['full_text']) else ''
                else:
                    query = str(row['question']) if pd.notna(row['question']) else ''
                if not query or len(query) < 10:
                    continue

                answer = str(row['answer']) if pd.notna(row['answer']) else ''
                fk_source = float(row['fk_grade']) if pd.notna(row['fk_grade']) else None

                try:
                    retrieval = retrieve(query, top_k=5)
                    routing_band = retrieval.get('routing', {}).get('band', band)
                    retrieved_docs = retrieval.get('retrieved', [])

                    actual = run_generation(query, routing_band, retrieved_docs, source, band, fk_source, answer, 'actual')
                    results.append(actual)
                    time.sleep(0.5)

                    if routing_band != band:
                        upper_bound = run_generation(query, band, retrieved_docs, source, band, fk_source, answer, 'upper_bound')
                        results.append(upper_bound)
                        time.sleep(0.5)

                except Exception as e:
                    print(f"    Error: {e}")
                    continue

    df_results = pd.DataFrame(results)

    print("\nComputing BERTScore for all generations (batched)...")
    valid = df_results[df_results['generated_text'].str.len() > 0].copy()
    P, R, F1 = bert_score(
        valid['generated_text'].tolist(),
        valid['answer_text'].tolist(),
        lang='en', verbose=False
    )
    valid['bertscore'] = F1.numpy()
    df_results = df_results.merge(
        valid[['bertscore']], left_index=True, right_index=True, how='left'
    )

    df_results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df_results)} results to {OUTPUT_PATH}")

    print("\n=== SUMMARY (condition=actual) ===")
    actual = df_results[df_results['condition'] == 'actual']
    summary = actual.groupby('source').agg(
        n=('query', 'count'),
        band_accuracy=('band_match', 'mean'),
        mean_fk_source=('fk_source', 'mean'),
        mean_fk_generated=('fk_generated', 'mean'),
        mean_fk_reduction=('fk_reduction', 'mean'),
        mean_rouge_l=('rouge_l', 'mean'),
        mean_bertscore=('bertscore', 'mean')
    ).round(3)
    print(summary.to_string())
    return df_results

if __name__ == '__main__':
    run_cross_dataset_eval()
