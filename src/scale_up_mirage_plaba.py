"""
ORACLE Cross-Dataset Scale-Up: mirage and plaba, n=25/band

Extends existing mirage/plaba pilot samples (n=5/band) toward n=25/band,
matching the scale-up already run for medmcqa/pubmedqa.

Appends to existing cross_dataset_results.csv, does not overwrite.
Excludes rows already sampled (condition='actual', these two sources),
so no duplicate queries. plaba has zero corpus records in the 'low'
literacy band -- that band is skipped for plaba, not padded or faked.
"""
import os
import sys
import time
import warnings
import pandas as pd
import textstat
from rouge_score import rouge_scorer as rouge_scorer_module
from bert_score import score as bert_score
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from retrieval_pipeline import retrieve
from generation_pipeline import generate_response, score_readability

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'oracle_corpus.csv')
RESULTS_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'cross_dataset_results.csv')

TARGET_N = 25
RANDOM_SEED = 42
SOURCES = ['mirage', 'plaba']
BANDS = ['low', 'medium', 'high', 'clinical']

def compute_rouge_l(reference, hypothesis):
    scorer = rouge_scorer_module.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, hypothesis)['rougeL'].fmeasure

def run_scale_up():
    print("Loading corpus and existing results...")
    corpus = pd.read_csv(CORPUS_PATH)
    existing = pd.read_csv(RESULTS_PATH)
    already_sampled_queries = set(
        existing[(existing['source'].isin(SOURCES)) & (existing['condition'] == 'actual')]['query']
    )
    print(f"Already sampled: {len(already_sampled_queries)} queries across {SOURCES}")

    new_results = []

    for source in SOURCES:
        source_df = corpus[corpus['source'] == source]
        print(f"\n--- Source: {source} ---")

        for band in BANDS:
            band_df = source_df[source_df['literacy_band'] == band]
            if len(band_df) == 0:
                print(f"  Band {band}: no records, skipping")
                continue

            n_target = min(TARGET_N, len(band_df))
            sample = band_df.sample(n=n_target, random_state=RANDOM_SEED)

            new_rows = []
            for _, row in sample.iterrows():
                if source == 'plaba':
                    query = str(row['full_text']) if pd.notna(row['full_text']) else ''
                else:
                    query = str(row['question']) if pd.notna(row['question']) else ''
                if not query or len(query) < 10:
                    continue
                query_truncated = query[:200]
                if query_truncated in already_sampled_queries:
                    continue
                new_rows.append(row)

            print(f"  Band {band}: {n_target} sampled, {len(new_rows)} new (not in pilot)")

            for row in new_rows:
                if source == 'plaba':
                    query = str(row['full_text']) if pd.notna(row['full_text']) else ''
                else:
                    query = str(row['question']) if pd.notna(row['question']) else ''
                answer = str(row['answer']) if pd.notna(row['answer']) else ''
                fk_source = float(row['fk_grade']) if pd.notna(row['fk_grade']) else None

                try:
                    retrieval = retrieve(query, top_k=5)
                    routing_band = retrieval.get('routing', {}).get('band', band)
                    retrieved_docs = retrieval.get('retrieved', [])

                    gen_result = generate_response(query, routing_band, retrieved_docs)
                    generated = gen_result.get('text', '')
                    readability = score_readability(generated)
                    fk_generated = readability.get('fk_grade')
                    fk_reduction = (fk_source - fk_generated) if fk_source is not None and fk_generated is not None else None
                    rouge_l = compute_rouge_l(answer, generated) if answer and generated else None

                    new_results.append({
                        'source': source,
                        'target_band': band,
                        'used_band': routing_band,
                        'band_match': band == routing_band,
                        'condition': 'actual',
                        'query': query[:200],
                        'generated_text': generated,
                        'answer_text': answer[:500],
                        'fk_source': fk_source,
                        'fk_generated': fk_generated,
                        'smog_generated': readability.get('smog_grade'),
                        'fk_reduction': fk_reduction,
                        'rouge_l': rouge_l,
                        'tokens': gen_result.get('usage', {}).get('total_tokens', 0),
                        'error': gen_result.get('error', '')
                    })
                    time.sleep(0.5)

                except Exception as e:
                    print(f"    Error: {e}")
                    continue

    if not new_results:
        print("\nNo new rows generated. Exiting.")
        return

    new_df = pd.DataFrame(new_results)
    print(f"\nComputing BERTScore for {len(new_df)} new generations...")
    valid = new_df[new_df['generated_text'].str.len() > 0].copy()
    P, R, F1 = bert_score(valid['generated_text'].tolist(), valid['answer_text'].tolist(), lang='en', verbose=False)
    valid['bertscore'] = F1.numpy()
    new_df = new_df.merge(valid[['bertscore']], left_index=True, right_index=True, how='left')

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(RESULTS_PATH, index=False)
    print(f"\nAppended {len(new_df)} new rows. Total dataset: {len(combined)} rows")

    print("\n=== NEW SUMMARY (mirage + plaba, actual condition) ===")
    actual = combined[(combined['source'].isin(SOURCES)) & (combined['condition'] == 'actual')]
    summary = actual.groupby('source').agg(
        n=('query', 'count'),
        band_accuracy=('band_match', 'mean'),
        mean_fk_reduction=('fk_reduction', 'mean'),
        mean_rouge_l=('rouge_l', 'mean')
    ).round(3)
    print(summary.to_string())

if __name__ == '__main__':
    run_scale_up()
