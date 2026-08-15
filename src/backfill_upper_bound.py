"""
Backfill upper_bound rows for misrouted queries in cross_dataset_results.csv.
General-purpose and idempotent, already_paired guards against re-backfilling
rows already done, so this is safe to rerun after any scale-up pass adds
new data.
"""
import os
import sys
import time
import warnings
import pandas as pd
from rouge_score import rouge_scorer as rouge_scorer_module
from bert_score import score as bert_score
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from retrieval_pipeline import retrieve
from generation_pipeline import generate_response, score_readability

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'cross_dataset_results.csv')

def compute_rouge_l(reference, hypothesis):
    scorer = rouge_scorer_module.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, hypothesis)['rougeL'].fmeasure

def run_backfill():
    df = pd.read_csv(RESULTS_PATH)
    actual = df[df['condition'] == 'actual']
    wrong = actual[actual['band_match'] == False].copy()
    upper = df[df['condition'] == 'upper_bound'].copy()

    already_paired = set(zip(upper['source'], upper['query']))
    to_backfill = wrong[~wrong.apply(lambda r: (r['source'], r['query']) in already_paired, axis=1)]

    print(f"Wrong-band rows total: {len(wrong)}")
    print(f"Already have upper_bound: {len(wrong) - len(to_backfill)}")
    print(f"Need backfill: {len(to_backfill)}")

    new_results = []
    for _, row in to_backfill.iterrows():
        source = row['source']
        target_band = row['target_band']
        query = row['query']
        answer = row['answer_text']
        fk_source = row['fk_source']

        try:
            retrieval = retrieve(query, top_k=5)
            retrieved_docs = retrieval.get('retrieved', [])

            gen_result = generate_response(query, target_band, retrieved_docs)
            generated = gen_result.get('text', '')
            readability = score_readability(generated)
            fk_generated = readability.get('fk_grade')
            fk_reduction = (fk_source - fk_generated) if pd.notna(fk_source) and fk_generated is not None else None
            rouge_l = compute_rouge_l(answer, generated) if answer and generated else None

            new_results.append({
                'source': source,
                'target_band': target_band,
                'used_band': target_band,
                'band_match': True,
                'condition': 'upper_bound',
                'query': query,
                'generated_text': generated,
                'answer_text': answer,
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
            print(f"Error on {source}/{target_band}: {e}")
            continue

    if not new_results:
        print("Nothing to backfill.")
        return

    new_df = pd.DataFrame(new_results)
    print(f"\nComputing BERTScore for {len(new_df)} backfilled rows...")
    valid = new_df[new_df['generated_text'].str.len() > 0].copy()
    P, R, F1 = bert_score(valid['generated_text'].tolist(), valid['answer_text'].tolist(), lang='en', verbose=False)
    valid['bertscore'] = F1.numpy()
    new_df = new_df.merge(valid[['bertscore']], left_index=True, right_index=True, how='left')

    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_csv(RESULTS_PATH, index=False)
    print(f"Appended {len(new_df)} upper_bound rows. New total: {len(combined)}")

if __name__ == '__main__':
    run_backfill()
