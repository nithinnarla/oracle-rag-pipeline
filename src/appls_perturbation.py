import sys
sys.path.insert(0, '/tmp/APPLS')
import pandas as pd
from perturbation.simplification import LexicalSimplification
from tqdm import tqdm

df = pd.read_csv('/tmp/APPLS/data/plaba_test_with_simple.csv')
batch = df['reference_text'].tolist()
batch_simple = df['simple_text'].tolist()
ids = df['id'].tolist()

print('Running simplification perturbation...')
template = LexicalSimplification()
output_ids, ref_texts, perturbed_texts = [], [], []
chunk_pcts, sent_pcts, word_pcts, tokens = [], [], [], []

for i in tqdm(range(len(batch))):
    try:
        perturbed_text, perturbed_chunk_pct, perturbed_sent_pct, perturbed_word_pct, perturbed_tokens = template.perturb_iteration(batch[i], batch_simple[i])
        for j in range(len(perturbed_tokens)):
            output_ids.append(ids[i])
            ref_texts.append(batch[i])
            perturbed_texts.append(perturbed_text[j])
            chunk_pcts.append(perturbed_chunk_pct[j])
            sent_pcts.append(perturbed_sent_pct[j])
            word_pcts.append(perturbed_word_pct[j])
            tokens.append(str(perturbed_tokens[j]))
    except Exception as e:
        print(f'Error on record {i}: {e}')
        continue

df_out = pd.DataFrame({
    'id': output_ids, 'reference_text': ref_texts, 'perturbed_text': perturbed_texts,
    'perturbed_chunk_percentage': chunk_pcts, 'perturbed_sentence_percentage': sent_pcts,
    'perturbed_word_percentage': word_pcts, 'perturbed_tokens': tokens
})
df_out.to_csv('/tmp/APPLS/data/simplification_plaba_test_perturbation.csv', index=False)
print(f'Simplification done: {len(df_out)} rows')
