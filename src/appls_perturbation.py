import sys
sys.path.insert(0, '/tmp/APPLS')
import pandas as pd
from perturbation.informativeness import DeleteSentence
from perturbation.coherent import SentencesShuffle4Coherent
from tqdm import tqdm

df = pd.read_csv('/tmp/APPLS/data/plaba_test.csv')
batch = df['reference_text'].tolist()
ids = df['id'].tolist()

# --- delete_sentence (informativeness) ---
print('Running delete_sentence...')
template = DeleteSentence()
output_ids, ref_texts, perturbed_texts, sent_pcts, word_pcts, tokens = [], [], [], [], [], []
for i in tqdm(range(len(batch))):
    new_claims, perturb_sent_pct, perturb_word_pct, replaced_tokens = template.perturb_iteration(batch[i])
    for j in range(len(new_claims)):
        output_ids.append(ids[i])
        ref_texts.append(batch[i])
        perturbed_texts.append(new_claims[j])
        sent_pcts.append(perturb_sent_pct[j])
        word_pcts.append(perturb_word_pct[j])
        tokens.append(str(replaced_tokens[j]))
df_out = pd.DataFrame({'id': output_ids, 'reference_text': ref_texts, 'perturbed_text': perturbed_texts,
                        'perturbed_sentence_percentage': sent_pcts, 'perturbed_word_percentage': word_pcts,
                        'perturbed_tokens': tokens})
df_out.to_csv('/tmp/APPLS/data/delete_sentence_plaba_test_perturbation.csv', index=False)
print(f'delete_sentence done: {len(df_out)} rows')

# --- coherent (coherence) ---
print('Running coherent...')
template = SentencesShuffle4Coherent()
output_ids, ref_texts, perturbed_texts, dist_pcts, token_list = [], [], [], [], []
for i in tqdm(range(len(batch))):
    perturbed_text, dist_percent_list, iter_list = template.perturb_iteration(batch[i])
    if iter_list is None:
        output_ids.append(ids[i])
        ref_texts.append(batch[i])
        perturbed_texts.append(perturbed_text)
        dist_pcts.append(0.0)
        token_list.append(None)
    else:
        for j in range(len(perturbed_text)):
            output_ids.append(ids[i])
            ref_texts.append(batch[i])
            perturbed_texts.append(perturbed_text[j])
            dist_pcts.append(dist_percent_list[j])
            token_list.append(str(iter_list[j]))
df_out = pd.DataFrame({'id': output_ids, 'reference_text': ref_texts, 'perturbed_text': perturbed_texts,
                        'perturbed_percentage': dist_pcts, 'perturbed_tokens': token_list})
df_out.to_csv('/tmp/APPLS/data/coherent_plaba_test_perturbation.csv', index=False)
print(f'coherent done: {len(df_out)} rows')
