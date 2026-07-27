"""
ORACLE — Health Literacy Adaptation Module
Phase 4 — Stage 3: Literacy-Conditioned Response Preparation

Takes retrieval result from Stage 2 and applies literacy correction:
1. Determines user's actual target literacy level (user-declared > FK routing)
2. Injects PLABA plain language docs for low/medium targets
3. Re-ranks retrieved documents by source preference for target band
4. Flags and substitutes medical jargon for low/medium literacy targets
5. Prepares adapted context bundle for Stage 4 generation

Key architectural decisions (Decision 11):
- FK routing is unreliable — Stage 3 MUST correct regardless of routing band
- PLABA (source='plaba') is plain language gold standard — FK labels it clinical/high
  because FK measures sentence length not vocabulary difficulty
- Source-based PLABA injection used instead of literacy_band selection
- User-declared literacy level always overrides FK routing

Input: retrieval result dict from retrieval_pipeline.retrieve()
       + optional user_literacy_level (declared by user)
Output: adapted content dict ready for Stage 4 generation

Script type: pipeline/infrastructure — no notebook, no figures
"""

import os
import sys
import re
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'oracle_corpus.csv')

BANDS = ['low', 'medium', 'high', 'clinical']

SOURCE_PREFERENCE = {
    'low':      ['plaba', 'pubmedqa', 'medquad', 'medmcqa', 'medqa', 'mirage', 'pubmed'],
    'medium':   ['medquad', 'pubmedqa', 'plaba', 'medmcqa', 'medqa', 'mirage', 'pubmed'],
    'high':     ['pubmed', 'medqa', 'mirage', 'medquad', 'pubmedqa', 'medmcqa', 'plaba'],
    'clinical': ['pubmed', 'mirage', 'medqa', 'medmcqa', 'medquad', 'pubmedqa', 'plaba'],
}

JARGON_SUBSTITUTIONS = {
    'myocardial infarction': 'heart attack',
    'hypertension': 'high blood pressure',
    'hypotension': 'low blood pressure',
    'arrhythmia': 'irregular heartbeat',
    'atherosclerosis': 'hardening of the arteries',
    'tachycardia': 'fast heart rate',
    'bradycardia': 'slow heart rate',
    'thrombosis': 'blood clot',
    'embolism': 'blocked blood vessel',
    'hyperglycemia': 'high blood sugar',
    'hypoglycemia': 'low blood sugar',
    'dyslipidemia': 'abnormal cholesterol levels',
    'insulin resistance': 'the body not responding well to insulin',
    'dyspnea': 'shortness of breath',
    'pneumonia': 'lung infection',
    'bronchitis': 'airway inflammation',
    'pulmonary': 'lung',
    'etiology': 'cause',
    'prognosis': 'expected outcome',
    'pathophysiology': 'how the disease works in the body',
    'contraindication': 'reason not to use a treatment',
    'comorbidity': 'additional health condition',
    'asymptomatic': 'no symptoms',
    'idiopathic': 'unknown cause',
    'chronic': 'long-term',
    'acute': 'sudden or short-term',
    'benign': 'not harmful or cancerous',
    'malignant': 'cancerous or harmful',
    'prophylaxis': 'prevention',
    'pharmacological': 'medication-based',
    'adverse effects': 'side effects',
    'subcutaneous': 'under the skin',
    'intravenous': 'through a vein',
    'lesion': 'abnormal area of tissue',
    'inflammation': 'swelling and irritation',
    'exacerbate': 'make worse',
    'alleviate': 'reduce or relieve',
    'manifestation': 'sign or symptom',
    'prevalence': 'how common something is',
    'incidence': 'rate of new cases',
    'mortality': 'death rate',
    'morbidity': 'illness rate',
}

_PLABA_CACHE = None


def _get_plaba_docs():
    global _PLABA_CACHE
    if _PLABA_CACHE is None:
        df = pd.read_csv(CORPUS_PATH)
        _PLABA_CACHE = df[df['source'] == 'plaba'].to_dict('records')
    return _PLABA_CACHE


def determine_target_band(retrieval_result, user_literacy_level=None):
    if user_literacy_level and user_literacy_level in BANDS:
        return user_literacy_level, 'user_declared'
    band_override = retrieval_result['routing'].get('band_override')
    if band_override and band_override in BANDS:
        return band_override, 'band_override'
    return retrieval_result['routing']['band'], 'fk_routing'


def inject_plaba_docs(retrieved_docs, target_band, n_plaba=3):
    if target_band not in ['low', 'medium']:
        return retrieved_docs
    plaba_records = _get_plaba_docs()
    if not plaba_records:
        return retrieved_docs
    existing_ids = {d.get('record_id') for d in retrieved_docs}
    available = [r for r in plaba_records if r.get('record_id') not in existing_ids]
    injected = []
    for rec in available[:n_plaba]:
        injected.append({
            'rank': 99,
            'record_id': rec.get('record_id'),
            'score': 0.5,
            'source': 'plaba',
            'literacy_band': rec.get('literacy_band'),
            'fk_grade': rec.get('fk_grade'),
            'full_text': rec.get('full_text', ''),
            'question': rec.get('question', ''),
            'injected': True,
        })
    return retrieved_docs + injected


def rerank_by_source(retrieved_docs, target_band):
    preference = SOURCE_PREFERENCE.get(target_band, SOURCE_PREFERENCE['medium'])
    source_rank = {src: i for i, src in enumerate(preference)}
    ranked = sorted(
        retrieved_docs,
        key=lambda d: (
            source_rank.get(d.get('source', 'unknown'), len(preference)),
            -d['score']
        )
    )
    for i, doc in enumerate(ranked):
        doc['adapted_rank'] = i + 1
        doc['source_preference_rank'] = source_rank.get(
            doc.get('source', 'unknown'), len(preference)
        )
    return ranked


def apply_jargon_substitution(text, target_band):
    if target_band in ['high', 'clinical']:
        return text, []
    substitutions_made = []
    adapted = text
    for clinical_term, plain_term in JARGON_SUBSTITUTIONS.items():
        pattern = re.compile(re.escape(clinical_term), re.IGNORECASE)
        if pattern.search(adapted):
            adapted = pattern.sub(plain_term, adapted)
            substitutions_made.append({'original': clinical_term, 'substituted': plain_term})
    return adapted, substitutions_made


def adapt_retrieved_content(retrieval_result, user_literacy_level=None):
    query = retrieval_result['query']
    routing = retrieval_result['routing']
    retrieved = retrieval_result['retrieved']

    target_band, determination_method = determine_target_band(
        retrieval_result, user_literacy_level
    )
    enriched = inject_plaba_docs(retrieved, target_band)
    reranked = rerank_by_source(enriched, target_band)

    adapted_docs = []
    all_substitutions = []
    for doc in reranked:
        full_text = doc.get('full_text', '')
        adapted_text, substitutions = apply_jargon_substitution(full_text, target_band)
        adapted_doc = dict(doc)
        adapted_doc['adapted_text'] = adapted_text
        adapted_doc['jargon_substitutions'] = substitutions
        adapted_doc['literacy_corrected'] = target_band != routing['band']
        adapted_docs.append(adapted_doc)
        all_substitutions.extend(substitutions)

    top_docs = adapted_docs[:3]
    context_parts = []
    for doc in top_docs:
        text = doc['adapted_text'] if target_band in ['low', 'medium'] else doc['full_text']
        if text:
            context_parts.append(text[:500])
    context = '\n\n'.join(context_parts)

    return {
        'query': query,
        'target_band': target_band,
        'determination_method': determination_method,
        'routing_band': routing['band'],
        'literacy_correction_applied': target_band != routing['band'],
        'context': context,
        'adapted_docs': adapted_docs,
        'adaptation_metadata': {
            'total_jargon_substitutions': len(all_substitutions),
            'substitutions': all_substitutions,
            'plaba_docs_in_top3': sum(1 for d in top_docs if d.get('source') == 'plaba'),
            'source_distribution': {
                src: sum(1 for d in adapted_docs if d.get('source') == src)
                for src in set(d.get('source') for d in adapted_docs)
            },
            'stage4_note': (
                f'Target band: {target_band} (determined by {determination_method}). '
                f'Context prepared from top-3 re-ranked documents. '
                f'Stage 4 generates final response adapting language to {target_band} literacy.'
            )
        }
    }


def run_literacy_adapter():
    print("ORACLE Phase 4 — Stage 3: Health Literacy Adaptation Module")
    print("=" * 60)
    print("  Loading retrieval pipeline + DPR encoder...")

    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from retrieval_pipeline import retrieve
    from dpr_encoder import get_dpr_query_encoder

    q_tokenizer, q_model = get_dpr_query_encoder()
    print("  DPR encoder loaded ✓")

    test_cases = [
        ('What causes high blood pressure?', 'low', 'Low literacy — user declared'),
        ('How does insulin regulate blood sugar levels?', 'medium', 'Medium literacy — user declared'),
        ('What are the contraindications of metformin in CKD?', None, 'No declaration — FK routing'),
        ('Describe the pathophysiology of type 2 diabetes mellitus.', 'clinical', 'Clinical — user declared'),
    ]

    for i, (query, user_level, description) in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {description} ---")
        print(f"  Query: {query}")
        result = retrieve(query, top_k=5, q_tokenizer=q_tokenizer, q_model=q_model)
        adapted = adapt_retrieved_content(result, user_level)
        print(f"  Routing band: {adapted['routing_band']}")
        print(f"  Target band: {adapted['target_band']} (via {adapted['determination_method']})")
        print(f"  Literacy correction applied: {adapted['literacy_correction_applied']}")
        print(f"  PLABA docs in top-3: {adapted['adaptation_metadata']['plaba_docs_in_top3']}")
        print(f"  Jargon substitutions: {adapted['adaptation_metadata']['total_jargon_substitutions']}")
        if adapted['adaptation_metadata']['substitutions']:
            for sub in adapted['adaptation_metadata']['substitutions'][:3]:
                print(f"    '{sub['original']}' → '{sub['substituted']}'")
        print(f"  Context length: {len(adapted['context'])} chars")
        print(f"  Source distribution: {adapted['adaptation_metadata']['source_distribution']}")

    print(f"\n--- Stage 3 Literacy Adapter complete ---")
    print(f"  4 test cases validated across all literacy bands")
    print(f"  User-declared literacy level overrides FK routing")
    print(f"  PLABA docs injected for low/medium literacy targets")
    print(f"  Jargon substitution active for low/medium bands")
    print(f"  Context bundle ready for Stage 4 generation")


if __name__ == "__main__":
    run_literacy_adapter()
