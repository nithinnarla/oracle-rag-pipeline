"""
ORACLE, Self-Retrieval Query Eligibility
Phase 4, Stage 2 correction, Sep 17 2026

The self-retrieval design uses each record's own question as its query. That
requires the record to have a question. PLABA does not: it is an abstract to
plain-language adaptation corpus, and its question column carries the health
topic index (75 distinct integer values, printed as "75 health topics" by
plaba_loader.py), not a natural-language question.

Because the column is non-empty, PLABA's 921 records passed the emptiness
guard in the self-retrieval scripts and were queried on strings like "1" and
"61". Their Precision@1 came out at 0.051 as-deployed and 0.077 pool-size-
controlled, against roughly 0.71 and 0.76 for every other source. That is a
property of querying a topic index, not a measurement of retrieval quality,
and because PLABA sits only in the clinical and high bands it depressed those
two bands while leaving the low band untouched.

The criterion applied here is the one the corpus build already declares in
Section 3.2: a minimum of 20 characters of question text. Checked against the
corpus, that threshold selects exactly the 921 PLABA records and no record
from any other source, so it removes the topic-index queries without
discarding a single usable one.

Re-querying PLABA on its answer or full_text instead is not a valid
alternative: document embeddings are built from full_text, so the query would
equal the document and Precision@1 would approach 1.0 by construction.

The results CSVs keep every row the run produced. This filter is applied at
analysis time, so the raw record stays intact and the exclusion is auditable.
"""

import os

import pandas as pd

# Section 3.2's stated minimum question length for corpus inclusion
MIN_QUERY_CHARS = 20


def filter_eligible(results_df, corpus_path):
    """Drop rows whose query text is too short to be a question.

    Returns (eligible, dropped). Falls back to returning everything unchanged
    if the corpus file is unavailable, so figure regeneration never silently
    produces a differently-filtered result than the eval run.
    """
    if not os.path.exists(corpus_path):
        raise IOError('corpus needed for query-eligibility filtering: %s' % corpus_path)

    corpus = pd.read_csv(corpus_path, usecols=['record_id', 'question'])
    corpus['query_chars'] = corpus.question.astype(str).str.strip().str.len()

    merged = results_df.merge(corpus[['record_id', 'query_chars']],
                              on='record_id', how='left')
    if merged.query_chars.isna().any():
        missing = int(merged.query_chars.isna().sum())
        raise ValueError('%d result rows have no matching corpus record' % missing)

    keep = merged.query_chars >= MIN_QUERY_CHARS
    eligible = merged[keep].drop(columns=['query_chars']).reset_index(drop=True)
    dropped = merged[~keep].drop(columns=['query_chars']).reset_index(drop=True)
    return eligible, dropped


def report_exclusions(eligible, dropped, label=''):
    total = len(eligible) + len(dropped)
    print('\n--- Query eligibility (Section 3.2, minimum %d characters of question text) ---'
          % MIN_QUERY_CHARS)
    if not len(dropped):
        print('  all %d queries eligible%s' % (total, label))
        return
    by_source = dropped.source.value_counts().to_dict() if 'source' in dropped else {}
    by_band = dropped.literacy_band.value_counts().to_dict() if 'literacy_band' in dropped else {}
    print('  excluded %d of %d queries%s' % (len(dropped), total, label))
    print('    by source: %s' % by_source)
    print('    by band:   %s' % by_band)
    print('  reason: question field holds a topic index, not a question '
          '(see module docstring)')


def band_significance(results_df, band_col='literacy_band', metric='precision_at_1'):
    """Fisher exact tests for the low band against the other bands.

    The paper described the low band's advantage on means alone. These tests
    are what let that advantage be stated, or not stated, as significant.
    """
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        return {}

    low = results_df[results_df[band_col] == 'low'][metric]
    others = results_df[results_df[band_col] != 'low']
    if not len(low) or not len(others):
        return {}

    out = {}

    def _test(a, b):
        table = [[int(a.sum()), int(len(a) - a.sum())],
                 [int(b.sum()), int(len(b) - b.sum())]]
        return fisher_exact(table)[1]

    out['low_vs_all_others'] = {
        'low_mean': float(low.mean()), 'low_n': int(len(low)),
        'other_mean': float(others[metric].mean()), 'other_n': int(len(others)),
        'p': float(_test(low, others[metric])),
    }

    means = others.groupby(band_col)[metric].mean()
    best = means.idxmax()
    comp = others[others[band_col] == best][metric]
    out['low_vs_best_other'] = {
        'band': best, 'mean': float(comp.mean()), 'n': int(len(comp)),
        'p': float(_test(low, comp)),
    }
    return out


def print_significance(stats):
    if not stats:
        print('  (scipy unavailable, significance not computed)')
        return
    a = stats['low_vs_all_others']
    b = stats['low_vs_best_other']
    print('\n--- Low band vs the rest, Fisher exact on Precision@1 ---')
    print('  low %.3f (n=%d) vs all other bands %.3f (n=%d):  p = %.5f  %s'
          % (a['low_mean'], a['low_n'], a['other_mean'], a['other_n'], a['p'],
             'significant' if a['p'] < 0.05 else 'NOT significant'))
    print('  low %.3f vs %s, the strongest other band %.3f (n=%d):  p = %.5f  %s'
          % (a['low_mean'], b['band'], b['mean'], b['n'], b['p'],
             'significant' if b['p'] < 0.05 else 'NOT significant'))


# --------------------------------------------------------------------------
# Sources whose evaluation query is derived from the text that defines their
# band, which makes any band-agreement measure on them circular.
# --------------------------------------------------------------------------
# cross_dataset_eval.py substitutes full_text as PLABA's query, because PLABA
# has no question (see above). Two consequences follow for the FK ablation:
#
#   band_match compares the band routed from that query against target_band,
#   which was itself derived from the same full_text. PLABA therefore scores
#   100% routing accuracy by construction, not as a measurement.
#
#   band_question_only scores the query column, which for PLABA is full_text
#   truncated to 200 characters. All 72 PLABA rows sit exactly at that cap and
#   every one is a full_text prefix, so its "question-only versus full-text"
#   flip rate compares the same text at two lengths rather than a question
#   against a question plus answer.
#
# Both are excluded rather than reported with a caveat, since neither number
# is measuring what the surrounding comparison measures. Every other source's
# value is unaffected by the exclusion.
DERIVED_QUERY_SOURCES = ('plaba',)


def exclude_derived_queries(df, source_col='source'):
    """Split off rows whose query is derived from their own band-defining text.

    Returns (kept, dropped).
    """
    is_derived = df[source_col].astype(str).str.lower().isin(DERIVED_QUERY_SOURCES)
    return (df[~is_derived].reset_index(drop=True),
            df[is_derived].reset_index(drop=True))
