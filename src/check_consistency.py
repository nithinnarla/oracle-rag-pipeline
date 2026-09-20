"""
ORACLE, Paper and Repository Consistency Checks
Phase 5, pre-commit gate

Mirrors fape-fairness-ml/src/check_consistency.py for this repo. Every check
here exists because the corresponding defect was actually found in this paper
on Sep 17 2026, so each one is a regression test, not a hypothetical.

Run before every commit:
    python src/check_consistency.py

Exits non-zero if any check fails, so it can be wired into .git/hooks/pre-commit.
"""

import os
import re
import sys
import glob
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

# Latin-1/Extended-A range for surnames such as Oguz, written as escapes so that
# this file contains no non-ASCII bytes of its own (check 9 would flag them).
SURNAME = '[A-Z][A-Za-z\u00C0-\u024F\\-]+'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO_ROOT, 'docs', 'paper_draft.md')
CORPUS = os.path.join(REPO_ROOT, 'data', 'processed', 'oracle_corpus.csv')

failures = []      # correctness: a number, claim or reference is wrong. Blocks the commit.
pendings = []      # readiness: true but not yet submission-ready. Blocks submission only.
notes = []


def fail(check, msg):
    failures.append('%s: %s' % (check, msg))


def pending(check, msg):
    """For things that are accurate as they stand but must be resolved before the
    manuscript goes out: figure resolution, front matter, an undeposited dataset.
    Reported every run, and fatal only under --submission."""
    pendings.append('%s: %s' % (check, msg))


def note(msg):
    notes.append(msg)


def paper_text():
    with open(PAPER, encoding='utf-8') as fh:
        return fh.read()


def body_and_refs(text):
    parts = text.split('## References', 1)
    return parts[0], (parts[1] if len(parts) > 1 else '')


# --------------------------------------------------------------------------
# 1. corpus totals in Table 1 and Table 2 match the corpus file
# --------------------------------------------------------------------------
def check_corpus_tables(text):
    name = 'Check 1 (corpus tables)'
    if not os.path.exists(CORPUS):
        note('%s skipped, %s not present' % (name, os.path.relpath(CORPUS, REPO_ROOT)))
        return None
    df = pd.read_csv(CORPUS, usecols=['source', 'literacy_band'])
    total = len(df)

    for claimed in set(re.findall(r'([\d,]{6,})[- ]record corpus', text)) | \
                   set(re.findall(r'totaling ([\d,]{6,}) records', text)):
        if int(claimed.replace(',', '')) != total:
            fail(name, 'paper says %s records, corpus has %s' % (claimed, format(total, ',')))

    src_map = {'MedMCQA': 'medmcqa', 'MedQA': 'medqa', 'MIRAGE': 'mirage',
               'PubMedQA': 'pubmedqa', 'PLABA': 'plaba'}
    for label, key in src_map.items():
        m = re.search(r'\|\s*%s\s*\|\s*([\d,]+)\s*\|\s*([\d.]+)%%' % label, text)
        if not m:
            continue
        n_claim, p_claim = int(m.group(1).replace(',', '')), float(m.group(2))
        n_real = int((df.source == key).sum())
        p_real = 100.0 * n_real / total
        if n_claim != n_real:
            fail(name, 'Table 1 %s says %s, data has %s' % (label, m.group(1), format(n_real, ',')))
        if abs(p_claim - p_real) > 0.05:
            fail(name, 'Table 1 %s share says %.1f%%, data gives %.1f%%' % (label, p_claim, p_real))

    vc = df.literacy_band.value_counts()
    for label in ('Low', 'Medium', 'High', 'Clinical'):
        m = re.search(r'\|\s*%s\s*\|\s*([\d,]+)\s*\|\s*([\d.]+)%%' % label, text)
        if not m:
            continue
        n_claim, p_claim = int(m.group(1).replace(',', '')), float(m.group(2))
        n_real = int(vc.get(label.lower(), 0))
        p_real = 100.0 * n_real / total
        if n_claim != n_real:
            fail(name, 'Table 2 %s says %s, data has %s' % (label, m.group(1), format(n_real, ',')))
        if abs(p_claim - p_real) > 0.05:
            fail(name, 'Table 2 %s share says %.1f%%, data gives %.1f%%' % (label, p_claim, p_real))
    return df


# --------------------------------------------------------------------------
# 2. headline statistics reproduce from the committed result files
# --------------------------------------------------------------------------
def check_headline_stats(text):
    name = 'Check 2 (headline statistics)'
    path = os.path.join(REPO_ROOT, 'data', 'processed', 'cross_dataset_results.csv')
    if not os.path.exists(path):
        note('%s skipped, cross_dataset_results.csv not present' % name)
        return
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        note('%s skipped, scipy not installed' % name)
        return

    df = pd.read_csv(path)
    wrong = df[(df.condition == 'actual') & (df.band_match == False)]
    upper = df[df.condition == 'upper_bound']
    paired = wrong.set_index(['source', 'query']).join(
        upper.set_index(['source', 'query']), lsuffix='_wrong', rsuffix='_upper', how='inner')

    for col, claim_n, claim_p in (('fk_reduction', 177, 0.0018),
                                  ('rouge_l', 180, 0.640),
                                  ('bertscore', 180, 0.937)):
        a = paired[col + '_wrong'].dropna()
        b = paired[col + '_upper'].dropna()
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        if len(a) != claim_n:
            fail(name, '%s paired n is %d, paper says %d' % (col, len(a), claim_n))
        _, p = wilcoxon(a, b)
        tol = 0.0005 if claim_p < 0.01 else 0.002
        if abs(p - claim_p) > tol:
            fail(name, '%s Wilcoxon p is %.4f, paper says %.4f' % (col, p, claim_p))


# --------------------------------------------------------------------------
# 3. per-source routing accuracy and flip rates match the ablation file
# --------------------------------------------------------------------------
def check_routing_and_flips(text):
    name = 'Check 3 (routing and flip rates)'
    path = os.path.join(REPO_ROOT, 'docs', 'fk_ablation_results.csv')
    if not os.path.exists(path):
        note('%s skipped, fk_ablation_results.csv not present' % name)
        return
    f = pd.read_csv(path)
    # the paper reports these with derived-query sources excluded, so the check must too
    from query_eligibility import exclude_derived_queries
    f, _ = exclude_derived_queries(f)

    m = re.search(r'On a ([\d,]+)-query evaluation set', text)
    if m and int(m.group(1).replace(',', '')) != len(f):
        fail(name, 'paper says a %s-query set, file has %d rows' % (m.group(1), len(f)))

    truth = f.band_match.astype(str).str.lower() == 'true'
    for label, key, claim in (('MedQA', 'medqa', 73.0), ('PubMedQA', 'pubmedqa', 44.7),
                              ('PubMed', 'pubmed', 42.7), ('Mirage', 'mirage', 32.0)):
        sub = truth[f.source == key]
        if not len(sub):
            continue
        got = 100.0 * sub.mean()
        if abs(got - claim) > 0.05:
            fail(name, '%s routing is %.1f%%, paper says %.1f%%' % (label, got, claim))

    flip = f.band_match.astype(str).str.lower() != f.band_match_question_only.astype(str).str.lower()
    m = re.search(r'changes routing correctness for ([\d.]+)% of queries overall', text)
    if m and abs(100.0 * flip.mean() - float(m.group(1))) > 0.05:
        fail(name, 'overall flip rate is %.1f%%, paper says %s%%' % (100.0 * flip.mean(), m.group(1)))


# --------------------------------------------------------------------------
# 4. every embedded figure exists, is cited, and is numbered in reading order
# --------------------------------------------------------------------------
def check_figures(text):
    name = 'Check 4 (figures)'
    embeds = re.findall(r'!\[.*?\]\(\.\./(.*?)\)', text)
    for rel in embeds:
        if not os.path.exists(os.path.join(REPO_ROOT, rel)):
            fail(name, 'embedded figure missing from disk: %s' % rel)

    captions = [int(n) for n in re.findall(r'^\*\*Figure (\d+)\.\*\*', text, re.M)]
    if captions != sorted(captions):
        fail(name, 'figure captions are out of order: %s' % captions)
    if captions and captions != list(range(1, len(captions) + 1)):
        fail(name, 'figure numbering is not 1..N with no gaps: %s' % captions)
    if len(captions) != len(embeds):
        fail(name, '%d embedded images but %d captions' % (len(embeds), len(captions)))

    body, _ = body_and_refs(text)
    for n in captions:
        # a caption alone is not a citation; the number must also appear in prose
        refs = len(re.findall(r'Figure %d\b' % n, body))
        if refs < 2:
            fail(name, 'Figure %d is captioned but never referred to in the text' % n)


# --------------------------------------------------------------------------
# 5. every embedded figure is drawn by a script, and is not older than it
# --------------------------------------------------------------------------
def check_figure_provenance(text):
    name = 'Check 5 (figure provenance)'
    srcs = {p: open(p, encoding='utf-8', errors='ignore').read()
            for p in glob.glob(os.path.join(REPO_ROOT, 'src', '*.py'))}
    for rel in re.findall(r'!\[.*?\]\(\.\./(.*?)\)', text):
        base = os.path.basename(rel)
        drawn = [p for p, t in srcs.items() if base in t]
        full = os.path.join(REPO_ROOT, rel)
        if not drawn:
            pending(name, '%s is embedded in the paper but no script in src/ draws it' % base)
            continue
        if len(drawn) > 1:
            pending(name, '%s is drawn by %d scripts: %s' %
                 (base, len(drawn), ', '.join(os.path.basename(d) for d in drawn)))
        if os.path.exists(full) and os.path.getmtime(full) < os.path.getmtime(drawn[0]):
            pending(name, '%s is older than %s, regenerate it' %
                 (base, os.path.basename(drawn[0])))


# --------------------------------------------------------------------------
# 6. a figure's caption must not contradict the title drawn onto it
# --------------------------------------------------------------------------
def check_caption_against_plot_title(text):
    """Figures 1 and 2 were retrieval-eval plots captioned as corpus statistics.
    The plot title lives in the script's set_title call, so compare the two."""
    name = 'Check 6 (caption vs plot title)'
    srcs = {p: open(p, encoding='utf-8', errors='ignore').read()
            for p in glob.glob(os.path.join(REPO_ROOT, 'src', '*.py'))}
    blocks = re.findall(r'!\[.*?\]\(\.\./(.*?)\)\s*\n\s*\n\*\*Figure (\d+)\.\*\*\s*(.*?)(?:\n\n|\Z)',
                        text, re.S)
    # words that mean different populations; a caption claiming one while the
    # plot claims the other is the defect this check exists for
    CONFLICTS = [({'corpus record', 'all corpus', 'corpus records'}, {'retrieved document'}),
                 ({'retrieved document'}, {'corpus record', 'all corpus'})]
    for rel, num, caption in blocks:
        base = os.path.basename(rel)
        titles = []
        for p, t in srcs.items():
            if base not in t:
                continue
            titles += re.findall(r"set_title\(\s*'(.*?)'", t, re.S)
            titles += re.findall(r"set_title\(\s*\"(.*?)\"", t, re.S)
            titles += re.findall(r"suptitle\(\s*\n?\s*f?'(.*?)'", t, re.S)
        blob = ' '.join(titles).lower().replace('\\n', ' ')
        cap = caption.lower()
        if not blob:
            continue
        for cap_terms, title_terms in CONFLICTS:
            if any(c in cap for c in cap_terms) and any(tt in blob for tt in title_terms):
                fail(name, 'Figure %s caption says %r but the plotted title says %r'
                     % (num,
                        next(c for c in cap_terms if c in cap),
                        next(tt for tt in title_terms if tt in blob)))


# --------------------------------------------------------------------------
# 7. citations: nothing cited that is unlisted, nothing listed that is uncited
# --------------------------------------------------------------------------
def check_citations(text):
    name = 'Check 7 (citations)'
    body, refs = body_and_refs(text)
    entries = [l.strip() for l in refs.split('\n') if l.strip() and not l.startswith('#')]

    seen = {}
    for e in entries:
        a = re.match('(%s)' % SURNAME, e)
        y = re.search(r'\((\d{4}[a-z]?)\)', e)
        if not (a and y):
            continue
        author, year = a.group(1), y.group(1)
        # every reference must be cited somewhere in the body
        hit = re.search(re.escape(author) + r'[^\n]{0,80}?' + re.escape(year), body) or \
              re.search(r'\b' + re.escape(author) + r'\b[^\n]{0,20}\(' + re.escape(year), body)
        if not hit:
            fail(name, 'reference never cited in the text: %s (%s)' % (author, year))
        # duplicate author+bare-year pairs must be disambiguated with a/b
        base_year = year[:4]
        key = (author, base_year)
        seen.setdefault(key, []).append(year)

    for (author, base_year), years in seen.items():
        if len(years) > 1 and any(y == base_year for y in years):
            fail(name, '%d references share (%s, %s) without a/b suffixes'
                 % (len(years), author, base_year))

    listed = {(re.match('(%s)' % SURNAME, e).group(1),
               re.search(r'\((\d{4}[a-z]?)\)', e).group(1))
              for e in entries
              if re.match('(%s)' % SURNAME, e) and re.search(r'\((\d{4}[a-z]?)\)', e)}
    for author, year in set(re.findall(
            '(%s)' % SURNAME + r'\s+(?:et al\.?|and\s+[A-Z][A-Za-z\-]+)\s*\((\d{4}[a-z]?)\)', body)):
        if (author, year) not in listed:
            fail(name, 'cited in text but not in the reference list: %s et al. (%s)' % (author, year))


# --------------------------------------------------------------------------
# 8. every "Section X.Y" pointer resolves to a real heading
# --------------------------------------------------------------------------
def check_cross_references(text):
    name = 'Check 8 (cross-references)'
    heads = set(re.findall(r'^#{2,3} (\d+(?:\.\d+)?)', text, re.M))
    for ref in set(re.findall(r'Section (\d+(?:\.\d+)?)', text)):
        if ref not in heads:
            fail(name, 'Section %s is referenced but no such heading exists' % ref)


# --------------------------------------------------------------------------
# 9. no AI-typography markers anywhere in the tracked prose
# --------------------------------------------------------------------------
def check_ai_typography():
    name = 'Check 9 (prose typography)'
    # written as escapes so this file holds no non-ASCII bytes of its own
    marks = {'em dash': '\u2014', 'en dash': '\u2013',
             'non-breaking space': '\u00a0', 'curly apostrophe': '\u2019',
             'curly open quote': '\u201c', 'curly close quote': '\u201d',
             'ellipsis character': '\u2026', 'zero-width space': '\u200b',
             'rightwards arrow': '\u2192'}
    # Not flagged, and deliberately so: box-drawing characters used in the
    # citation-chain diagrams, mathematical symbols that are clearer than their
    # ASCII equivalents, and diacritics in author surnames such as Oguz,
    # Kuttler and Rocktaschel, which must be preserved exactly.
    # requirements.txt and .gitignore were missed by an earlier version of this
    # check and each carried em dashes in its comments. Data files are excluded
    # deliberately: the CSVs hold model-generated text, and editing a dash out
    # of a recorded generation would falsify the data.
    targets = (glob.glob(os.path.join(REPO_ROOT, 'docs', '*.md')) +
               [os.path.join(REPO_ROOT, 'README.md'),
                os.path.join(REPO_ROOT, 'requirements.txt'),
                os.path.join(REPO_ROOT, '.gitignore')] +
               glob.glob(os.path.join(REPO_ROOT, 'src', '*.py')) +
               glob.glob(os.path.join(REPO_ROOT, 'notebooks', '*.ipynb')))
    for f in targets:
        if not os.path.exists(f):
            continue
        if os.path.abspath(f) == os.path.abspath(__file__):
            continue  # this file stores the marker characters as literals
        t = open(f, encoding='utf-8', errors='ignore').read()
        if f.endswith('.ipynb'):
            try:
                nb = json.loads(t)
                t = '\n'.join(''.join(c.get('source', [])) for c in nb.get('cells', []))
            except ValueError:
                pass
        for label, ch in marks.items():
            if ch in t:
                fail(name, '%s contains %s (%d)' %
                     (os.path.relpath(f, REPO_ROOT), label, t.count(ch)))


# --------------------------------------------------------------------------
# 10. requirements.txt covers every third-party import in src/
# --------------------------------------------------------------------------
def check_requirements_cover_imports():
    name = 'Check 10 (requirements coverage)'
    req_path = os.path.join(REPO_ROOT, 'requirements.txt')
    if not os.path.exists(req_path):
        fail(name, 'requirements.txt is missing')
        return
    req = open(req_path, encoding='utf-8').read().lower()
    local = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(REPO_ROOT, 'src', '*.py'))}
    stdlib = set(getattr(sys, 'stdlib_module_names', ()))
    alias = {'sklearn': 'scikit-learn', 'dotenv': 'python-dotenv', 'bert_score': 'bert-score',
             'rouge_score': 'rouge-score', 'sentence_transformers': 'sentence-transformers',
             'rank_bm25': 'rank-bm25', 'PIL': 'pillow', 'yaml': 'pyyaml'}
    # imported from the external APPLS clone documented in requirements.txt, not on PyPI
    external = {'perturbation'}
    # PIL is used by this checker only; it is pinned in requirements.txt as pillow
    for p in glob.glob(os.path.join(REPO_ROOT, 'src', '*.py')):
        for line in open(p, encoding='utf-8', errors='ignore'):
            m = re.match(r'\s*(?:from|import)\s+([A-Za-z_][\w]*)', line)
            if not m:
                continue
            mod = m.group(1)
            if mod in stdlib or mod in local or mod in external:
                continue
            # skip prose inside docstrings, e.g. "from provided PubMed abstracts"
            if not re.match(r'\s*(?:from|import)\s+[A-Za-z_][\w]*\s*(?:import\s+[\w,*\s()]+)?\s*(?:#.*)?$', line):
                continue
            name_pypi = alias.get(mod, mod).lower()
            if name_pypi not in req and mod.lower() not in req:
                fail(name, '%s imports %s but requirements.txt does not list it'
                     % (os.path.basename(p), mod))


# --------------------------------------------------------------------------
# 11. package versions quoted in the paper match requirements.txt
# --------------------------------------------------------------------------
def check_versions_match_requirements(text):
    name = 'Check 11 (declared versions)'
    req_path = os.path.join(REPO_ROOT, 'requirements.txt')
    if not os.path.exists(req_path):
        return
    pinned = dict(re.findall(r'^([A-Za-z0-9_.\-]+)==([\d.]+)', open(req_path, encoding='utf-8').read(), re.M))
    pinned = {k.lower(): v for k, v in pinned.items()}
    spoken = {'transformers': r'transformers \(([\d.]+)\)',
              'torch': r'torch \(([\d.]+)\)',
              'openai': r'openai ([\d.]+)\)',
              'rouge-score': r'rouge_score \(([\d.]+)\)',
              'bert-score': r'bert-score \(([\d.]+)\)',
              'peft': r'peft\s+library \(([\d.]+)\)'}
    for pkg, pat in spoken.items():
        m = re.search(pat, text)
        if not m:
            continue
        said = m.group(1)
        have = pinned.get(pkg)
        if have and said != have:
            fail(name, 'paper says %s %s, requirements.txt pins %s' % (pkg, said, have))


# --------------------------------------------------------------------------
# 12. data the paper claims is committed must actually be tracked by git
# --------------------------------------------------------------------------
def check_claimed_data_is_tracked(text):
    name = 'Check 12 (data availability claim)'
    if 'are committed alongside the scripts' not in text and \
       'committed alongside the scripts that produced them' not in text:
        return
    import subprocess
    # Section 4.2 names these as committed; the corpus is deliberately excluded
    # there and rebuilt from the loaders instead, so it is not required here
    required = ['data/processed/cross_dataset_results.csv',
                'data/processed/factual_consistency_results.csv',
                'data/processed/self_retrieval_results.csv',
                'data/processed/self_retrieval_controlled_results.csv',
                'data/appls/appls_oracle_results.csv',
                'docs/fk_ablation_results.csv']
    for rel in required:
        r = subprocess.run(['git', '-C', REPO_ROOT, 'ls-files', '--error-unmatch', rel],
                           capture_output=True)
        if r.returncode != 0:
            fail(name, 'Section 4.2 claims the evaluation outputs are committed, '
                       'but %s is untracked' % rel)


# --------------------------------------------------------------------------
# 13. embedded figures must be at submission resolution
# --------------------------------------------------------------------------
def check_figure_dpi(text):
    name = 'Check 13 (figure resolution)'
    try:
        from PIL import Image
    except ImportError:
        note('%s skipped, Pillow not installed' % name)
        return
    for rel in re.findall(r'!\[.*?\]\(\.\./(.*?)\)', text):
        full = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(full):
            continue
        dpi = Image.open(full).info.get('dpi', (0, 0))[0]
        if dpi < 299.5:
            pending(name, '%s is %g dpi, submission needs 300 or more'
                 % (os.path.basename(rel), round(dpi, 1)))


# --------------------------------------------------------------------------
# 14. no savefig in src/ may write below 300 dpi
# --------------------------------------------------------------------------
def check_savefig_dpi():
    name = 'Check 14 (savefig dpi)'
    for p in glob.glob(os.path.join(REPO_ROOT, 'src', '*.py')):
        t = open(p, encoding='utf-8', errors='ignore').read()
        for m in re.finditer(r'dpi\s*=\s*(\d+)', t):
            if int(m.group(1)) < 300:
                fail(name, '%s writes figures at dpi=%s' % (os.path.basename(p), m.group(1)))


# --------------------------------------------------------------------------
# 15. front and back matter required before submission
# --------------------------------------------------------------------------
def check_front_back_matter(text):
    name = 'Check 15 (front and back matter)'
    first = text.split('\n', 1)[0]
    if 'Paper Draft' in first or first.strip() in ('# ORACLE', '#'):
        pending(name, 'the title is still a placeholder: %r' % first.strip())
    required = {'Keywords': r'(?im)^\s*\*{0,2}Keywords',
                'Highlights': r'(?im)^#{0,3}\s*\*{0,2}Highlights',
                'Declarations': r'(?im)declaration of (?:competing )?interest|^#{0,3}\s*Declarations',
                'Data availability statement': r'(?im)^#{0,3}\s*\*{0,2}Data availability',
                'author block': r'(?im)^\s*(?:Nithin|\*\*Nithin)'}
    for label, pat in required.items():
        if not re.search(pat, text):
            pending(name, 'missing before submission: %s' % label)



# --------------------------------------------------------------------------
# 16. a claimed external deposit must actually cite a resolvable identifier
# --------------------------------------------------------------------------
def check_deposit_claim(text):
    """Section 4.2 and the data availability statement say the corpus is
    deposited separately. Until a DOI or URL is present that is a promise, not
    a fact, and it must not reach a reviewer unsupported."""
    name = 'Check 16 (deposit claim)'
    if 'deposited separately' not in text:
        return
    if not re.search(r'(10\.\d{4,9}/\S+|zenodo\.org/\S+|figshare\.com/\S+|dataverse\S*/\S+)', text):
        pending(name, 'the paper says the corpus is "deposited separately" but cites no '
                   'DOI or repository URL; deposit it and add the identifier, or reword the claim')


# --------------------------------------------------------------------------
# 17. the 20-query routing diagnostic must match its recorded source
# --------------------------------------------------------------------------
def check_handwritten_routing(text):
    """Section 5.2 reports the hand-written 20-query routing counts. Their source
    is retrieval_eval_routing.csv once the evaluation has been re-run, and until
    then the stored output of notebooks/retrieval_evaluation.ipynb."""
    name = 'Check 17 (20-query routing diagnostic)'
    m = re.search(r'low (\d+) of (\d+), medium (\d+) of (\d+), high (\d+) of (\d+), clinical (\d+) of (\d+)', text)
    if not m:
        return
    claimed = {'low': (int(m.group(1)), int(m.group(2))),
               'medium': (int(m.group(3)), int(m.group(4))),
               'high': (int(m.group(5)), int(m.group(6))),
               'clinical': (int(m.group(7)), int(m.group(8)))}

    actual = {}
    csv_path = os.path.join(REPO_ROOT, 'data', 'processed', 'retrieval_eval_routing.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        actual = {r['band']: (int(r['correct']), int(r['total'])) for _, r in df.iterrows()}
        source = 'retrieval_eval_routing.csv'
    else:
        nb_path = os.path.join(REPO_ROOT, 'notebooks', 'retrieval_evaluation.ipynb')
        if not os.path.exists(nb_path):
            note('%s skipped, neither the CSV nor the notebook is present' % name)
            return
        with open(nb_path, encoding='utf-8') as fh:
            nb = json.load(fh)
        blob = ''
        for cell in nb.get('cells', []):
            for out in cell.get('outputs', []):
                t = out.get('text', '')
                blob += ''.join(t) if isinstance(t, list) else str(t)
        for band, correct, total in re.findall(
                r'(low|medium|high|clinical)\s+(\d+)/(\d+)\s+\(', blob):
            actual[band] = (int(correct), int(total))
        source = 'notebooks/retrieval_evaluation.ipynb output'

    if not actual:
        note('%s skipped, no routing counts found in %s' % (name, source))
        return
    for band, (c, t) in claimed.items():
        if band not in actual:
            fail(name, 'paper reports %s but %s has no value for it' % (band, source))
        elif actual[band] != (c, t):
            fail(name, '%s is %d of %d in %s, paper says %d of %d'
                 % (band, actual[band][0], actual[band][1], source, c, t))


# --------------------------------------------------------------------------
# 18. the embedding index must describe the corpus the paper reports on
# --------------------------------------------------------------------------
def check_embedding_index(text):
    """The per-band embedding files were built before PubMed was dropped, so they
    held 412 vectors the corpus no longer contains, and Figure 5 quoted Table 2
    counts as if they were the deployed pools. Whenever the two disagree the
    paper must say so explicitly."""
    name = 'Check 18 (embedding index vs corpus)'
    emb_dir = os.path.join(REPO_ROOT, 'data', 'processed', 'embeddings')
    if not os.path.exists(emb_dir) or not os.path.exists(CORPUS):
        note('%s skipped, embeddings or corpus not present' % name)
        return
    try:
        import numpy as np
    except ImportError:
        note('%s skipped, numpy unavailable' % name)
        return

    counts = pd.read_csv(CORPUS, usecols=['literacy_band']).literacy_band.value_counts()
    drift = {}
    for band in ('low', 'medium', 'high', 'clinical'):
        f = os.path.join(emb_dir, 'embeddings_%s.npy' % band)
        if not os.path.exists(f):
            continue
        n_emb = int(np.load(f, mmap_mode='r').shape[0])
        n_corpus = int(counts.get(band, 0))
        if n_emb != n_corpus:
            drift[band] = (n_emb, n_corpus)

    if not drift:
        return
    # drift is tolerated only while the paper discloses it
    disclosed = ('embedding index was built before' in text
                 or 'taken from the embedding index' in text)
    if not disclosed:
        fail(name, 'the embedding index disagrees with the corpus (%s) and the paper '
                   'does not disclose it' %
                   ', '.join('%s %d vs %d' % (b, a, c) for b, (a, c) in sorted(drift.items())))
        return
    # and only while Figure 5 quotes the index rather than Table 2
    for band, (n_emb, _n) in drift.items():
        if format(n_emb, ',') not in text:
            fail(name, 'the %s band pool is %s in the index but that figure does not '
                       'appear in the paper' % (band, format(n_emb, ',')))


# --------------------------------------------------------------------------
# 19. the target journal's own submission requirements
# --------------------------------------------------------------------------
def check_journal_requirements(text):
    """Journal of Biomedical Informatics asks for a structured abstract of at
    most 300 words under Objective/Methods/Results/Conclusion, a body of at most
    6,000 words, a CRediT contribution statement, a data availability statement,
    and a graphical abstract supplied as a separate file."""
    name = 'Check 19 (JBI requirements)'
    lines = text.split('\n')

    # body: Section 1 through the end of Section 7
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith('## 1. '))
        end = next(i for i, l in enumerate(lines) if l.startswith('## Declarations'))
    except StopIteration:
        return
    body = [l for l in lines[start:end]
            if not l.startswith('![') and not l.startswith('**Figure')
            and not l.startswith('**Table')]
    n_body = len(' '.join(body).split())
    if n_body > 6000:
        pending(name, 'body is %d words, JBI allows 6,000' % n_body)

    # structured abstract
    try:
        ai = next(i for i, l in enumerate(lines) if l.startswith('## Abstract'))
        abstract = next(l for l in lines[ai + 1:ai + 6] if l.strip())
    except StopIteration:
        abstract = ''
    if abstract:
        n_abs = len(abstract.split())
        if n_abs > 300:
            pending(name, 'abstract is %d words, JBI allows 300' % n_abs)
        missing = [h for h in ('Objective', 'Methods', 'Results', 'Conclusion')
                   if ('**%s' % h) not in abstract and ('%s:' % h) not in abstract]
        if missing:
            pending(name, 'abstract is unstructured, JBI wants %s headings'
                    % '/'.join(missing))

    if 'CRediT' not in text:
        pending(name, 'no CRediT author contribution statement')
    if not re.search(r'(?im)^\*\*Data availability', text):
        pending(name, 'no data availability statement')

    # graphical abstract, supplied separately, at least 1328 x 531 pixels
    ga = [f for f in glob.glob(os.path.join(REPO_ROOT, 'figures', '*graphical*'))
          + glob.glob(os.path.join(REPO_ROOT, 'figures', '**', '*graphical*'), recursive=True)]
    if not ga:
        pending(name, 'no graphical abstract found in figures/, JBI expects one as a '
                      'separate file of at least 1328 x 531 pixels')
    else:
        try:
            from PIL import Image
            for f in ga:
                w, h = Image.open(f).size
                if w < 1328 or h < 531:
                    pending(name, '%s is %dx%d, JBI wants at least 1328x531'
                            % (os.path.basename(f), w, h))
        except ImportError:
            pass

# --------------------------------------------------------------------------
# 20. a notebook that displays a figure must show that figure's current bytes
# --------------------------------------------------------------------------
def check_notebook_figure_outputs():
    """A cell whose source is Image('../figures/x.png') stores a copy of that
    file in its output. Redrawing the figure does not touch the stored copy, so
    the notebook goes on displaying the old image from the same path. Seven
    outputs had drifted this way across two notebooks, five of them surviving
    the September dpi sweep unnoticed, which is exactly the kind of staleness
    nobody thinks to look for."""
    name = 'Check 20 (notebook figure outputs)'
    nb_dir = os.path.join(REPO_ROOT, 'notebooks')
    if not os.path.isdir(nb_dir):
        note('%s skipped, no notebooks directory' % name)
        return
    import base64
    import hashlib
    import json as _json
    pattern = re.compile(r"""Image\(['"]\.\./(figures/[^'"]+)['"]\)""")
    for fn in sorted(os.listdir(nb_dir)):
        if not fn.endswith('.ipynb'):
            continue
        try:
            nb = _json.load(open(os.path.join(nb_dir, fn), encoding='utf-8'))
        except ValueError:
            fail(name, '%s is not valid JSON' % fn)
            continue
        for i, cell in enumerate(nb.get('cells', [])):
            m = pattern.search(''.join(cell.get('source', [])))
            if not m:
                continue
            target = os.path.join(REPO_ROOT, m.group(1))
            if not os.path.exists(target):
                fail(name, '%s cell %d displays %s, which does not exist'
                     % (fn, i, m.group(1)))
                continue
            current = hashlib.md5(open(target, 'rb').read()).hexdigest()
            for out in cell.get('outputs', []):
                for mime, payload in (out.get('data') or {}).items():
                    if not mime.startswith('image/png'):
                        continue
                    raw = base64.b64decode(
                        payload if isinstance(payload, str) else ''.join(payload))
                    if hashlib.md5(raw).hexdigest() != current:
                        fail(name, '%s cell %d shows a stale copy of %s, '
                                   'rerun the cell or refresh its output'
                             % (fn, i, os.path.basename(m.group(1))))


def main():
    if not os.path.exists(PAPER):
        print('paper not found at %s' % PAPER)
        return 1
    text = paper_text()

    check_corpus_tables(text)
    check_headline_stats(text)
    check_routing_and_flips(text)
    check_figures(text)
    check_figure_provenance(text)
    check_caption_against_plot_title(text)
    check_citations(text)
    check_cross_references(text)
    check_ai_typography()
    check_requirements_cover_imports()
    check_versions_match_requirements(text)
    check_claimed_data_is_tracked(text)
    check_figure_dpi(text)
    check_savefig_dpi()
    check_front_back_matter(text)
    check_deposit_claim(text)
    check_handwritten_routing(text)
    check_embedding_index(text)
    check_journal_requirements(text)
    check_notebook_figure_outputs()

    strict = '--submission' in sys.argv

    print('ORACLE consistency checks')
    print('=' * 62)
    for n in notes:
        print('  note: %s' % n)

    if failures:
        print('\n%d correctness problem(s), these block the commit:\n' % len(failures))
        for f in failures:
            print('  - %s' % f)
    if pendings:
        print('\n%d item(s) pending before submission:\n' % len(pendings))
        for f in pendings:
            print('  - %s' % f)

    if failures:
        print('\nFAILED')
        return 1
    if pendings and strict:
        print('\nNOT SUBMISSION READY')
        return 1
    if pendings:
        print('\n20 checks: no correctness problems. %d item(s) still pending '
              'before submission, listed above.' % len(pendings))
        print('Run with --submission to treat those as fatal.')
        return 0
    print('\n20 checks passed, and nothing pending')
    return 0


if __name__ == '__main__':
    sys.exit(main())
