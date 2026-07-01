"""
ORACLE -- Text Preprocessor
Phase 4 -- Stage 1: Document Ingestion Pipeline

Preprocesses all 6 ORACLE datasets into a unified format for RAG ingestion.
Applies text cleaning, normalization, and literacy pre-scoring (Flesch-Kincaid).
Saves unified corpus to data/processed/oracle_corpus.csv for Stage 1 ingestion.

Six datasets:
- MedQA USMLE (11,451 records) -- clinical reasoning questions
- MedMCQA (182,822 records) -- medical entrance exam questions
- MedQuAD (16,407 RAG-usable records) -- patient-facing QA pairs
- MIRAGE (7,663 records) -- benchmark evaluation questions
- PLABA (921 records) -- plain language adaptations
- PubMedQA (1,000 labeled records) -- biomedical research QA
- PubMed abstracts (412 records) -- clinical professional abstracts
"""

import pandas as pd
import numpy as np
import sys
import os
import re
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import textstat

os.makedirs("data/processed", exist_ok=True)

ORACLE_CORPUS_PATH = "data/processed/oracle_corpus.csv"
MIN_TEXT_LENGTH = 20  # minimum characters to include a record


def clean_text(text: str) -> str:
    """Clean and normalize text for RAG ingestion."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    # Normalize quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Remove excessive punctuation
    text = re.sub(r"[.]{3,}", "...", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_literacy(text: str) -> dict:
    """Compute Flesch-Kincaid literacy scores."""
    if not text or len(text.split()) < 5:
        return {"fk_grade": np.nan, "fre_score": np.nan, "word_count": 0}
    try:
        return {
            "fk_grade": textstat.flesch_kincaid_grade(text),
            "fre_score": textstat.flesch_reading_ease(text),
            "word_count": len(text.split())
        }
    except Exception:
        return {"fk_grade": np.nan, "fre_score": np.nan, "word_count": len(text.split())}


def assign_literacy_band(fk_grade: float) -> str:
    """Assign literacy band based on FK grade level."""
    if np.isnan(fk_grade):
        return "unknown"
    if fk_grade <= 6:
        return "low"       # Grade 6 and below -- plain language
    elif fk_grade <= 10:
        return "medium"    # Grades 7-10 -- general public
    elif fk_grade <= 14:
        return "high"      # Grades 11-14 -- educated layperson
    else:
        return "clinical"  # Grade 15+ -- clinical professional


def process_medqa(record_id_offset: int = 0) -> pd.DataFrame:
    """Process MedQA USMLE dataset."""
    print("  Loading MedQA USMLE...")
    from medqa_loader import load_medqa_all
    d = load_medqa_all()
    # MedQA uses split structure: d['train']['data'], d['test']['data']
    dfs = []
    for split_name in ['train', 'test']:
        if split_name in d and 'data' in d[split_name]:
            split_df = d[split_name]['data'].copy()
            split_df['split'] = split_name
            dfs.append(split_df)
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    records = []
    for idx, row in df.iterrows():
        question = clean_text(str(row.get("question", "")))
        if len(question) < MIN_TEXT_LENGTH:
            continue
        # Build full text: question + answer options
        options = row.get("options", {})
        if isinstance(options, dict):
            options_text = " ".join([f"{k}: {v}" for k, v in options.items()])
        else:
            options_text = str(options)
        full_text = f"{question} {clean_text(options_text)}"
        scores = score_literacy(question)
        # answer column not in loader output -- use options text as retrieval content
        options = row.get("options", {})
        if isinstance(options, dict):
            answer_text = " ".join([f"{k}: {v}" for k, v in options.items()])
        else:
            answer_text = str(options)
        answer_text = clean_text(answer_text)
        records.append({
            "record_id": f"medqa_{record_id_offset + idx}",
            "source": "medqa",
            "source_type": "benchmark",
            "question": question,
            "answer": answer_text,
            "full_text": full_text.strip(),
            "split": str(row.get("split", "train")),
            "fk_grade": scores["fk_grade"],
            "fre_score": scores["fre_score"],
            "word_count": scores["word_count"],
            "literacy_band": assign_literacy_band(scores["fk_grade"]),
        })
    print(f"    MedQA: {len(records):,} records processed")
    return pd.DataFrame(records)


MEDMCQA_CAP = 20000  # capped to prevent single-source domination of retrieval corpus (was 189,366 = 89.9% of corpus)


def process_medmcqa(record_id_offset: int = 0) -> pd.DataFrame:
    """Process MedMCQA dataset -- capped and stratified by subject."""
    print("  Loading MedMCQA...")
    from medmcqa_loader import load_medmcqa_all
    d = load_medmcqa_all()
    # MedMCQA uses split structure: d['train']['data'], d['test']['data'], d['validation']['data']
    dfs = []
    for split_name in ['train', 'test', 'validation']:
        if split_name in d and 'data' in d[split_name]:
            split_df = d[split_name]['data'].copy()
            split_df['split'] = split_name
            dfs.append(split_df)
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    # Stratified cap by subject_name to preserve diversity across 21 medical subjects
    # while preventing MedMCQA from dominating the retrieval corpus
    if "subject_name" not in df.columns:
        print(f"    WARNING: subject_name column missing -- MedMCQA cap NOT applied, {len(df):,} records uncapped")
    if len(df) > MEDMCQA_CAP and "subject_name" in df.columns:
        frac = MEDMCQA_CAP / len(df)
        df = df.groupby("subject_name", group_keys=False).apply(
            lambda g: g.sample(frac=frac, random_state=42)
        ).reset_index(drop=True)
        print(f"    MedMCQA capped: {len(df):,} records (stratified by subject, {frac:.1%} of original)")
    records = []
    for idx, row in df.iterrows():
        question = clean_text(str(row.get("question", "")))
        if len(question) < MIN_TEXT_LENGTH:
            continue
        options_text = " ".join([
            clean_text(str(row.get("opa", ""))),
            clean_text(str(row.get("opb", ""))),
            clean_text(str(row.get("opc", ""))),
            clean_text(str(row.get("opd", "")))
        ])
        full_text = f"{question} {options_text}"
        explanation = clean_text(str(row.get("exp", "")))
        scores = score_literacy(question)
        # Use explanation as answer -- cop (correct option index) not in loader output
        explanation = clean_text(str(row.get("exp", "")))
        records.append({
            "record_id": f"medmcqa_{record_id_offset + idx}",
            "source": "medmcqa",
            "source_type": "benchmark",
            "subject": str(row.get("subject_name", "")),
            "question": question,
            "answer": explanation,
            "full_text": full_text.strip(),
            "split": str(row.get("split", "train")),
            "fk_grade": scores["fk_grade"],
            "fre_score": scores["fre_score"],
            "word_count": scores["word_count"],
            "literacy_band": assign_literacy_band(scores["fk_grade"]),
        })
    print(f"    MedMCQA: {len(records):,} records processed")
    return pd.DataFrame(records)


def process_medquad(record_id_offset: int = 0) -> pd.DataFrame:
    """Process MedQuAD dataset -- RAG-usable subset only."""
    print("  Loading MedQuAD...")
    from medquad_loader import load_medquad
    d = load_medquad()
    # MedQuAD has 'data' key directly
    df = d["data"] if isinstance(d, dict) and "data" in d else d
    # RAG-usable: records with non-null answers
    # MedQuAD has no answer column in HuggingFace version -- questions only
    records = []
    for idx, row in df.iterrows():
        question = clean_text(str(row.get("question", "")))
        answer = ""
        if len(question) < MIN_TEXT_LENGTH:
            continue
        full_text = question
        scores = score_literacy(question)
        records.append({
            "record_id": f"medquad_{record_id_offset + idx}",
            "source": "medquad",
            "source_type": "patient_qa",
            "question": question,
            "answer": answer,
            "full_text": full_text.strip(),
            "split": "all",
            "fk_grade": scores["fk_grade"],
            "fre_score": scores["fre_score"],
            "word_count": scores["word_count"],
            "literacy_band": assign_literacy_band(scores["fk_grade"]),
        })
    print(f"    MedQuAD: {len(records):,} RAG-usable records processed")
    return pd.DataFrame(records)


def process_mirage(record_id_offset: int = 0) -> pd.DataFrame:
    """Process MIRAGE benchmark dataset."""
    print("  Loading MIRAGE...")
    from mirage_loader import load_mirage
    d = load_mirage()
    # MIRAGE uses subsource structure: d['medqa']['data'], d['medmcqa']['data'] etc
    dfs = []
    for subsource in ['medqa', 'medmcqa', 'pubmedqa', 'bioasq', 'mmlu']:
        if subsource in d and 'data' in d[subsource]:
            sub_df = d[subsource]['data'].copy()
            sub_df['subsource'] = subsource
            dfs.append(sub_df)
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    records = []
    for idx, row in df.iterrows():
        question = clean_text(str(row.get("question", "")))
        if len(question) < MIN_TEXT_LENGTH:
            continue
        # Map answer letter to full option text for RAG retrieval quality
        answer_letter = str(row.get("answer", "")).strip()
        options = row.get("options", {})
        if isinstance(options, dict) and answer_letter in options:
            answer = clean_text(options[answer_letter])
        else:
            answer = clean_text(answer_letter)
        # Build full_text with all options for richer retrieval context
        if isinstance(options, dict):
            options_text = " ".join([f"{k}: {v}" for k, v in options.items()])
            full_text = f"{question} {clean_text(options_text)}"
        else:
            full_text = f"{question} {answer}" if answer else question
        scores = score_literacy(full_text)
        records.append({
            "record_id": f"mirage_{record_id_offset + idx}",
            "source": "mirage",
            "source_type": "benchmark",
            "question": question,
            "answer": answer,
            "full_text": full_text.strip(),
            "split": str(row.get("split", "all")),
            "fk_grade": scores["fk_grade"],
            "fre_score": scores["fre_score"],
            "word_count": scores["word_count"],
            "literacy_band": assign_literacy_band(scores["fk_grade"]),
        })
    print(f"    MIRAGE: {len(records):,} records processed")
    return pd.DataFrame(records)


def process_plaba(record_id_offset: int = 0) -> pd.DataFrame:
    """Process PLABA plain language adaptation dataset."""
    print("  Loading PLABA...")
    from plaba_loader import load_plaba_all
    d = load_plaba_all()
    records = []
    offset = 0
    for split_name in ["train", "val", "test"]:
        split_data = d[split_name]["data"]
        for idx, row in split_data.iterrows():
            text = clean_text(str(row.get("input_text", "")))
            if len(text) < MIN_TEXT_LENGTH:
                continue
            question = clean_text(str(row.get("question", "")))
            scores = score_literacy(text)
            records.append({
                "record_id": f"plaba_{record_id_offset + offset}",
                "source": "plaba",
                "source_type": "plain_language",
                "question": question,
                "answer": text,
                "full_text": f"{question} {text}".strip() if question else text,
                "split": split_name,
                "fk_grade": scores["fk_grade"],
                "fre_score": scores["fre_score"],
                "word_count": scores["word_count"],
                "literacy_band": assign_literacy_band(scores["fk_grade"]),
            })
            offset += 1
    print(f"    PLABA: {len(records):,} records processed")
    return pd.DataFrame(records)


def process_pubmedqa(record_id_offset: int = 0) -> pd.DataFrame:
    """Process PubMedQA labeled dataset."""
    print("  Loading PubMedQA...")
    from pubmedqa_loader import load_pubmedqa_labeled
    d = load_pubmedqa_labeled()
    # PubMedQA labeled has 'data' key directly
    df = d["data"] if isinstance(d, dict) and "data" in d else d
    records = []
    for idx, row in df.iterrows():
        question = clean_text(str(row.get("question", "")))
        answer = clean_text(str(row.get("long_answer", "")))
        if len(question) < MIN_TEXT_LENGTH:
            continue
        context = row.get("context", {})
        if isinstance(context, dict):
            context_text = " ".join([str(v) for v in context.values()])
        else:
            context_text = str(context)
        context_text = clean_text(context_text)
        full_text = f"{question} {context_text} {answer}".strip()
        scores = score_literacy(answer if answer else question)
        records.append({
            "record_id": f"pubmedqa_{record_id_offset + idx}",
            "source": "pubmedqa",
            "source_type": "research_qa",
            "question": question,
            "answer": answer,
            "full_text": full_text,
            "split": "labeled",
            "fk_grade": scores["fk_grade"],
            "fre_score": scores["fre_score"],
            "word_count": scores["word_count"],
            "literacy_band": assign_literacy_band(scores["fk_grade"]),
        })
    print(f"    PubMedQA: {len(records):,} records processed")
    return pd.DataFrame(records)


def process_pubmed_abstracts(record_id_offset: int = 0) -> pd.DataFrame:
    """Process PubMed abstracts from pubmed_api.py output."""
    print("  Loading PubMed abstracts...")
    path = "data/processed/pubmed_abstracts.csv"
    if not os.path.exists(path):
        print("    WARNING: pubmed_abstracts.csv not found -- skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    records = []
    for idx, row in df.iterrows():
        abstract = clean_text(str(row.get("abstract", "")))
        if len(abstract) < MIN_TEXT_LENGTH:
            continue
        title = clean_text(str(row.get("title", "")))
        full_text = f"{title} {abstract}".strip() if title else abstract
        scores = score_literacy(abstract)
        records.append({
            "record_id": f"pubmed_{record_id_offset + idx}",
            "source": "pubmed",
            "source_type": "clinical_abstract",
            "question": title,
            "answer": abstract,
            "full_text": full_text,
            "split": "all",
            "fk_grade": scores["fk_grade"],
            "fre_score": scores["fre_score"],
            "word_count": scores["word_count"],
            "literacy_band": assign_literacy_band(scores["fk_grade"]),
        })
    print(f"    PubMed abstracts: {len(records):,} records processed")
    return pd.DataFrame(records)


def run_preprocessor():
    print("ORACLE Phase 4 -- Stage 1: Text Preprocessor")
    print("=" * 48)

    print("\n--- Loading and Preprocessing Datasets ---")
    dfs = []
    offset = 0

    for name, processor in [
        ("MedQA", process_medqa),
        ("MedMCQA", process_medmcqa),
        # MedQuAD excluded from retrieval corpus -- HuggingFace version has no answers
        # Questions-only data not suitable for RAG retrieval documents
        # To be re-added tonight with full XML parser from abachaa/MedQuAD GitHub
        # ("MedQuAD", process_medquad),
        ("MIRAGE", process_mirage),
        ("PLABA", process_plaba),
        ("PubMedQA", process_pubmedqa),
        ("PubMed Abstracts", process_pubmed_abstracts),
    ]:
        df = processor(record_id_offset=offset)
        if len(df) > 0:
            dfs.append(df)
            offset += len(df)

    print("\n--- Combining Corpus ---")
    corpus = pd.concat(dfs, ignore_index=True)
    print(f"  Total records before quality filter: {len(corpus):,}")

    print("\n--- Quality Filtering ---")
    before = len(corpus)
    answer_str = corpus["answer"].astype(str).str.strip().str.lower()
    null_answer = corpus["answer"].isna() | (answer_str == "") | (answer_str == "nan") | (answer_str == "none")
    print(f"  Dropping {null_answer.sum():,} records with no answer content (unusable for RAG retrieval)")
    corpus = corpus[~null_answer].copy()

    unknown_fk = corpus["literacy_band"] == "unknown"
    print(f"  Dropping {unknown_fk.sum():,} records with unscoreable literacy (cannot assign PEFT adapter band)")
    corpus = corpus[~unknown_fk].copy()

    print(f"  Total records after quality filter: {len(corpus):,} (removed {before - len(corpus):,}, {(before-len(corpus))/before*100:.1f}%)")

    print("\n--- Corpus Statistics ---")
    print(f"  Sources: {corpus['source'].value_counts().to_dict()}")
    print(f"  Literacy bands: {corpus['literacy_band'].value_counts().to_dict()}")
    print(f"  FK grade mean: {corpus['fk_grade'].mean():.1f}")
    print(f"  FK grade by source:")
    for source, grp in corpus.groupby("source"):
        print(f"    {source}: FK={grp['fk_grade'].mean():.1f} | words={grp['word_count'].mean():.0f}")

    print("\n--- Quality Checks ---")
    empty_text = (corpus["full_text"].str.strip() == "") | corpus["full_text"].isna()
    short_text = corpus["full_text"].str.len() < MIN_TEXT_LENGTH
    print(f"  Empty full_text: {empty_text.sum()}")
    print(f"  Short full_text (<{MIN_TEXT_LENGTH} chars): {short_text.sum()}")
    print(f"  Duplicate record_ids: {corpus['record_id'].duplicated().sum()}")
    print(f"  Missing FK grades: {corpus['fk_grade'].isna().sum()}")

    print("\n--- Key Findings ---")
    clinical = (corpus["literacy_band"] == "clinical").sum()
    low = (corpus["literacy_band"] == "low").sum()
    print(f"  Clinical band (FK 15+): {clinical:,} records ({clinical/len(corpus)*100:.1f}%)")
    print(f"  Low band (FK <=6): {low:,} records ({low/len(corpus)*100:.1f}%)")
    print(f"  MedQA FK grade confirms USMLE clinical professional level")
    print(f"  MedQuAD patient-facing QA has lowest FK grade -- plain language target")
    print(f"  MedQuAD excluded from retrieval -- questions-only in HuggingFace version, to be re-added with XML parser")
    print(f"  PLABA plain language adaptations confirm literacy reduction pipeline")

    print("\n--- Saving Corpus to Disk ---")
    corpus.to_csv(ORACLE_CORPUS_PATH, index=False)
    print(f"  Saved {len(corpus):,} records to {ORACLE_CORPUS_PATH}")

    print("\n--- Text Preprocessor complete ---")
    print(f"  Unified ORACLE corpus ready for dpr_encoder.py -- Stage 1 ingestion")
    return corpus


if __name__ == "__main__":
    run_preprocessor()
