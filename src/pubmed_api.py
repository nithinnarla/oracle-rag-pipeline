"""
ORACLE, PubMed API Integration
Phase 4, Stage 1: Document Ingestion Pipeline

Fetches biomedical abstracts from NCBI E-utilities API for ORACLE retrieval corpus.
Uses PubMedQA PMIDs as seed set, then expands via MeSH term queries.
Rate-limited to NCBI guidelines: 10 requests/second with API key, 3/second without.

NCBI E-utilities endpoints:
- esearch: search PubMed by query, returns PMIDs
- efetch: fetch abstract text for given PMIDs
- elink: find related articles for seed PMIDs

Output: abstracts with readability scores for Stage 1 literacy-conditioned ingestion.
"""

import requests
import time
import json
import os
import sys
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# NCBI E-utilities base URL
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_EMAIL = "nithincf20013@gmail.com"
NCBI_TOOL = "oracle-rag-pipeline"

# Rate limiting: 3 requests/second without API key
REQUEST_DELAY = 0.34
MAX_RETRIES = 3
BATCH_SIZE = 200


def esearch(query, db="pubmed", retmax=1000, retstart=0):
    """Search PubMed and return list of PMIDs."""
    params = {
        "db": db,
        "term": query,
        "retmax": retmax,
        "retstart": retstart,
        "retmode": "json",
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL,
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            total = int(data.get("esearchresult", {}).get("count", 0))
            time.sleep(REQUEST_DELAY)
            return pmids, total
        except Exception as e:
            print(f"  esearch attempt {attempt+1} failed: {e}")
            time.sleep(REQUEST_DELAY * 3)
    return [], 0


def efetch_abstracts(pmids, db="pubmed"):
    """Fetch abstracts for a list of PMIDs. Returns list of dicts."""
    if not pmids:
        return []
    pmid_str = ",".join(str(p) for p in pmids)
    params = {
        "db": db,
        "id": pmid_str,
        "rettype": "abstract",
        "retmode": "xml",
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL,
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=60)
            response.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return parse_xml_abstracts(response.text, pmids)
        except Exception as e:
            print(f"  efetch attempt {attempt+1} failed: {e}")
            time.sleep(REQUEST_DELAY * 3)
    return []


def parse_xml_abstracts(xml_text, pmids):
    """Parse PubMed XML response and extract abstracts."""
    import xml.etree.ElementTree as ET
    results = []
    try:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            try:
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else None
                title_el = article.find(".//ArticleTitle")
                title = "".join(title_el.itertext()) if title_el is not None else ""
                abstract_els = article.findall(".//AbstractText")
                abstract = " ".join("".join(el.itertext()) for el in abstract_els)
                journal_el = article.find(".//Journal/Title")
                journal = journal_el.text if journal_el is not None else ""
                year_el = article.find(".//PubDate/Year")
                year = year_el.text if year_el is not None else ""
                mesh_terms = [
                    el.find("DescriptorName").text
                    for el in article.findall(".//MeshHeading")
                    if el.find("DescriptorName") is not None
                ]
                if pmid and abstract:
                    results.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "journal": journal,
                        "year": year,
                        "mesh_terms": "; ".join(mesh_terms),
                        "abstract_length": len(abstract),
                        "word_count": len(abstract.split()),
                    })
            except Exception:
                continue
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
    return results


def fetch_pmids_batch(pmids, desc=""):
    """Fetch abstracts for PMIDs in batches."""
    all_results = []
    batches = [pmids[i:i+BATCH_SIZE] for i in range(0, len(pmids), BATCH_SIZE)]
    for i, batch in enumerate(batches):
        print(f"  Fetching batch {i+1}/{len(batches)} ({len(batch)} PMIDs) {desc}")
        results = efetch_abstracts(batch)
        all_results.extend(results)
        print(f"    Retrieved {len(results)} abstracts")
    return all_results


def run_pipeline():
    print("ORACLE Phase 4 - Stage 1: PubMed API Integration")
    print("=" * 52)

    print(f"\n--- Loading PubMedQA Seed PMIDs ---")
    from pubmedqa_loader import load_pubmedqa_labeled
    labeled = load_pubmedqa_labeled()
    labeled_df = labeled["data"]
    seed_pmids = labeled_df["pubid"].astype(str).tolist()
    print(f"  Seed PMIDs from PubMedQA labeled: {len(seed_pmids)}")

    print(f"\n--- Verifying NCBI API Connectivity ---")
    test_pmids, total = esearch("health literacy patient education", retmax=5)
    if test_pmids:
        print(f"  NCBI API: Connected - test query returned {len(test_pmids)} PMIDs")
    else:
        print(f"  NCBI API: Connection failed or rate limited - check network")
        print(f"  Note: Running in offline mode - using PubMedQA embedded contexts only")
        return

    print(f"\n--- Fetching Seed PMID Abstracts ---")
    seed_results = fetch_pmids_batch(seed_pmids[:200], desc="(PubMedQA seed set)")
    print(f"  Fetched {len(seed_results)} abstracts from seed PMIDs")

    print(f"\n--- MeSH Query Expansion ---")
    mesh_queries = [
        ("health literacy patient education plain language", 200),
        ("biomedical question answering clinical NLP", 200),
        ("medical information retrieval patient comprehension", 200),
        ("discharge summary patient education readability", 100),
        ("health information seeking behavior low literacy", 100),
    ]
    expanded_pmids = set(seed_pmids)
    expansion_results = []
    for query, retmax in mesh_queries:
        pmids, total = esearch(query, retmax=retmax)
        new_pmids = [p for p in pmids if p not in expanded_pmids]
        expanded_pmids.update(new_pmids)
        print(f"  Query: {query[:50]}...")
        print(f"    Total results: {total:,} | New PMIDs: {len(new_pmids)}")
        if new_pmids:
            results = fetch_pmids_batch(new_pmids[:50], desc=f"({query[:30]})")
            expansion_results.extend(results)

    print(f"\n--- Corpus Statistics ---")
    all_results = seed_results + expansion_results
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.drop_duplicates(subset=["pmid"])
        print(f"  Total unique abstracts: {len(df):,}")
        print(f"  Mean abstract length: {df['abstract_length'].mean():.0f} chars")
        print(f"  Mean word count: {df['word_count'].mean():.0f} words")
        print(f"  Abstracts with MeSH terms: {(df['mesh_terms'].str.len() > 0).sum():,}")
        print(f"  Year range: {df['year'].min()} - {df['year'].max()}")
        wc = df["word_count"]
        print(f"  Word count distribution:")
        print(f"    <50 words (very short): {(wc<50).sum():,}")
        print(f"    50-150 words (short): {((wc>=50)&(wc<150)).sum():,}")
        print(f"    150-300 words (standard): {((wc>=150)&(wc<300)).sum():,}")
        print(f"    >300 words (long): {(wc>=300).sum():,}")
    else:
        print(f"  No abstracts retrieved - API may be unavailable")

    print(f"\n--- Readability Pre-scoring ---")
    if all_results and len(df) > 0:
        try:
            import textstat
            sample = df.head(50)
            fk_scores = [textstat.flesch_kincaid_grade(t) for t in sample["abstract"]]
            fre_scores = [textstat.flesch_reading_ease(t) for t in sample["abstract"]]
            print(f"  Sample FK grade (n=50): mean={np.mean(fk_scores):.1f} std={np.std(fk_scores):.1f}")
            print(f"  Sample FRE (n=50): mean={np.mean(fre_scores):.1f} std={np.std(fre_scores):.1f}")
            print(f"  Expected: FK ~12-16 for biomedical abstracts (clinical professional level)")
        except ImportError:
            print(f"  textstat not available - readability scoring deferred to Stage 1 pipeline")
    else:
        print(f"  Readability scoring skipped - no abstracts retrieved")

    print(f"\n--- Key Findings ---")
    print(f"  PubMed E-utilities API verified - rate limit: 3 req/sec without API key")
    print(f"  Seed PMIDs from PubMedQA: {len(seed_pmids)} (labeled split)")
    print(f"  MeSH query expansion: {len(mesh_queries)} queries covering health literacy + clinical NLP")
    print(f"  XML parsing pipeline: title + abstract + journal + year + MeSH terms")
    print(f"  Readability pre-scoring: FK grade + FRE on abstract text")
    print(f"  Stage 1 integration: abstracts feed literacy-conditioned ingestion pipeline")
    print(f"  Note: Full 35M+ PubMed corpus requires bulk download via FTP - API used for seed set")

    print(f"\n--- Saving Corpus to Disk ---")
    if all_results and len(df) > 0:
        os.makedirs("data/processed", exist_ok=True)
        output_path = "data/processed/pubmed_abstracts.csv"
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df):,} abstracts to {output_path}")
        print(f"  Columns: {list(df.columns)}")
    else:
        print(f"  No abstracts to save - API unavailable")

    print(f"\n--- PubMed API Pipeline complete ---")
    print(f"  Ready for text_preprocessor.py - Stage 1 literacy scoring at ingestion time")


if __name__ == "__main__":
    run_pipeline()
