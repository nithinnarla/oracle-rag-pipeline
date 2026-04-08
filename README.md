# ORACLE — Optimized Retrieval Augmented Clinical and Lay-language Enquiry

## Overview
ORACLE is a personalized RAG pipeline designed to make complex biomedical information accessible to non-expert users. It dynamically adapts retrieved health content to match a user's health literacy level, background, and information needs — bridging the gap between complex medical knowledge and diverse healthcare consumers.

Evaluated across a retrieval corpus of 35M+ biomedical documents and 400K+ QA evaluation instances spanning clinical, patient-facing, and plain language domains.

## Research Question
**Broad motivation:** How can RAG systems improve health information accessibility for general users?

**This paper specifically asks:** How can RAG systems dynamically adapt retrieved biomedical content to match diverse user health literacy levels, improving accessibility for non-expert healthcare consumers?

## Pipeline Architecture

**Stage 1 — Document Ingestion:**
- PubMed corpus ingestion and chunking
- Semantic segmentation of biomedical text
- Health literacy scoring of source documents
- Metadata tagging by domain and complexity

**Stage 2 — Dense Retrieval:**
- FAISS vector store for efficient similarity search
- BioSentVec + sentence-transformers embeddings
- Hybrid BM25 + dense retrieval fusion
- Query expansion for lay terminology

**Stage 3 — Health Literacy Adaptation:**
- User literacy level classification
- Personalized context window construction
- Jargon identification and simplification
- Adaptive plain language generation

**Stage 4 — Generation and Evaluation:**
- LLM-based lay language summarization
- Readability scoring (Flesch-Kincaid, SMOG)
- Factual consistency verification
- APPLS metric evaluation

## Datasets

| Dataset | Year | Size | Domain | Purpose |
|---------|------|------|--------|---------|
| PubMed Abstracts | 1966-2025 | 35,000,000+ | Biomedical | Retrieval corpus |
| PubMedQA | 2019 | 211,000+ | Biomedical QA | Retrieval eval |
| MedMCQA | 2022 | 194,000+ | Medical QA | Retrieval eval |
| MedQA (USMLE) | 2021 | 12,723 | Clinical QA | Retrieval eval |
| MIRAGE Benchmark | 2024 | 7,663 | Multi-domain QA | RAG eval |
| Consumer Health QA | 2020 | 3,000+ | Patient Q&A | Accessibility eval |
| PLABA | 2023 | 750+ | Plain language | Accessibility eval |

**Retrieval corpus: 35M+ documents**
**Evaluation: 400K+ QA instances across 7 datasets**

Dataset notes:
- PubMed corpus provides large-scale up-to-date biomedical retrieval knowledge base (1966-2025)
- PubMedQA and MedMCQA used for standard biomedical QA retrieval benchmarking
- MIRAGE (2024) is the most comprehensive RAG-specific medical evaluation benchmark available
- Consumer Health QA and PLABA specifically evaluate plain language accessibility — core to this project
- Cross-dataset evaluation ensures generalizability across clinical and patient-facing contexts

## Evaluation Metrics
- Retrieval: Precision@K, Recall@K, MRR, NDCG
- Generation: ROUGE, BLEU, BERTScore
- Readability: Flesch-Kincaid Grade Level, SMOG Index
- Faithfulness: factual consistency score
- Accessibility: APPLS plain language evaluation metrics
- Statistical significance testing across user literacy groups

## Tech Stack
Python, LangChain, FAISS, HuggingFace Transformers, sentence-transformers, BioSentVec, OpenAI GPT-4, BM25, pandas, numpy, matplotlib, seaborn

## Research Timeline
- January 2026: Research conception and literature review
- February 2026: Architecture design and corpus setup
- March 2026: Stage 1-2 retrieval pipeline implementation
- April 2026: Stage 3 health literacy adaptation module
- May 2026: Stage 4 generation and evaluation
- July 2026: Cross-dataset evaluation and ablation studies
- September 2026: Target submission to ACL/SIGIR

## Status
🔬 Research in progress
Target venue: ACL/SIGIR 2026

## Paper
"ORACLE: Optimized Retrieval Augmented Generation for Personalized Health Information Accessibility" — Under development

## References
- Guo et al. (2023) — Retrieval Augmentation of LLMs 
  for Lay Language Generation, Journal of Biomedical 
  Informatics
- Guo et al. (2024) — Personalized Jargon Identification 
  for Enhanced Interdisciplinary Communication, NAACL
- Guo et al. (2023) — APPLS: Evaluating Evaluation 
  Metrics for Plain Language Summarization
- Lewis et al. (2020) — Retrieval-Augmented Generation 
  for Knowledge-Intensive NLP Tasks
- Xiong et al. (2024) — MIRAGE: Benchmarking RAG 
  for Medicine, ACL Findings
- Jin et al. (2019) — PubMedQA: A Biomedical Research 
  Question Answering Dataset
- Pal et al. (2022) — MedMCQA: Large-scale Medical QA
- Jin et al. (2021) — MedQA: USMLE Dataset
- Ben Abacha et al. (2020) — Consumer Health QA Dataset
- Koreeda et al. (2023) — PLABA: Plain Language 
  Adaptations of Biomedical Abstracts
- Johnson et al. (2016) — MIMIC-III Clinical Database
