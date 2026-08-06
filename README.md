# ORACLE — Optimized Retrieval Augmented Clinical and Lay-language Enquiry

## The Problem I Kept Running Into

Eight years of building ML systems in regulated domains — healthcare triage, clinical NLP, enterprise data pipelines — and the same failure mode kept showing up in health information systems. A retrieval system surfaces the right document. The factual content is accurate. The source is credible. And the person reading it has no idea what it means.

Readability and factual accuracy are treated as separate problems in the literature. They are not separate in deployment. A perfectly retrieved, perfectly accurate biomedical passage that a newly-diagnosed diabetic patient cannot parse is a system failure — just one that no standard evaluation metric catches.

ORACLE is my attempt to make that failure mode measurable and fixable.

---

## Research Question

**Broad motivation:** How can RAG systems improve health information accessibility for users across diverse literacy levels?

**This paper specifically asks:** Does conditioning both retrieval and generation on estimated user health-literacy profiles improve downstream comprehension outcomes compared to standard RAG systems — and does that improvement hold across clinical, patient-facing, and plain language evaluation domains?

This is a Comparative research question. The comparison is explicit: literacy-conditioned RAG vs standard RAG, evaluated not just on retrieval quality but on whether users at different literacy levels actually understand the output.

---

## What Existing Systems Get Wrong

Current biomedical RAG systems optimize for retrieval precision and factual consistency. Those are necessary but not sufficient.

Three failure modes that appear in production and are invisible to standard benchmarks:

**Failure 1 — Personalization drift.** A literacy profile estimated at query time degrades over a multi-turn conversation. By turn five, a system that correctly identified a user as a low-literacy patient is confidently serving them nurse-level elaborations because the profile estimator compounded small errors.

**Failure 2 — Factuality-accessibility inversion.** Hallucination rates climb as target reading level drops. Systems optimized for plain language generation produce more factual errors on simpler outputs — the opposite of what monitoring dashboards are tuned to catch. Guo et al.'s PlainQAFact (2025) documents this phenomenon; no deployed system instruments for it.

**Failure 3 — Comprehension degradation at model refresh.** Offline readability metrics (Flesch-Kincaid, SMOG) remain stable after a model update while actual user comprehension — measured by downstream task success — drops below the expert-written baseline within weeks. The evaluation signal and the deployment reality decouple completely.

ORACLE is designed to surface all three.

---

## Pipeline Architecture

**Stage 1 — Document Ingestion:**
- PubMed corpus ingestion and chunking (35M+ documents)
- Semantic segmentation of biomedical text
- Health literacy scoring of source documents at ingestion time
- Metadata tagging by domain, complexity, and target audience

**Stage 2 — Literacy-Conditioned Dense Retrieval:**
- FAISS vector store for efficient similarity search
- BioSentVec + sentence-transformers embeddings
- Hybrid BM25 + dense retrieval fusion
- Query expansion for lay terminology
- Retrieval conditioned on estimated user literacy band — not a post-processing step

**Stage 3 — Health Literacy Adaptation:**
- User literacy level classification from interaction history
- Personalized context window construction per literacy band
- Jargon identification and substitution (building on Guo et al. NAACL 2024)
- PEFT adapter stack per literacy band for adaptive generation

**Stage 4 — Generation and Evaluation:**
- LLM-based lay language summarization
- Readability scoring (Flesch-Kincaid, SMOG Index)
- Factual consistency verification via PlainQAFact methodology
- APPLS metric evaluation for plain language quality
- Comprehension outcome measurement across literacy groups

---

## What Makes This Different from Standard RAG

Most health RAG systems apply plain language simplification as a post-processing step after retrieval. The retrieval itself is literacy-agnostic — the same documents surface for a nurse and a newly-diagnosed patient asking the same question.

ORACLE conditions retrieval on literacy profile before ranking. Different users get different retrieved neighbors, different elaborative spans, different jargon substitution policies — not because the output is simplified afterward, but because the retrieval itself is personalized. This distinction matters for factual consistency: post-processing simplification introduces errors that literacy-conditioned retrieval avoids because simpler documents are retrieved in the first place.

This is contested territory. The field is actively debating whether literacy-conditioned retrieval outperforms post-hoc simplification, and whether PEFT adapter stacks per literacy band generalize or overfit. ORACLE runs both ablations explicitly.

---

## Datasets

| Dataset | Year | Size | Domain | Purpose |
|---------|------|------|--------|---------|
| PubMed Abstracts | 1966-2025 | — | Biomedical | Retrieval corpus |
| PubMedQA | 2019 | 273,518 | Biomedical QA | Retrieval eval |
| MedMCQA | 2022 | 193,155 | Medical QA | Retrieval eval |
| MedQA (USMLE) | 2021 | 11,451 | Clinical QA | Retrieval eval |
| MIRAGE Benchmark | 2024 | 7,663 | Multi-domain QA | RAG eval |
| MedQuAD | 2019 | 47,441 | Patient Q&A | Accessibility eval |
| PLABA | 2023 | 921 | Plain language | Accessibility eval |
| MIMIC-III Clinical | 2001-2012 | — | Clinical | Discharge summary accessibility eval |

**Confirmed: 534,149 records — PubMedQA + MedMCQA + MedQA + MIRAGE + MedQuAD + PLABA verified | MIMIC-III pending PhysioNet approval**

Dataset notes:
- PubMed provides large-scale up-to-date biomedical retrieval base (1966-2025) — pending API access via NCBI E-utilities
- PubMedQA and MedMCQA benchmark standard biomedical QA retrieval — baseline comparison
- MIRAGE (2024) is the most comprehensive RAG-specific medical evaluation benchmark currently available
- MedQuAD and PLABA are the critical datasets — both evaluate plain language accessibility, which is the core research question
- MIMIC-III discharge summaries represent the clinical-to-patient translation challenge — documents written for clinicians that patients and caregivers must navigate after hospital discharge
- Access via PhysioNet credentialed registration (same credentials as FAPE)
- Cross-dataset evaluation ensures ORACLE generalizes across consumer, clinical, and plain language contexts

---

## Evaluation Metrics

- **Retrieval:** Precision@K, Recall@K, MRR, NDCG — by literacy group
- **Generation:** ROUGE, BLEU, BERTScore
- **Readability:** Flesch-Kincaid Grade Level, SMOG Index
- **Faithfulness:** PlainQAFact factual consistency score
- **Accessibility:** APPLS plain language evaluation metrics
- **Comprehension:** Downstream task success rate by literacy group
- **Statistical significance:** Bootstrap confidence intervals across all literacy groups

---

## Tech Stack

Python 3.10+, LangChain, FAISS, HuggingFace Transformers, sentence-transformers, BioSentVec, OpenAI GPT-4, BM25, pandas, numpy, matplotlib, seaborn

Full dependency list: `requirements.txt`

---

## Research Timeline

- January 2026: Research conception — health information accessibility gap identified in clinical NLP deployments
- February 2026: Architecture design — 4-stage literacy-conditioned RAG pipeline designed, dataset corpus planned
- March 2026: Literature review — RAG, health literacy, and plain language summarization domains scoped
- April 2026: GitHub repository created, pipeline architecture and research question documented
- May 2026: Dataset corpus planned, loaders under development
- June 2026: Stage 1 complete — corpus pipeline (37,076 records, 6 sources, 4 literacy bands); EDA complete (67 figures); FK-based literacy scoring
- July 2026: Stage 2 complete — DPR encoder (768-dim embeddings); FK rule-based query router; literacy-conditioned retrieval pipeline; retrieval evaluation (20 queries, 7 figures)
- July 2026: Stage 3 complete — health literacy adaptation (literacy_adapter.py); PEFT LoRA adapters per band (peft_adapter.py); medical jargon identifier 1,491 terms 5 figures (jargon_identifier.py)
- August 2026: Stage 4 complete — gpt-4o-mini generation pipeline (generation_pipeline.py, FK grade increases low(6.3)→medium(11.6)→high(15.1)→clinical(16.1)); lay language summarizer (lay_summarizer.py, PLABA abstracts source FK 14.9→generated FK 8.4 vs expert FK 13.5, 1 figure); generation-level literacy conditioning confirmed, retrieval-level conditioning still unreliable for low band (see known limitation below); pre-paper audit Aug 13; paper writing starts Aug 14; submission JBI Sep 4 2026

---

## Status

🔬 Research in progress — Stages 1-4 complete, paper writing starts Aug 14 2026

**Stage 1 (complete):** Corpus pipeline — 37,076 records across 6 sources, 4 literacy bands — 67 EDA figures.
**Stage 2 (complete):** DPR retrieval pipeline with literacy-conditioned band indexing — 7 figures.
**Stage 3 (complete):** Health literacy adaptation — literacy_adapter.py (PLABA injection, 38-term jargon substitution); peft_adapter.py (LoRA adapters 0.40% trainable params); jargon_identifier.py (1,491 medical terms, 5 figures).
**Stage 4 (complete):** Generation pipeline, evaluation, and factual consistency check.

- **Generation-level literacy conditioning confirmed:** gpt-4o-mini FK grade increases low(6.3)->medium(11.6)->high(15.1)->clinical(16.1). The system prompt per band produces text at the intended reading level regardless of what was retrieved.
- **Retrieval-level conditioning is NOT confirmed** -- the FK-based router misclassifies roughly half of queries. For the misrouted low-band query tested, all 5 retrieved documents were MedQA/MIRAGE exam-question fragments rather than plain-language content (one entirely irrelevant to the query topic). This is an already-documented, accepted architectural risk, not a new finding: Decision 10 states MedMCQA+MedQA+MIRAGE make up 93.6% of the retrieval corpus and are clinical-professional content, not patient-facing; Decision 11 documents the same 4/8 (50%) routing accuracy on test queries and specifies that Stage 3 -- not Stage 2 routing -- is architecturally responsible for correcting literacy mismatches.
- **New from Aug 1 verification:** the misrouted query didn't just land in the wrong literacy band -- it returned one document entirely off-topic to the query (a sports-psychology question for a diabetes query), a relevance failure distinct from the literacy-band failure already documented (see research_design_rationale.md).
- **Lay language summarizer:** PLABA abstracts rewritten by gpt-4o-mini score FK 8.4 mean vs. source 14.9 and expert PLABA adaptations 13.5 mean -- generated text notably simpler than the expert baseline. 1 figure.
- **Factual consistency (Aug 4 2026):** evaluated via an adapted GPT-4o-mini methodology, not the official PlainQAFact metric (see Decision 13 for why). Result: ~0.96 overall consistency, with elaboration claims (0.856-0.862) notably less consistent than simplification claims (0.972-0.980) -- matching PlainQAFact's own documented finding that elaborative explanations are more hallucination-prone. Official PlainQAFact metric: complete, all 20 records, run twice for stability (internal_mean ~0.65, external_mean ~0.26, overall_mean ~0.33). Diverges substantially from the adapted result (~0.96) -- confirmed both mechanistically (direct retrieval test) and systematically (64.1% of 192 external claims scored below 0.3, vs. only 21.4% of 42 internal claims) to reflect a genuine domain mismatch, not a pipeline error: PlainQAFact's Textbooks knowledge base covers foundational medical education content, not the clinical-trial-specific claims in ORACLE's PLABA source texts, so retrieval surfaces topically-similar but factually-non-matching passages. StatPearls (the other half of the combined knowledge base) verified separately as genuinely functional and more precise for the same query type, but does not fully compensate. See Decision 13 for the complete investigation, both runs' numbers, and what this means for reporting both scores. 2 figures.

Target venue: Journal of Biomedical Informatics — submission Sep 4 2026

---

## Paper

"ORACLE: Optimized Retrieval Augmented Generation for Personalized Health Information Accessibility" — Under development

---

## References

- Guo et al. (2025) — PlainQAFact: Retrieval-augmented Factual Consistency Evaluation for Biomedical Plain Language Summarization, arXiv
- Guo et al. (2024) — Personalized Jargon Identification for Enhanced Interdisciplinary Communication, NAACL
- Guo et al. (2024) — APPLS: Evaluating Evaluation Metrics for Plain Language Summarization, EMNLP
- Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS
- Xiong et al. (2024) — MIRAGE: Benchmarking RAG for Medicine, ACL Findings
- Jin et al. (2019) — PubMedQA: A Biomedical Research Question Answering Dataset, EMNLP
- Pal et al. (2022) — MedMCQA: Large-scale Medical QA, CHIL
- Jin et al. (2021) — MedQA: USMLE Dataset, Applied Sciences
- Ben Abacha & Demner-Fushman (2019) — MedQuAD: A Manually Curated Question-Answer Dataset, BMC Bioinformatics
- Koreeda et al. (2023) — PLABA: Plain Language Adaptations of Biomedical Abstracts, arXiv
- Johnson et al. (2016) — MIMIC-III Clinical Database, Scientific Data
