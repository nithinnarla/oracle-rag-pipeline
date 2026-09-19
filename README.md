# ORACLE, Optimized Retrieval Augmented Clinical and Lay-language Enquiry

## The Problem I Kept Running Into

Eight years of building ML systems in regulated domains, healthcare triage, clinical NLP, enterprise data pipelines, and the same failure mode kept showing up in health information systems. A retrieval system surfaces the right document. The factual content is accurate. The source is credible. And the person reading it has no idea what it means.

Readability and factual accuracy are treated as separate problems in the literature. They are not separate in deployment. A perfectly retrieved, perfectly accurate biomedical passage that a newly-diagnosed diabetic patient cannot parse is a system failure, just one that no standard evaluation metric catches.

ORACLE is my attempt to make that failure mode measurable and fixable.

---

## Research Question

**Broad motivation:** How can RAG systems improve health information accessibility for users across diverse literacy levels?

**This paper specifically asks:** where in a retrieval-augmented pipeline does literacy conditioning actually reach? Holding the retrieved context fixed and varying only whether generation used the correct literacy band, which measured outcomes move and which do not?

That is narrower than the motivation above, and deliberately so. The comparison run is correct band against wrong band on identical context. It is **not** literacy-conditioned RAG against standard unconditioned RAG: no such baseline was built, which is the study's main limitation and is stated in Section 6.4 of the paper. Comprehension outcomes were not measured either. The question as originally framed, quoted above the rewrite, is the motivation; the question answered is the one stated here.

---

## What Existing Systems Get Wrong

Current biomedical RAG systems optimize for retrieval precision and factual consistency. Those are necessary but not sufficient.

Three failure modes that appear in production and are invisible to standard benchmarks:

**Failure 1, Personalization drift.** A literacy profile estimated at query time degrades over a multi-turn conversation. By turn five, a system that correctly identified a user as a low-literacy patient is confidently serving them nurse-level elaborations because the profile estimator compounded small errors.

**Failure 2, Factuality-accessibility inversion.** Hallucination rates climb as target reading level drops. Systems optimized for plain language generation produce more factual errors on simpler outputs, the opposite of what monitoring dashboards are tuned to catch. You and Guo's PlainQAFact (2026, Journal of Biomedical Informatics) documents this phenomenon; no deployed system instruments for it.

**Failure 3, Readability metrics are not comprehension.** Flesch-Kincaid and SMOG measure surface features, sentence length and syllable count, and a text can score as accessible while remaining unreadable in practice. This corpus shows the failure directly: MedMCQA's short, jargon-dense exam stems make up 80.3% of the low-literacy band, while PLABA, the one source that is genuine plain-language writing, contributes none of it.

An earlier version of this section claimed that readability scores stay stable after a model update while user comprehension drops within weeks. That claim had no source and nothing here measures it; comprehension was never measured at all. What is measured is metric sensitivity, in Section 5.5 of the paper.

ORACLE surfaces the first two directly and documents the third as an unmitigated limitation.

---

## Pipeline Architecture

> **Record note, September 17 2026.** This section described the system as designed
> in early 2026. It is rewritten here to describe what was actually built and
> evaluated, because several planned components were never implemented and a reader
> comparing this file with the paper would have found them contradicting each other.
> docs/paper_draft.md is the current statement of the study, and Section 6.4 lists
> what remains future work.

**Stage 1, Document ingestion and literacy scoring:**
- Five public sources loaded and unified into one corpus of 36,664 records
- Flesch-Kincaid grade scored per record at ingestion
- Each record assigned to one of four literacy bands, low FK <= 6, medium 6 < FK <= 10, high 10 < FK <= 14, clinical FK > 14
- Minimum question-length quality filter applied across sources

**Stage 2, Literacy-conditioned dense retrieval:**
- DPR bi-encoder, facebook/dpr-question_encoder-single-nq-base, used unmodified
- Per-band embedding matrices, cosine similarity computed directly over numpy arrays
- Hard band selection: a query's estimated band restricts the candidate pool to that band's records before ranking
- Conditioning acts on which records are eligible, not on the query representation

**Stage 3, Band-conditioned generation:**
- Band-specific system prompt templates, one per literacy band, combined with the retrieved documents
- Medical jargon frequency analysis over the corpus, 1,491 terms identified, feeding a 38-term substitution table
- A PEFT/LoRA adapter stack per band was built and structurally validated but never trained to convergence, and produces none of the reported results

**Stage 4, Evaluation:**
- Self-retrieval evaluation, each record's own question as query and its own record_id as ground truth, Precision@1, Recall@K and MRR by band
- Routing-correctness evaluation, wrong band against correct band on identical retrieved context
- Readability via Flesch-Kincaid and SMOG
- Generation quality via ROUGE-L and BERTScore
- Factual consistency two ways, an adapted GPT-4o-mini method and the official PlainQAFact metric
- APPLS perturbation testing applied to this corpus to validate the metric suite

Not built, and named as future work in the paper: PubMed-scale ingestion, FAISS
indexing, BM25 hybrid fusion, BioSentVec embeddings, query expansion, literacy
estimation from interaction history, trained band adapters, and any measurement of
human comprehension.

---

## What Makes This Different from Standard RAG

Most health RAG systems apply plain language simplification after retrieval. The
retrieval itself is literacy-agnostic, so the same documents surface for a nurse and
a newly diagnosed patient asking the same question.

ORACLE conditions retrieval instead: a query's estimated literacy band selects that
band's candidate pool before ranking, so the constraint acts upstream of generation
rather than on its output.

What the evaluation found, and it is narrower than the design hoped for: correct
band routing significantly improves readability, Wilcoxon n=177 p=0.0018, and has no
measurable effect on retrieval relevance or factual consistency, n=180 p=0.640 and
p=0.937. The earlier version of this section claimed that conditioning retrieval
avoids the factual errors post-hoc simplification introduces. This work does not
show that. It found no fidelity difference either way, and it never ran the
comparison against post-hoc simplification that such a claim would need.

---

## Datasets

| Dataset | Year | In corpus | Domain | Role |
|---------|------|-----------|--------|------|
| MedMCQA | 2022 | 15,732 | Medical exam QA | Corpus, retrieval eval |
| MedQA (USMLE) | 2021 | 11,431 | Clinical QA | Corpus, retrieval eval |
| MIRAGE Benchmark | 2024 | 7,580 | Multi-domain medical QA | Corpus, retrieval eval |
| PubMedQA | 2019 | 1,000 | Biomedical QA | Corpus, retrieval eval |
| PLABA | 2023 | 921 | Plain language | Corpus, plain-language reference |

**Corpus total: 36,664 records across five sources.** The per-source figures above
are the counts after capping and quality filtering, not the published dataset sizes;
MedMCQA for instance contributes 15,732 of its 193,155 records after a 20,000-record
subject-stratified cap and a minimum question-length filter.

Evaluated and not included:
- **PubMed abstracts.** 412 records were in an earlier corpus build and were dropped at commit adc79d5. They remain in the cross-dataset evaluation only, at 118 rows, which is why the paper reports five corpus sources and six evaluation sources.
- **MedQuAD.** 47,457 published QA pairs, of which the HuggingFace redistribution carries 47,441. Its authors removed the answers from three subsets to comply with MedlinePlus copyright, leaving about 16,407 usable records. Re-integrating it means crawling the removed answers and is named as future work; it is the one source in this space built for patient-facing question answering, so its absence is a real limitation.
- **MIMIC-III.** PhysioNet credentialed access was never granted, no loader was written, and it appears nowhere in the study.

---

## Evaluation Metrics

- **Retrieval:** Precision@1, Recall@5, Recall@10, MRR, by literacy band, and under a pool-size-controlled comparison
- **Readability:** Flesch-Kincaid grade, SMOG index
- **Generation quality:** ROUGE-L, BERTScore
- **Factual consistency:** an adapted GPT-4o-mini method against source abstracts, and the official PlainQAFact metric against an external knowledge base
- **Metric validation:** APPLS perturbation testing on this corpus, to check the metrics are sensitive to the transformations this pipeline cares about
- **Significance:** paired Wilcoxon for routing comparisons, Fisher exact for band differences, point-biserial and chi-square for the ablation

Not measured: NDCG, BLEU, bootstrap confidence intervals, and comprehension
outcomes. The last is the substantive gap and is stated as such in the paper.

---

## Tech Stack

Python 3.11.9, HuggingFace Transformers with DPR, torch, OpenAI gpt-4o-mini,
rouge-score, bert-score, textstat, scipy, pandas, numpy, matplotlib, seaborn.

Full dependency list: `requirements.txt`

---

## Research Timeline

- January 2026: Research conception, health information accessibility gap identified in clinical NLP deployments
- February 2026: Architecture design, 4-stage literacy-conditioned RAG pipeline designed, dataset corpus planned
- March 2026: Literature review, RAG, health literacy, and plain language summarization domains scoped
- April 2026: GitHub repository created, pipeline architecture and research question documented
- May 2026: Dataset corpus planned, loaders under development
- June 2026: Stage 1 complete, corpus pipeline (37,076 records, 6 sources, 4 literacy bands); EDA complete (67 figures); FK-based literacy scoring
- July 2026: Stage 2 complete, DPR encoder (768-dim embeddings); FK rule-based query router; literacy-conditioned retrieval pipeline; retrieval evaluation (20 queries, 7 figures)
- July 2026: Stage 3 complete, health literacy adaptation (literacy_adapter.py); PEFT LoRA adapters per band (peft_adapter.py); medical jargon identifier 1,491 terms 5 figures (jargon_identifier.py)
- August 2026: Stage 4 complete
  - gpt-4o-mini generation pipeline (generation_pipeline.py): FK grade increases low(6.3)->medium(11.6)->high(15.1)->clinical(16.1)
  - Lay language summarizer (lay_summarizer.py): PLABA abstracts source FK 14.9->generated FK 8.4 vs expert FK 12.6, 1 figure
  - Generation-level literacy conditioning confirmed; retrieval-level conditioning still unreliable for low band (see known limitation below)
  - Pre-paper audit Aug 13
- September 2026: Paper outline (1,993 words) and draft (5,870 words, 12 figures) complete, targeting Sep 22 submission to JBI

---

## Status

Research in progress, Stages 1-4 complete, paper draft complete and under pre-submission audit

**Stage 1 (complete):** Corpus pipeline, 36,664 records across 5 sources, 4 literacy bands, 67 EDA figures. (An earlier build counted 37,076 across 6 sources; the 412 PubMed abstracts were dropped from the corpus at commit adc79d5 and now appear only in the cross-dataset evaluation.)
**Stage 2 (complete):** DPR retrieval pipeline with literacy-conditioned band indexing, 7 figures.
**Stage 3 (complete):** Health literacy adaptation, literacy_adapter.py (PLABA injection, 38-term jargon substitution); peft_adapter.py (LoRA adapters 0.40% trainable params); jargon_identifier.py (1,491 medical terms, 5 figures).
**Stage 4 (complete):** Generation pipeline, evaluation, and factual consistency check.

- **Generation-level literacy conditioning confirmed:** gpt-4o-mini FK grade increases low(6.3)->medium(11.6)->high(15.1)->clinical(16.1). The system prompt per band produces text at the intended reading level regardless of what was retrieved.
- **Retrieval-level conditioning is NOT confirmed**: the FK-based router misclassifies roughly half of queries. For the misrouted low-band query tested, all 5 retrieved documents were MedQA/MIRAGE exam-question fragments rather than plain-language content (one entirely irrelevant to the query topic). This is an already-documented, accepted architectural risk, not a new finding: Decision 10 states MedMCQA+MedQA+MIRAGE make up 93.6% of the retrieval corpus, which is 94.8% of the current five-source corpus and are clinical-professional content, not patient-facing; Decision 11 documents the same 4/8 (50%) routing accuracy on test queries and specifies that Stage 3, not Stage 2 routing, is architecturally responsible for correcting literacy mismatches.
- **New from Aug 1 verification:** the misrouted query didn't just land in the wrong literacy band; it returned one document entirely off-topic to the query (a sports-psychology question for a diabetes query), a relevance failure distinct from the literacy-band failure already documented (see research_design_rationale.md).
- **Lay language summarizer:** PLABA abstracts rewritten by gpt-4o-mini score FK 8.4 mean vs. source 14.9 and expert PLABA adaptations 12.6 mean; generated text notably simpler than the expert baseline. 1 figure.
- **Factual consistency (Aug 4 2026):** evaluated via an adapted GPT-4o-mini methodology, not the official PlainQAFact metric (see Decision 13 for why). Result: ~0.96 overall consistency, with elaboration claims (0.856-0.862) notably less consistent than simplification claims (0.972-0.980), matching PlainQAFact's own documented finding that elaborative explanations are more hallucination-prone. Official PlainQAFact metric: complete, all 20 records, run twice for stability (internal_mean ~0.65, external_mean ~0.26, overall_mean ~0.33). Diverges substantially from the adapted result (~0.96); confirmed both mechanistically (direct retrieval test) and systematically (64.1% of 192 external claims scored below 0.3, vs. only 21.4% of 42 internal claims) to reflect a genuine domain mismatch, not a pipeline error: PlainQAFact's Textbooks knowledge base covers foundational medical education content, not the clinical-trial-specific claims in ORACLE's PLABA source texts, so retrieval surfaces topically-similar but factually-non-matching passages. StatPearls (the other half of the combined knowledge base) verified separately as genuinely functional and more precise for the same query type, but does not fully compensate. See Decision 13 for the complete investigation, both runs' numbers, and what this means for reporting both scores. 2 figures.

Target venue: Journal of Biomedical Informatics, submission Sep 22 2026

---

## Paper

"Where Literacy Conditioning Reaches in a Retrieval-Augmented Pipeline: Readability Improves, Retrieval Relevance and Factual Fidelity Do Not"

Submitted to the Journal of Biomedical Informatics. Manuscript: docs/paper_draft.md.
The earlier working title, "ORACLE: Optimized Retrieval Augmented Generation for
Personalized Health Information Accessibility", described the system rather than the
finding and was replaced.

---

## References

- You & Guo (2026), PlainQAFact: Retrieval-augmented factual consistency evaluation metric for biomedical plain language summarization, Journal of Biomedical Informatics 178, 105019
- Guo et al. (2024), Personalized Jargon Identification for Enhanced Interdisciplinary Communication, NAACL
- Guo et al. (2024), APPLS: Evaluating Evaluation Metrics for Plain Language Summarization, EMNLP
- Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS
- Xiong et al. (2024), MIRAGE: Benchmarking RAG for Medicine, ACL Findings
- Jin et al. (2019), PubMedQA: A Biomedical Research Question Answering Dataset, EMNLP
- Pal et al. (2022), MedMCQA: Large-scale Medical QA, CHIL
- Jin et al. (2021), MedQA: USMLE Dataset, Applied Sciences
- Ben Abacha & Demner-Fushman (2019), A question-entailment approach to question answering, BMC Bioinformatics 20(1), 511 (introduces MedQuAD)
- Attal, Ondov & Demner-Fushman (2023), A dataset for plain language adaptation of biomedical abstracts, Scientific Data 10, 8 (PLABA)
- Johnson et al. (2016), MIMIC-III Clinical Database, Scientific Data (evaluated for inclusion, access never granted, not used)
