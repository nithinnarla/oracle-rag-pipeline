# ORACLE — Literature Review
## Health Information Accessibility and Literacy-Conditioned RAG

**Period:** January 2026 — April 2026
**Researcher:** Nithin Narla
**Status:** Complete — informed ORACLE framework design

---

## Why I Started Looking At This

Working with a major healthcare client on clinical NLP pipelines — ingesting discharge summaries, clinical notes, and medical literature to answer patient and caregiver queries. The retrieval was accurate. The factual content was correct. And the system was failing patients consistently.

A newly diagnosed diabetic patient asking what they should eat gets a retrieved passage about glycemic index, HbA1c management, and carbohydrate counting written for clinicians. Factually perfect. Completely inaccessible to someone who just received a diagnosis and is frightened. Every standard evaluation metric — ROUGE, BERTScore, retrieval precision — showed the system performing well. The failure was invisible to benchmarks because benchmarks don't measure whether a real patient at a specific literacy level actually understood the output.

Four failure modes kept appearing across deployments:

**Failure 1 — Readability mismatch.** Retrieved content is accurate but written for the wrong audience. The system doesn't know who is asking.

**Failure 2 — Literacy drift across conversation turns.** A system that correctly identifies a patient as low-literacy at turn 1 is confidently serving nurse-level elaborations by turn 5 because the literacy profile estimator compounds small errors over the conversation.

**Failure 3 — Factuality-accessibility inversion.** When outputs were simplified for low-literacy patients, hallucination rates went up. The system dropped medical qualifiers, merged distinct concepts, omitted contraindications to make language simpler. The plain language version was more readable but less safe. Standard monitoring dashboards showed no degradation.

**Failure 4 — Post-processing as the wrong architecture.** Every simplification tool treated plain language as a post-processing step — retrieve accurately, then simplify the output. But simplification introduces errors because you're rewriting content written for a different audience. The right fix is upstream — retrieve documents already written at the right literacy level, or condition retrieval itself on literacy. Nobody had built that.

It is not just patients either. Family members navigating discharge instructions after a loved one's hospitalization — often with low health literacy, high stress, non-native English speakers — face the same system built for clinicians. The population being failed is larger than patients alone.

I started pulling on this thread in January 2026. What I found in the literature confirmed all four failure modes are documented — but no system had addressed the retrieval architecture problem. That is ORACLE.

---

## 1. Literature Review

### RAG Foundations

**Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (NeurIPS)**
The foundational RAG paper. Dense retrieval over a fixed corpus, generation conditioned on retrieved passages. What it does not address: retrieved passages are selected purely on semantic similarity to the query. No notion of whether retrieved content is appropriate for the user's literacy level. The retrieval is literacy-agnostic — the same documents surface for a nurse and a newly diagnosed patient asking the same question. This is the architectural gap ORACLE addresses.

**Izacard & Grave (2021) — Leveraging Passage Retrieval with Generative Models (EACL)**
Fusion-in-Decoder — retrieve multiple passages, fuse during generation. Strong on factual QA. Same limitation as Lewis et al. — retrieval is literacy-agnostic. Fusing multiple clinical passages does not help if all of them are written for clinicians.

**Karpukhin et al. (2020) — Dense Passage Retrieval for Open-Domain QA (EMNLP)**
DPR — the dense retrieval backbone that makes modern RAG practical. Bi-encoder architecture, efficient similarity search. ORACLE builds on this but adds literacy conditioning to the query encoder. The query representation carries both semantic intent and literacy band — changing what gets retrieved, not just how the output is worded.

### Biomedical QA Benchmarks

**Jin et al. (2019) — PubMedQA: A Biomedical Research Question Answering Dataset (EMNLP)**
273,518 biomedical research questions with yes/no/maybe answers grounded in PubMed abstracts (1,000 expert-labeled, 61,249 unlabeled, 211,269 artificially generated). The standard biomedical QA benchmark. Critical limitation for ORACLE: PubMedQA assumes the user can read and interpret biomedical research. Questions are written by researchers, answered by researchers. Patient-facing QA is a different task entirely.

**Pal et al. (2022) — MedMCQA: Large-Scale Multi-Subject Multi-Choice Medical QA (CHIL)**
194K medical school exam questions. Strong coverage of clinical knowledge. Same limitation — tests medical professional knowledge, not patient-facing health information accessibility. ORACLE uses MedMCQA as a retrieval evaluation baseline while acknowledging this scope limitation.

**Jin et al. (2021) — MedQA: USMLE Dataset (Applied Sciences)**
12,723 USMLE board exam questions. The clinical QA benchmark requiring genuine medical reasoning. Useful for evaluating Stage 1 retrieval quality on clinical queries. Same limitation: USMLE tests clinicians, not patients.

**Xiong et al. (2024) — MIRAGE: Benchmarking RAG for Medicine (ACL Findings)**
The most important recent benchmark paper for ORACLE. MIRAGE is the first comprehensive RAG-specific evaluation for medicine — testing retrieval, generation, and faithfulness together. What MIRAGE does not test: accessibility. It evaluates whether the RAG system produces medically accurate outputs, not whether those outputs are understandable to patients at different literacy levels. ORACLE extends MIRAGE's evaluation framework with accessibility metrics.

### Plain Language and Health Literacy

**Guo et al. (2025) — PlainQAFact: Retrieval-Augmented Factual Consistency Evaluation for Biomedical Plain Language Summarization (arXiv)**
The paper that anchored ORACLE's design. PlainQAFact documents what I observed in production — factual consistency degrades when generating plain language summaries of biomedical content. The simplification process introduces errors. The paper proposes a retrieval-augmented evaluation framework to catch these errors. What it does not address: the upstream retrieval problem. PlainQAFact evaluates plain language generation quality; ORACLE conditions retrieval on literacy before generation begins.

**Guo et al. (2024) — Personalized Jargon Identification for Enhanced Interdisciplinary Communication (NAACL)**
Jargon identification as a personalization task — different users need different technical terms explained. Directly relevant to ORACLE's Stage 3 jargon substitution policy. The finding that jargon identification needs to be personalized rather than universal is built into ORACLE's per-literacy-band PEFT adapter design.

**Guo et al. (2024) — APPLS: Evaluating Evaluation Metrics for Plain Language Summarization (EMNLP)**
Meta-evaluation of plain language metrics — which metrics actually predict whether humans understand plain language summaries. The finding: standard NLP metrics do not correlate well with human comprehension. APPLS provides the evaluation framework ORACLE uses in Stage 4. This paper is why ORACLE reports comprehension outcome measurement rather than just readability scores.

**Attal et al. (2023) — PLABA: A Dataset for Plain Language Adaptation of Biomedical Abstracts (Scientific Data)**
750+ biomedical abstracts with expert plain language adaptations. Small dataset, high quality. The critical dataset for ORACLE's Stage 4 evaluation — it has gold-standard plain language references that allow factual consistency verification. Limitation: 750 examples is a small evaluation set. ORACLE uses PLABA for evaluation, not training.

**Ben Abacha & Demner-Fushman (2019) — MedQuAD: A Manually Curated Question-Answer Dataset (BMC Bioinformatics)**
47,441 patient-facing QA pairs from 12 NIH websites including MedlinePlus, NIDDK, NCI, and GARD. Questions written by health consumers, answers written for health consumers. Initially planned as Consumer Health QA (Ben Abacha et al. 2020) — during dataset verification May 2026, MedQuAD was identified as the appropriate downloadable resource from the same NLM/NIH research group. More comprehensive, better documented, and directly relevant to ORACLE's health information accessibility evaluation.

### Clinical Text and MIMIC-III

**Johnson et al. (2016) — MIMIC-III Clinical Database (Scientific Data)**
46,000+ ICU patient records including discharge summaries. The clinical text dataset representing the hardest accessibility challenge — discharge summaries are written by clinicians for clinicians, then handed to patients and caregivers who must navigate post-discharge care. The literacy gap between document author and document reader is largest here. ORACLE uses MIMIC-III discharge summaries as the primary clinical-to-patient translation evaluation. Access via PhysioNet credentialed registration — same credentials as FAPE.

### Health Literacy Frameworks

The NLP literature treats readability as a text property — Flesch-Kincaid scores, SMOG index, syllable counts. The health literacy literature treats it as an interaction between text, reader, and context. These two literatures are not in conversation and the gap matters for ORACLE.

Nutbeam (2000) defined three levels of health literacy — functional, communicative, and critical — that map directly onto ORACLE's literacy band classification. Baker (2006) showed that health literacy predicts health outcomes independently of education and income. The Institute of Medicine (2004) documented that health literacy affects medication adherence, hospitalization rates, and preventive care utilization. These findings are why ORACLE's evaluation includes comprehension outcome measurement — readability scores are a proxy for what actually matters, which is whether patients can use the information to make decisions.

---

## 2. Systematic Review
### RAG for Health Information Accessibility

The RAG literature has developed rapidly since Lewis et al. (2020) but has focused almost entirely on factual accuracy and retrieval precision. The accessibility dimension — whether retrieved and generated content is understandable to the intended user — is absent from every major RAG benchmark including MIRAGE (2024).

The plain language literature has developed separately. Guo et al.'s series at UIUC represents the most systematic work connecting plain language quality to retrieval and generation. PlainQAFact (2025) is the first paper to use retrieval-augmented evaluation for plain language factual consistency. But the retrieval architecture itself remains literacy-agnostic across all published work.

The gap: no paper has conditioned retrieval on user literacy profile. Every system retrieves the same documents regardless of who is asking, then attempts to simplify the output after retrieval. ORACLE's core contribution is moving literacy conditioning upstream into the retrieval step.

---

## 3. Scoping Review
### What Has and Hasn't Been Tried

**Has been tried:**
- Post-hoc simplification of retrieved content
- Readability scoring of generated outputs
- Plain language evaluation metrics via APPLS
- Factual consistency verification for plain language via PlainQAFact
- Biomedical RAG evaluation via MIRAGE

**Has not been tried:**
- Literacy-conditioned dense retrieval
- Per-literacy-band PEFT adapters for generation
- Continuous literacy profile estimation across conversation turns
- Comprehension outcome measurement as primary evaluation metric
- End-to-end evaluation across clinical, consumer, and plain language domains simultaneously

---

## 4. Meta-Analysis
### Readability Metrics vs Comprehension Outcomes

Guo et al. (2024) APPLS is the key paper. Standard readability metrics predict surface features of text — sentence length, syllable count — that correlate with but do not determine comprehension. APPLS shows human comprehension judgments align poorly with these metrics for biomedical content specifically.

The implication for ORACLE: reporting Flesch-Kincaid scores is necessary for comparability with prior work but insufficient as primary evidence of accessibility improvement. ORACLE's Stage 4 evaluation includes downstream task success rate by literacy group — the metric that actually measures whether the system is doing what it claims.

---

## 5. Narrative Review
### Why Literacy-Conditioned Retrieval Changes the Problem

Post-processing simplification has a fundamental limitation: it rewrites content written for a different audience. The errors introduced in simplification come from the mismatch between the source document's intended reader and the target reader. Simplifying a passage written for a clinician into a passage for a patient requires judgment calls about which technical details to drop, which concepts to merge, which qualifications to omit. These judgment calls introduce factual errors — exactly what PlainQAFact documents.

Literacy-conditioned retrieval sidesteps this by retrieving documents already written for the target audience. MedQuAD answers are written for health consumers by NIH subject matter experts. PLABA adaptations are written for general audiences. If the retrieval step surfaces these documents for low-literacy users instead of PubMed abstracts written for researchers, the generation step starts from content appropriate for the audience — reducing the simplification burden and the associated error rate.

This is the architectural insight ORACLE is built on. It is not a new idea in accessibility research — matching content to reader has been studied in educational technology for decades. What is new is applying it to RAG systems for health information and evaluating it rigorously against the production failure modes that motivated the research.

---

## Key Gaps ORACLE Addresses

**Gap 1 — Literacy-conditioned retrieval has never been built or evaluated.**
Every RAG system retrieves literacy-agnostically. ORACLE conditions retrieval on estimated user literacy band. No prior work.

**Gap 2 — Accessibility is treated as post-processing, not architecture.**
Simplification after retrieval introduces errors. ORACLE moves literacy conditioning upstream. Architectural shift not incremental improvement.

**Gap 3 — Production failure modes are documented but not instrumented.**
PlainQAFact (2025) documents the factuality-accessibility inversion. No deployed system catches it in production. ORACLE's Stage 4 monitoring instruments for it continuously.

**Gap 4 — Evaluation measures text properties, not comprehension outcomes.**
MIRAGE measures factual accuracy. APPLS measures plain language quality. No benchmark measures whether patients at different literacy levels actually understand and can use the output. ORACLE measures this.

---

## References

- Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS
- Izacard & Grave (2021) — Leveraging Passage Retrieval with Generative Models, EACL
- Karpukhin et al. (2020) — Dense Passage Retrieval for Open-Domain QA, EMNLP
- Jin et al. (2019) — PubMedQA: A Biomedical Research Question Answering Dataset, EMNLP
- Pal et al. (2022) — MedMCQA: Large-Scale Multi-Subject Multi-Choice Medical QA, CHIL
- Jin et al. (2021) — MedQA: USMLE Dataset, Applied Sciences
- Xiong et al. (2024) — MIRAGE: Benchmarking RAG for Medicine, ACL Findings
- Guo et al. (2025) — PlainQAFact: Retrieval-Augmented Factual Consistency Evaluation for Biomedical Plain Language Summarization, arXiv
- Guo et al. (2024) — Personalized Jargon Identification for Enhanced Interdisciplinary Communication, NAACL
- Guo et al. (2024) — APPLS: Evaluating Evaluation Metrics for Plain Language Summarization, EMNLP
- Attal et al. (2023) — PLABA: A Dataset for Plain Language Adaptation of Biomedical Abstracts, Scientific Data
- Ben Abacha & Demner-Fushman (2019) — MedQuAD: A Manually Curated Question-Answer Dataset, BMC Bioinformatics
- Johnson et al. (2016) — MIMIC-III Clinical Database, Scientific Data
- Nutbeam (2000) — Health Literacy as a Public Health Goal, Health Promotion International
- Baker (2006) — The Meaning and Measure of Health Literacy, Journal of General Internal Medicine
- Institute of Medicine (2004) — Health Literacy: A Prescription to End Confusion, National Academies Press
