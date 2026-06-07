# ORACLE — Literature Analysis
## Health Information Accessibility and Literacy-Conditioned RAG

**Period:** January 2026 — April 2026
**Researcher:** Nithin Narla
**Status:** Complete — all 9 protocols applied across 16 papers

**Verified dataset pipeline at time of analysis:**
- PubMedQA: 273,518 records (Jin et al. 2019)
- MedMCQA: 193,155 records (Pal et al. 2022)
- MedQA USMLE: 11,451 records — 4-options version (Jin et al. 2021)
- MIRAGE: 7,663 records (Xiong et al. 2024)
- MedQuAD: 47,441 records — replaces Consumer Health QA, same NLM/NIH group (Ben Abacha & Demner-Fushman 2019)
- PLABA: 921 records across 75 health topics (Attal et al. 2023)
- MIMIC-III: pending PhysioNet credentialed access (Johnson et al. 2016)
- PubMed Abstracts: 35M+ pending API integration (Phase 4)

**Confirmed verified total: 534,149 records**

---

## Protocol 1 — Intake: Paper Table, Clusters, Conflicts

### Paper Table

| Paper | Year | Venue | Type | Relevance |
|-------|------|-------|------|-----------|
| Lewis et al. — RAG | 2020 | NeurIPS | Foundational | RAG architecture baseline |
| Karpukhin et al. — DPR | 2020 | EMNLP | Foundational | Dense retrieval backbone |
| Izacard & Grave — FiD | 2021 | EACL | Architecture | Multi-passage fusion |
| Jin et al. — PubMedQA | 2019 | EMNLP | Dataset | Biomedical QA benchmark |
| Pal et al. — MedMCQA | 2022 | CHIL | Dataset | Clinical QA benchmark |
| Jin et al. — MedQA | 2021 | Applied Sciences | Dataset | USMLE clinical benchmark |
| Xiong et al. — MIRAGE | 2024 | ACL Findings | Benchmark | RAG medical evaluation |
| Guo et al. — PlainQAFact | 2025 | arXiv | Evaluation | Plain language factuality |
| Guo et al. — Jargon | 2024 | NAACL | Method | Personalized jargon ID |
| Guo et al. — APPLS | 2024 | EMNLP | Evaluation | Plain language metrics |
| Attal et al. — PLABA | 2023 | Scientific Data | Dataset | Plain language gold standard |
| Ben Abacha & Demner-Fushman — MedQuAD | 2019 | BMC Bioinformatics | Dataset | Patient-facing QA |
| Johnson et al. — MIMIC-III | 2016 | Scientific Data | Dataset | Clinical discharge summaries |
| Nutbeam — Health Literacy | 2000 | Health Promo Intl | Framework | Literacy classification |
| Baker — Health Literacy | 2006 | J Gen Internal Med | Framework | Literacy-outcomes link |
| Institute of Medicine | 2004 | National Academies | Framework | Health literacy policy |

### Clusters

**Cluster 1 — RAG Architecture (3 papers):**
Lewis 2020, Karpukhin 2020, Izacard 2021 — all focused on retrieval accuracy and factual QA. None address accessibility. This cluster defines the architectural gap ORACLE fills.

**Cluster 2 — Biomedical QA Benchmarks (4 papers):**
PubMedQA, MedMCQA, MedQA, MIRAGE — all evaluate clinical professional knowledge. None measure patient-facing health information comprehension. This cluster defines the evaluation gap.

**Cluster 3 — Plain Language Research (4 papers):**
Guo et al. series + PLABA — documents production failure modes, proposes evaluation frameworks. Most systematic recent work on plain language quality in biomedical NLP. This cluster motivates ORACLE's Stage 4 evaluation design.

**Cluster 4 — Patient-Facing Datasets (2 papers):**
MedQuAD + MIMIC-III — the two datasets that represent real patient information needs. MedQuAD is patient-facing from the start. MIMIC-III is clinician-facing and handed to patients. Both are critical for ORACLE.

**Cluster 5 — Health Literacy Frameworks (3 papers):**
Nutbeam 2000, Baker 2006, IOM 2004 — public health literature that NLP has ignored. Provides the theoretical basis for ORACLE's literacy band classification and outcome measurement framework.

### Conflicts

**Conflict 1 — Readability metrics vs comprehension outcomes:**
APPLS (2024) shows standard readability metrics predict surface features, not comprehension. Most prior work uses Flesch-Kincaid as primary metric. For ORACLE's evaluation design these two positions cannot coexist — reporting Flesch-Kincaid as primary evidence of accessibility improvement would contradict what APPLS demonstrates. ORACLE reports readability metrics for comparability with prior work but treats downstream task success rate as primary.

**Conflict 2 — Simplification accuracy vs accessibility:**
PlainQAFact (2025) documents that simplification degrades factual consistency. Prior plain language work treated simplification as uniformly beneficial. This is not a minor disagreement — it changes the architectural direction entirely. Post-hoc simplification optimizes the wrong thing. ORACLE's upstream retrieval conditioning is the architectural response.

**Conflict 3 — Benchmark coverage vs patient population:**
MIRAGE (2024) is the strongest RAG medical benchmark but evaluates clinical knowledge not patient comprehension. Treating MIRAGE performance as evidence of accessibility improvement would be misleading. ORACLE uses MIRAGE for retrieval quality evaluation and adds separate comprehension outcome metrics for accessibility evaluation.

---

## Protocol 2 — Contradiction Finder

**Contradiction 1 — RAG is literacy-agnostic by design:**
Lewis et al. (2020) retrieves based on semantic similarity. Karpukhin et al. (2020) optimizes for factual recall. Both assume the same document is appropriate for all users. The entire RAG literature takes this as given. Health literacy research (Nutbeam 2000, Baker 2006) has documented for 25 years that the same health information is not accessible to all users. These two bodies of literature have never been brought into direct conflict. ORACLE is the first system to treat this as a design problem rather than an edge case.

**Contradiction 2 — Simplification helps vs simplification introduces errors:**
The plain language literature before 2024 treated simplification as uniformly beneficial — simpler language improves health outcomes, therefore simplify everything. PlainQAFact (2025) directly contradicts this — simplification of biomedical content systematically degrades factual consistency. Simplification helps when the source content is already appropriate for the audience. It introduces errors when rewriting content written for a different audience. ORACLE's architecture resolves this by retrieving audience-appropriate content rather than rewriting.

**Contradiction 3 — Personalization is useful vs personalization is unscalable:**
Guo et al. (2024) jargon paper shows personalized jargon identification outperforms universal jargon dictionaries. Standard biomedical NLP treats vocabulary as universal. Per-user models resolve the personalization problem but do not scale. ORACLE uses per-literacy-band PEFT adapters — personalization by literacy group rather than individual, balancing quality and scalability.

---

## Protocol 3 — Citation Chain: Three Concepts Tracked

**Concept 1 — Literacy-Conditioned Retrieval:**
DPR (Karpukhin 2020) → RAG (Lewis 2020) → FiD (Izacard 2021) → MIRAGE (Xiong 2024)
The citation chain for retrieval architecture stops at factual accuracy. No paper in this chain adds literacy conditioning. ORACLE is the next step — DPR backbone with literacy-conditioned query encoder.

**Concept 2 — Plain Language Factuality:**
PLABA (Attal 2023) → APPLS (Guo 2024) → PlainQAFact (Guo 2025)
Clean three-paper chain. PLABA creates the gold standard data. APPLS identifies that standard metrics fail on plain language. PlainQAFact builds retrieval-augmented evaluation to catch factual errors. ORACLE's Stage 4 evaluation builds directly on this chain.

**Concept 3 — Health Literacy as Outcome Predictor:**
Nutbeam (2000) → Baker (2006) → IOM (2004) → [gap] → ORACLE
The public health chain establishes that health literacy predicts outcomes. The NLP chain has never picked this up. ORACLE connects these two chains — using health literacy frameworks to design evaluation metrics that measure what actually matters, not just what is easy to compute.

---

## Protocol 4 — Gap Scanner: Five Gaps Ranked

**Gap 1 — No literacy-conditioned retrieval system exists (Critical):**
Every RAG system retrieves literacy-agnostically. The entire RAG literature — Lewis, Karpukhin, Izacard, MIRAGE — assumes retrieval quality is independent of user literacy. This is the core architectural gap ORACLE addresses. No prior work. No partial solution. Clear contribution.

**Gap 2 — Accessibility evaluation is absent from medical RAG benchmarks (Critical):**
MIRAGE evaluates factual accuracy and retrieval precision. No existing benchmark measures whether RAG outputs are comprehensible to patients at different literacy levels. ORACLE adds this evaluation layer. Without it, a system can score well on MIRAGE while systematically failing patients.

**Gap 3 — Simplification as post-processing is architecturally wrong (High):**
PlainQAFact documents the error. No paper has proposed the architectural fix. ORACLE's upstream retrieval conditioning is the fix. This gap exists in both the RAG literature and the plain language literature simultaneously.

**Gap 4 — Health literacy frameworks have never been operationalized in NLP systems (High):**
Nutbeam's three-level framework and Baker's outcome research have been cited in health informatics for 20+ years but never implemented as NLP system design constraints. ORACLE's literacy band classification is the first operationalization in a RAG pipeline.

**Gap 5 — Clinical-to-patient translation lacks end-to-end evaluation (Medium):**
MIMIC-III discharge summaries represent the hardest accessibility challenge — clinician documents handed to patients. No existing benchmark evaluates end-to-end performance on this specific translation task with comprehension outcome measurement. ORACLE evaluates this with MIMIC-III (pending PhysioNet) and PLABA jointly.

**Dataset coverage of gaps at time of analysis:**
- Gap 1 — addressed by pipeline design; no existing dataset covers it
- Gap 2 — addressed by MIRAGE (7,663) + comprehension metrics
- Gap 3 — addressed by MedQuAD (47,441) + PLABA (921)
- Gap 4 — addressed by pipeline design + all 6 verified datasets
- Gap 5 — addressed by MIMIC-III (pending) + PLABA (921)

---

## Protocol 5 — Methodology Audit

**Lewis et al. (2020) — RAG:**
Dense retrieval + seq2seq generation. Strong methodology for factual QA. Critical limitation: retrieval optimization is literacy-agnostic by construction. The bi-encoder loss function has no literacy term. Cannot be extended to literacy conditioning without architectural modification.

**Karpukhin et al. (2020) — DPR:**
Bi-encoder trained on Natural Questions. In-batch negatives for efficient training. Strong retrieval baseline. Limitation: trained on web QA — domain shift to biomedical content requires fine-tuning. ORACLE fine-tunes on MedQuAD and PubMedQA.

**Xiong et al. (2024) — MIRAGE:**
Evaluates 7 RAG systems across 5 medical QA datasets — MedQA, MedMCQA, PubMedQA, BioASQ, MMLU-Med. 7,663 total questions. Strong benchmark design. Limitation: no accessibility evaluation. Factual accuracy is necessary but not sufficient for ORACLE's use case.

**Guo et al. (2025) — PlainQAFact:**
Retrieval-augmented factual consistency evaluation using question generation and answering. Demonstrates factual degradation during plain language generation. Strong evidence base. Limitation: evaluation framework only — does not propose the architectural fix that would prevent the errors it documents.

**Guo et al. (2024) — APPLS:**
Meta-evaluation across 8 automatic metrics and human judgments on 200 summaries. Shows poor correlation between standard metrics and comprehension. Directionally correct but the small sample size means effect sizes need replication at scale. ORACLE treats APPLS findings as motivation for evaluation design, not as settled science.

**Attal et al. (2023) — PLABA:**
Expert annotators from NLM — strong quality control. 75 topics, 10 abstracts per topic, multiple adaptation versions per abstract yielding 921 rows from 750 abstracts. Small by NLP standards but gold standard quality. Limitation: 75 topics is not comprehensive coverage of medical domains. ORACLE uses PLABA for evaluation not training.

---

## Protocol 6 — Master Synthesis (400 words)

The RAG literature and the health literacy literature have been developing in parallel for 25 years without meaningful contact. The RAG literature, from Lewis et al. (2020) through MIRAGE (2024), has built increasingly sophisticated retrieval and generation systems optimized for factual accuracy. The health literacy literature, from Nutbeam (2000) through the IOM (2004), has documented with increasing precision that health literacy predicts health outcomes independently of education and income — and that the same health information is not equally accessible to all populations. The failure to connect these two literatures is not an oversight. It is a structural problem: the NLP community evaluates systems against benchmarks designed by and for clinical professionals, and health literacy research has not produced datasets or evaluation protocols that NLP systems can use directly.

ORACLE is built on the observation that this gap has concrete consequences. The failure modes I observed in production — readability mismatch, literacy drift across turns, factuality-accessibility inversion, post-processing as the wrong architecture — are all downstream of one root cause: RAG systems retrieve the same documents for all users regardless of literacy level, then attempt to simplify outputs written for the wrong audience.

The Guo et al. series at UIUC (2024-2025) represents the most systematic recent work on the NLP side of this problem. APPLS (2024) shows that standard readability metrics fail to predict human comprehension of biomedical plain language. PlainQAFact (2025) shows that simplification of biomedical content degrades factual consistency. Together these papers document the problem precisely. Neither proposes the architectural fix because both treat simplification as the intervention — they are trying to do it better, not questioning whether it is the right approach.

The fix is upstream. Literacy-conditioned retrieval changes what gets retrieved based on who is asking, not just what they are asking. If a low-literacy user is identified, the retrieval step should surface MedQuAD answers written by NIH for health consumers and PLABA adaptations written for general audiences — not PubMed abstracts written for researchers. The generation step then starts from content appropriate for the audience, reducing the simplification burden and the associated error rate simultaneously.

Three design choices follow from this analysis. First, literacy conditioning must be applied at the query encoder level — changing the representation used for retrieval, not filtering outputs afterward. Second, per-literacy-band PEFT adapters rather than a single generation model — different literacy bands require systematically different generation policies. Third, evaluation must include comprehension outcome measurement — APPLS shows text-level metrics are insufficient, and 534,149 verified records across six datasets still leave the accessibility evaluation gap completely open.

---

## Protocol 7 — Assumption Killer (6 Assumptions)

**Assumption 1 — Literacy band can be reliably estimated from a short conversation:**
The entire ORACLE pipeline depends on accurate literacy profile estimation. Baker (2006) treats literacy as a relatively stable individual characteristic measured through standardized instruments. Estimating it from 2-3 conversational turns is a fundamentally different and noisier task. If the estimator overestimates literacy, low-literacy users receive content they cannot understand — and the system fails silently because no standard evaluation metric catches it. ORACLE needs explicit uncertainty quantification on literacy estimates with conservative fallback behavior when confidence is low.

**Assumption 2 — MedQuAD and PLABA represent the patient literacy spectrum:**
MedQuAD covers 12 NIH websites written for a general health consumer audience. PLABA covers 75 health topics with expert adaptations. Both are professionally written patient-facing content — they represent what well-resourced health communication looks like, not what low-literacy patients actually encounter or can process. Low-literacy users may generate queries that do not match the vocabulary of either dataset. The retrieval could systematically fail to surface patient-appropriate content for precisely the patients who need it most.

**Assumption 3 — Upstream retrieval conditioning eliminates downstream simplification errors:**
PlainQAFact documents errors in post-hoc simplification. ORACLE assumes that retrieving audience-appropriate content reduces these errors. This is directionally correct but not guaranteed — MedQuAD answers are written for a general health consumer audience, not for a patient at a specific literacy level. There is still a generation step that may introduce errors for very low literacy users. The error reduction is real but the degree of reduction is an empirical question that Stage 4 must answer.

**Assumption 4 — PEFT adapters per literacy band scale appropriately:**
Three literacy bands with PEFT adapters introduces switching latency at inference time. On clinical hardware processing queries from patients in real time, adapter switching overhead could be unacceptable. This is a systems assumption embedded in a research architecture. Needs explicit latency benchmarking in Stage 4 before any claims about deployment feasibility.

**Assumption 5 — Three literacy bands are sufficient granularity:**
Nutbeam (2000) defines three levels — functional, communicative, critical. ORACLE adopts this classification. But health literacy exists on a continuous spectrum and the boundaries between bands are not clinically validated for RAG system design. A patient near the boundary between bands could receive systematically wrong content if misclassified. The three-band design is a practical choice, not a validated one — band boundaries need empirical calibration against comprehension outcomes in Stage 4.

**Assumption 6 — Comprehension outcome measurement is feasible at evaluation scale:**
ORACLE's Stage 4 evaluation includes downstream task success rate by literacy group. This requires either human evaluation or a proxy task for comprehension. Human evaluation does not scale to the full dataset. Proxy tasks — follow-up question answering, cloze tasks — are imperfect proxies for actual comprehension. The evaluation design needs to specify exactly what comprehension measurement looks like in practice and what its limitations are before Stage 4 begins.

---

## Protocol 8 — Knowledge Map

ORACLE Knowledge Map — June 2026
Verified: 534,149 records across 6 datasets

RAG ARCHITECTURE CLUSTER
Lewis et al. 2020 (NeurIPS) — Foundational RAG
  literacy-agnostic by design → ORACLE GAP 1
Karpukhin et al. 2020 (EMNLP) — DPR backbone
  ORACLE builds on DPR + adds literacy conditioning
Izacard & Grave 2021 (EACL) — FiD multi-passage
  multi-passage fusion → informs Stage 2 design

BIOMEDICAL BENCHMARK CLUSTER
Jin et al. 2019 — PubMedQA 273,518 records
Pal et al. 2022 — MedMCQA 193,155 records
Jin et al. 2021 — MedQA 11,451 records
Xiong et al. 2024 — MIRAGE 7,663 records
  best available RAG medical benchmark
  no accessibility evaluation → ORACLE GAP 2

PLAIN LANGUAGE CLUSTER (Guo et al. UIUC series)
APPLS 2024 — standard metrics fail on plain language
  ORACLE uses downstream task success rate
Jargon 2024 — personalization required
  ORACLE per-literacy-band PEFT adapters
PlainQAFact 2025 — simplification degrades factuality
  ORACLE upstream retrieval conditioning

PATIENT-FACING DATASET CLUSTER
MedQuAD 47,441 records (NIH, 12 websites)
  replaces Consumer Health QA — same NLM/NIH group
MIMIC-III pending PhysioNet
  discharge summaries — hardest accessibility challenge

PLAIN LANGUAGE EVALUATION CLUSTER
Attal et al. 2023 — PLABA 921 records
  75 health topics, gold standard NLM annotations
  primary plain language evaluation dataset

HEALTH LITERACY FRAMEWORK CLUSTER
Nutbeam 2000 — three-level literacy classification
  ORACLE literacy band design
Baker 2006 — literacy predicts health outcomes
  justifies comprehension outcome measurement
IOM 2004 — health literacy policy framework
  motivates production deployment framing

ORACLE CORE CONTRIBUTION
Literacy-conditioned dense retrieval
  Gap 1: first system to condition retrieval on literacy
  Gap 2: first RAG benchmark with accessibility evaluation
  Gap 3: architectural fix for simplification errors
  Gap 4: first NLP operationalization of Nutbeam framework
  Gap 5: end-to-end clinical-to-patient evaluation

---

## Protocol 9 — So What Test (3 Points in Plain English)

**Point 1 — The RAG literature has a blind spot that matters clinically:**
Every major RAG system — from Lewis et al. (2020) to MIRAGE (2024) — retrieves the same documents for all users. A nurse and a newly diagnosed patient asking the same question get the same retrieved content. This is not an edge case. 36% of US adults have basic or below basic health literacy (IOM 2004). The system being built for patients is failing a third of them by design. ORACLE addresses this by conditioning retrieval on who is asking, not just what they are asking.

**Point 2 — Simplification as post-processing is the wrong fix for the right problem:**
The plain language NLP community has spent years building better simplification models. PlainQAFact (2025) shows those models introduce factual errors. The problem is not that simplification models are inadequate — it is that rewriting content written for a different audience is architecturally the wrong approach. ORACLE moves literacy conditioning upstream into retrieval. The generation step starts from content already appropriate for the audience, eliminating the source of simplification errors rather than trying to reduce them.

**Point 3 — 534,149 verified records and no existing system evaluates accessibility across any of them:**
PubMedQA, MedMCQA, MedQA, and MIRAGE collectively represent the strongest available biomedical QA evaluation. None of them measure whether a patient at a specific literacy level understood the output. MedQuAD and PLABA represent real patient-facing content — 47,441 NIH QA pairs and 921 expert plain language adaptations. ORACLE is the first system that evaluates across all of these simultaneously with comprehension outcome measurement as the primary metric. The gap is not a niche corner case — it is the difference between evaluating whether the system knows medicine and evaluating whether the system helps patients.
