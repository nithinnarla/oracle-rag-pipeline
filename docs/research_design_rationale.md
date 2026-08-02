# ORACLE — Research Design Rationale
## Why We Built It This Way

**Period:** April 2026 — June 2026
**Researcher:** Nithin Narla
**Status:** Complete — major design decisions documented before Phase 4 implementation

---

## Why I'm Writing This Down

The same reason I wrote this for FAPE. Design decisions made in February look obvious in May and completely mysterious in October when I'm writing the paper and a reviewer asks why I conditioned retrieval on literacy rather than conditioning generation. I need to be able to answer that without reconstructing reasoning I should have captured when it was fresh.

The specific risk with ORACLE is that the architecture has three contested design choices that reviewers will push on: literacy-conditioned retrieval vs post-hoc simplification, PEFT adapter stacks per literacy band vs a single fine-tuned model, and the choice of evaluation metrics. Each of those decisions has a specific justification rooted in documented failures I have seen in deployment. This document is where those justifications live.

---

## The Research Question — Why Comparative and Not Something Easier

The easy version of this question is Descriptive: document readability gaps in existing biomedical RAG systems. That is a real contribution but it does not tell practitioners what to do about it. Guo et al.'s PlainQAFact (2025) already documented the factuality-accessibility inversion problem. A paper that only confirms what Guo found is not a paper worth writing.

The Comparative framing is the right choice: does literacy-conditioned RAG outperform standard RAG on comprehension outcomes across literacy levels? The comparison is explicit — ORACLE vs. standard RAG baseline — evaluated not just on retrieval quality but on whether users at different literacy levels actually understand the output. That question has a direct answer and direct implications for system design.

The honest constraint this framing imposes: I need a controlled comparison, which means the standard RAG baseline has to use identical retrieval infrastructure, identical generation model, identical evaluation pipeline — the only difference is whether retrieval is conditioned on literacy. Any other difference becomes a confound. That constraint shapes every implementation decision downstream.

---

## The 4-Stage Pipeline — The Decision Behind Each Stage

**Stage 1 — Why document ingestion with literacy scoring at ingestion time**

The standard approach is to retrieve first, then assess readability of retrieved content. I rejected this because it means the retrieval step has already committed to a document that may be inappropriate for the user — and then the system has to either serve it anyway or discard it and retrieve again.

Literacy scoring at ingestion time means every document in the corpus has a readability label before any query arrives. When a query comes in with an estimated literacy band, the retrieval step can weight candidates by both semantic similarity and readability appropriateness simultaneously. This is architecturally different from post-hoc simplification and it is why ORACLE avoids the factuality-accessibility inversion that Guo et al. (2025) documented — simpler documents are retrieved rather than complex documents rewritten.

The implementation cost is front-loaded: scoring 35M+ PubMed abstracts at ingestion time is expensive. The payoff is cleaner retrieval semantics at query time and a corpus that can be queried by literacy band directly.

**Stage 2 — Why hybrid BM25 + dense retrieval and not dense alone**

DPR alone has a known failure mode on short health queries from low-literacy users. Karpukhin et al.'s (2020) DPR was evaluated on NaturalQuestions and TriviaQA — both written by people who can construct detailed queries. Low-literacy health queries are shorter, vaguer, and use lay terminology rather than medical vocabulary. Dense retrieval alone performs poorly on these queries because there is not enough semantic signal.

BM25 handles exact-match lay terminology well. Dense retrieval handles semantic similarity for longer queries well. The hybrid fusion gets the best of both. FAISS handles the scale — 35M documents at query time requires approximate nearest neighbor search, not exhaustive search.

The literacy conditioning is applied as a re-ranking step after initial retrieval — not as a hard filter. Hard filtering would eliminate documents that are slightly above target reading level but otherwise highly relevant. Re-ranking preserves relevance while shifting the distribution toward appropriate literacy level.

**Stage 3 — Why PEFT adapter stacks per literacy band and not a single fine-tuned model**

A single fine-tuned model for plain language generation learns to produce one style. The problem is that appropriate literacy level is not one style — it is a range from SMOG Grade 6 for low-literacy patients to SMOG Grade 14 for clinicians reviewing discharge summaries. A single model optimized for plain language produces accessible outputs for low-literacy users and inappropriate outputs for clinical users, or vice versa.

PEFT adapter stacks — one adapter per literacy band — allow the base model's general language understanding to be preserved while each adapter specializes for one literacy register. The adapters are small, less than 1% of model parameters, so the computational cost is manageable. The key advantage is that swapping adapters at inference time costs microseconds, making real-time literacy adaptation feasible.

The honest limitation: adapter stacks require labeled examples from each literacy band. For clinical-level text I have MIMIC-III discharge summaries. For patient-level text I have MedQuAD and PLABA. For intermediate levels the training data is sparse. The paper will report performance separately by literacy band and acknowledge where training data was limited.

**Stage 4 — Why comprehension outcome measurement and not just readability scores**

Flesch-Kincaid and SMOG are proxy metrics. They measure surface features of text — sentence length, syllable count — as proxies for actual comprehension difficulty. Guo et al. (2025) showed that readability scores remain stable after model updates while actual user comprehension drops. If ORACLE's evaluation relies only on FK and SMOG, it will miss exactly the failure mode it was designed to address.

PlainQAFact measures factual consistency specifically in plain language outputs — catching the hallucinations introduced by simplification. APPLS measures plain language quality across multiple dimensions rather than just readability. Downstream task success rate — can a user at a specific literacy level correctly answer comprehension questions about the output — is the ultimate ground truth metric.

The honest constraint: measuring actual comprehension requires human subjects. I do not have access to a patient panel. Comprehension outcome measurement in ORACLE is simulated using MedQuAD and PLABA question-answer pairs as proxies — the assumption is that a correct answer on a follow-up question indicates comprehension of the retrieved content. This is a proxy, not a direct measurement, and the paper will say so explicitly.

---

## Dataset Selection — The Reasoning Behind Each Choice

**PubMedQA (273,518 records verified) — Biomedical QA retrieval base**

Jin et al. (2019) built the standard biomedical QA benchmark. Using it establishes comparability with prior work. The yes/no/maybe answer format is a limitation for ORACLE's use case — patients do not ask yes/no questions — but PubMedQA provides the retrieval ground truth that MIRAGE uses for evaluation, so including it is necessary for the MIRAGE benchmark to work correctly.

**MedMCQA (193,155 records verified) — Clinical QA benchmark**

Pal et al. (2022) covers 2,400+ healthcare topics from Indian medical entrance exams. The scale and topic coverage make it the best available source for evaluating whether ORACLE's retrieval generalizes across medical domains. The limitation — exam questions are written for medical students, not patients — means MedMCQA is a retrieval quality benchmark for ORACLE, not a comprehension benchmark.

**MedQA USMLE (11,451 records verified) — Clinical reasoning benchmark**

The USMLE clinical vignette format is the hardest retrieval challenge in the benchmark suite — long multi-paragraph questions, multiple plausible answer choices, clinical reasoning required. Including MedQA establishes that ORACLE's retrieval can handle complex clinical queries even though its primary use case is patient-facing accessibility. A system that improves accessibility while degrading clinical query performance is not a useful system.

**MIRAGE (7,663 records verified) — RAG-specific evaluation benchmark**

Xiong et al. (2024) built the only evaluation benchmark designed specifically for medical RAG systems. MIRAGE provides the retrieval ground truth — PMID references for correct sources — that lets me measure retrieval precision and recall directly rather than inferring them from downstream QA performance. This is the benchmark that will appear in ORACLE's primary results table.

**MedQuAD (47,441 records verified) — Patient-facing health information**

Ben Abacha and Demner-Fushman (2019) built MedQuAD from 12 NIH websites specifically designed for patient-facing health information. This is the closest existing dataset to ORACLE's actual use case — real patient questions about real health topics, answers written for a lay audience by medical professionals. The 65.4% structural answer missingness documented in EDA does not affect the 16,407 RAG-usable records with complete question-answer pairs.

**PLABA (921 records verified) — Plain language adaptation gold standard**

Attal et al. (2023). The only dataset with human-expert plain language adaptations of biomedical abstracts paired with the original clinical text. The EDA findings are the Stage 4 generation targets: FK grade drops from 14.9 to 12.6, FRE improves by 16.1 points, plain language versions use slightly more words (ratio 1.064). PLABA is small but it is the only expert-validated plain language benchmark that exists.

**MIMIC-III (pending PhysioNet credentialed access) — Clinical discharge summaries**

Johnson et al. (2016). The clinical-to-patient translation problem in its purest form — documents written by clinicians for clinical handoff that patients and family members must navigate after discharge. MIMIC-III is the dataset that makes that translation gap measurable in ORACLE's evaluation. Access pending PhysioNet credentialed registration.

---

## Evaluation Metric Selection — Why Each Metric Is in the Paper

**Retrieval metrics by literacy group**

Standard IR metrics — Precision@K, Recall@K, MRR, NDCG — but evaluated separately by literacy group. Aggregate retrieval metrics hide the disparity ORACLE is designed to fix.

**Readability metrics (Flesch-Kincaid, SMOG)**

Included because reviewers expect them and because they provide a surface-level sanity check. Not relied on as primary evidence. Reported alongside comprehension metrics with explicit acknowledgment that they are proxies.

**PlainQAFact factual consistency score**

Guo et al. (2025). Built specifically to catch hallucinations introduced by plain language simplification. Without PlainQAFact, ORACLE has no way to demonstrate that literacy-conditioned retrieval avoids the factuality-accessibility inversion that post-hoc simplification produces.

**APPLS plain language evaluation metrics**

Guo et al. (2024) EMNLP. Multi-dimensional plain language quality assessment beyond readability scores. Measures whether the output actually follows plain language principles — active voice, concrete examples, defined terms — not just whether it is short enough.

**Downstream task success rate by literacy group**

The ultimate ground truth. Simulated via MedQuAD and PLABA QA pairs. It is a proxy for real comprehension measurement but it is the closest available approximation without a human subjects study.

---

## Architecture Decisions I Considered and Rejected

**Post-hoc simplification instead of literacy-conditioned retrieval**

Every existing system does this. Guo et al. (2025) documented why it fails. I am not building another system that does post-hoc simplification and hoping the failure mode does not appear in evaluation. The architecture choice is the paper's core contribution.

**Single literacy band instead of discrete band classification**

Continuous literacy estimation is more theoretically elegant but operationally fragile. Discrete bands — low, intermediate, high, clinical — are coarser but robust to estimation noise and map directly to the PEFT adapter stack architecture. Band boundaries are informed by Nutbeam (2000) and Baker (2006) health literacy frameworks.

**BERT-based retrieval instead of BioSentVec + sentence-transformers hybrid**

PubMedBERT or BioBERT would be reasonable choices. I chose BioSentVec for biomedical semantic similarity and sentence-transformers for hybrid dense retrieval because the combination has been validated on MIRAGE specifically. Comparability with prior work on MIRAGE is worth more than novelty in retrieval architecture.

---

## What I Got Wrong and Corrected During EDA

**MedQuAD structural missingness**

Expected MedQuAD to be a clean dataset from NIH. The EDA found 65.4% structural answer missingness across ADAM, MPlusDrugs, and MPlusHerbsSupplements categories. The loader now filters to complete question-answer pairs before any downstream processing.

**PLABA compression ratio near 1.0**

Expected plain language adaptations to be significantly shorter. The EDA found compression ratio approximately 1.0 and word ratio 1.064 — plain language versions are actually slightly longer. Stage 4 generation target adjusted: optimizing for lower FK grade while maintaining or slightly increasing length is the correct objective, not optimizing for shorter output.

**MIRAGE answer format heterogeneity**

The EDA found MIRAGE's 5 source datasets use incompatible answer formats — binary, three-way, four-way. Stage 4 evaluation uses source-specific accuracy metrics for each MIRAGE subset rather than aggregate accuracy.

---

*This document was written after completing all 6 dataset EDAs and before beginning Stage 1 pipeline implementation. Design decisions documented here reflect what the literature and EDA findings actually support.*

---

## Corrections During Stage 1 Implementation

**MedMCQA corpus domination**

Did not anticipate that MedMCQA (182,822 raw records) would outscale every other source by an order of magnitude once combined into a single retrieval corpus. Uncapped, MedMCQA was 89.9% of the corpus (189,366 of 210,731 records) -- a health-literacy RAG paper whose retrieval index is 90% medical entrance-exam questions undermines the patient-facing framing this project claims. Capped MedMCQA to 20,000 records, stratified by subject_name to preserve all 21 medical subjects proportionally. Post-cap: MedMCQA 42.4% of corpus, no single source above 50%.

**Answer field quality -- literal null-string bug**

MedMCQA's answer field (mapped from the exp/explanation column, since the correct-option index is not present in the loader output) contained records where exp was missing. The code used row.get("exp", "") to read it, which only returns the default value when the key itself is absent from the row -- if the key exists but the value is NaN, str(nan) produces the literal three-character string "nan", which passed initial null/empty checks undetected. 2,306 records had the literal string "nan" as their stored answer before this was caught. Quality filter now explicitly checks for the string tokens "nan" and "none" in addition to true null/empty. Combined with dropping records with unscoreable literacy (no FK grade assignable), final corpus after quality filtering: 37,096 records, zero null answers, zero unassigned literacy bands.

**Retrieval content for exam-format sources (MedQA, MedMCQA, MIRAGE) -- open architectural risk**

UPDATE: MedQA's missing-answer problem was a loader oversight, not missing data -- the loader returns answer_idx in a separate "labels" key that the preprocessor never read. Fixed: answer_idx is now pulled and mapped to resolved option text, same pattern as MIRAGE. MedQA answers are now genuinely resolved (verified: mean 3.6 words, e.g. "Haemophilus ducreyi", "Administer desmopressin" -- not option dumps).

Remaining question is narrower than originally stated: MedMCQA and MIRAGE both HAVE resolved answer content in their answer field (explanation text and mapped option letter respectively) -- the open question is only whether full_text (the actual retrieval unit passed to the encoder) should bake that resolved answer in, or stay as question+options with the answer held separately. Currently full_text for MedQA/MedMCQA/MIRAGE is question+options only; the answer field exists correctly but full_text does not include it. This is a full_text construction decision, not a missing-data problem for any of the three sources anymore. OPEN RISK, not yet resolved: if Stage 3/4 generation requires the resolved answer to be present in the retrieved full_text (rather than pulled separately from the answer column) to perform literacy simplification, the current full_text construction does not support that for these three sources -- medmcqa (42.4%) + medqa (30.8%) + mirage (20.4%) = 93.6% of the corpus by source, all three using question+options-only full_text. Must be validated when Stage 3 generation pipeline is designed -- do not assume this is settled.

**Flesch-Kincaid as literacy proxy -- confirmed failure mode, not yet mitigated**

Checked literacy_band distribution by source after the quality filter. PLABA -- the one source that is explicitly plain-language-adapted health text -- has zero records classified as "low" literacy band (0 of 921). PubMed abstracts: zero. PubMedQA: one. Meanwhile MedMCQA contributes 3,422 of 4,365 total "low" band records (78%) -- these are short exam-question stems ("All of the following are surgical options for X, Except"), not accessible patient-facing text. FK grade measures sentence length and syllable count, not vocabulary difficulty or domain jargon; a 10-word question using clinical terminology scores as "low" grade by FK while being inaccessible to an actual low-literacy reader. This means the corpus currently cannot reliably serve genuinely low-literacy retrieval requests -- it serves short clinical fragments instead. Not fixed in Stage 1. Needs either a supplementary readability signal (e.g. medical jargon density, SMOG index cross-check) or acceptance as a stated limitation before Stage 4 evaluation claims literacy-band-conditioned retrieval is working as designed.

**Aug 1 2026 update -- checked live, still open:** Ran Stage 4 generation_pipeline.py tonight. The misrouted low-band query ('What is diabetes and how does it affect the body?', routed to medium band by the FK classifier) retrieved 5 documents, all from MedQA/MIRAGE -- zero from PLABA -- confirming this contamination is not theoretical. One retrieved document was entirely off-topic (a sports-psychology exam question), retrieved only because it scored highest within the wrong band's index. Generation-level conditioning still produced a low-FK response because the system prompt conditions on target band, not retrieved content -- but this means the FK-grade result reported for Stage 4 is a generation-only result and does not demonstrate working retrieval-level conditioning. Fix or explicit limitation statement still required before any paper claim about retrieval-level literacy conditioning. Not resolved tonight; carrying forward to the Aug 13 pre-paper-writing audit.
