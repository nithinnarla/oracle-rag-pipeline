# ORACLE: Paper Draft
## JBI Submission Target Sep 22 2026

---

## 3. Methodology

### 3.1 Framework Overview

ORACLE addresses literacy-agnostic retrieval through a four-stage pipeline: document ingestion with literacy scoring, hybrid retrieval with literacy-band re-ranking, PEFT adapter generation per band, and comprehension-outcome evaluation. This architecture reflects a specific design choice: literacy conditioning happens upstream, at the retrieval stage, rather than downstream, as a post-processing simplification step applied after generation. You and Guo (2026) documented that post-hoc simplification introduces factual errors precisely because it operates on already-generated content without access to the original source material's full context; conditioning retrieval itself on literacy band avoids this failure mode by ensuring the system never retrieves content it will need to distort.

### 3.2 Retrieval Corpus

The retrieval corpus draws on five sources: MedMCQA, MedQA, MIRAGE, PubMedQA, and PLABA, totaling 36,664 records after processing. Table 1 summarizes the corpus composition by source.

| Source | Records | Corpus Share |
|---|---|---|
| MedMCQA | 15,732 | 42.9% |
| MedQA | 11,431 | 31.2% |
| MIRAGE | 7,580 | 20.7% |
| PubMedQA | 1,000 | 2.7% |
| PLABA | 921 | 2.5% |

**Table 1.** Corpus composition by source, five datasets totaling 36,664 records.

MedMCQA required explicit correction before inclusion. In its raw form, MedMCQA contributes 193,155 records, which would constitute 89.9% of the combined corpus and effectively crowd out every other source. To prevent this single-source domination, MedMCQA was capped to a target of 20,000 records via stratified sampling across its 21 medical subject categories, preserving subject-level diversity rather than sampling uniformly at random. This stratified step alone brings the source to approximately 20,000 records; a subsequent minimum-length quality filter, applied identically across all five sources, removes records with fewer than 20 characters of question text, reducing MedMCQA's final contribution to 15,732 records, 42.9% of the finished corpus. No single source exceeds half of the total, a deliberate outcome of the capping decision rather than an incidental property of the raw data.

MedQuAD, a genuinely patient-facing question-answer dataset with 47,441 records, was evaluated for inclusion and excluded. The publicly available HuggingFace version of MedQuAD has a substantial null-answer problem affecting the majority of records; a usable subset of approximately 16,407 records with genuine, non-null answers does exist within the dataset, but integrating that subset was not completed before this corpus was finalized. This exclusion is a real limitation of the current corpus, since MedQuAD is the one source in this space designed explicitly for patient-facing question answering rather than clinical assessment; its absence is addressed directly in Section 6.4 and logged as future work rather than treated as a settled decision.

### 3.3 Literacy Band Classification

Every document and query is scored on ingestion using Flesch-Kincaid grade level, then assigned to one of four discrete bands: low (FK ≤ 6, plain language), medium (FK 7-10, general public), high (FK 11-14, educated layperson), and clinical (FK 15+, clinical professional). This four-band structure operationalizes Nutbeam's (2000) three-level literacy framework at finer granularity.

FK is a known imperfect proxy: it measures surface features, sentence length and syllable count, rather than vocabulary difficulty or domain-specific jargon. This produces a documented, unmitigated failure mode in the corpus itself. MedMCQA, composed of short clinical exam-stem fragments, contributes 80.3% of all low-band-classified records despite containing dense medical terminology inappropriate for a genuinely low-literacy reader; PLABA, the one corpus source that is actual gold-standard plain-language writing, contributes zero low-band records. A classifier scoring surface complexity cannot distinguish a short, jargon-dense exam question from a genuinely accessible sentence of similar length.

### 3.4 Retrieval and Generation

Literacy conditioning is enforced at the retrieval stage through hard band selection, not post-hoc re-ranking. Each incoming query is first classified into a literacy band; retrieval then searches only the embedding pool for that band, so a query classified as low-literacy can only retrieve low-band documents, never a higher-band document later filtered or demoted. This is a stronger conditioning mechanism than re-ranking: literacy-inappropriate content is structurally unreachable at retrieval time rather than deprioritized after the fact. The system also supports an explicit band override, used to force retrieval from a specified band independent of the query's own classified band, supporting the literacy-correction evaluation described in Section 5.6.

Retrieval itself uses a DPR backbone (Karpukhin et al., 2020), with queries and documents encoded into a shared vector space and ranked by cosine similarity within the selected band's pool. Generation runs on gpt-4o-mini, selected for cost efficiency, a full evaluation set runs at approximately $2-3, and to reflect realistic production-deployment constraints, since a system this affordable to run is one an organization could plausibly deploy rather than one requiring research-grade compute budgets. Literacy conditioning at the generation stage is implemented through band-specific system prompts, a distinct system prompt template per band, combined with the retrieved documents to construct the final generation prompt. A parallel PEFT/LoRA adapter architecture was built and validated per band, but adapter training was simulated rather than run to convergence on real gradient updates; it is not the mechanism producing the results reported in Section 5, and is scoped accordingly in Section 6.4 as an architecture-validated direction for future work rather than a component of the current evaluated system.

### 3.5 Evaluation Metrics

Retrieval quality is evaluated via self-supervised retrieval: each record's own question is used as a query, and the record's own record_id serves as its ground-truth relevant document, since it is definitionally the source the question was drawn from. This design permits standard information-retrieval metrics, Precision@1, Recall@K, and Mean Reciprocal Rank, evaluated separately within each literacy band's embedding pool. This measures whether the system can recover a record's source given its own question; it is a distinct and narrower guarantee than Precision@K measured against independently human-annotated relevance judgments, since only one document is treated as relevant per query by construction. Band-routing accuracy on a fixed 20-query evaluation set (5 per band), source diversity of retrieved documents, and FK-grade alignment between query and retrieved content are reported alongside these metrics as further proxies for retrieval behavior.

Factual consistency is evaluated two ways: an adapted GPT-4o-mini method using source-abstract ground truth, and the official PlainQAFact metric (You & Guo, 2026), reported together with an explanation of what each measures rather than treated as interchangeable. Readability is measured via FK and SMOG, reported for comparability but not treated as the study's primary evidence on their own. Generation quality is measured via ROUGE-L and BERTScore. APPLS-based perturbation testing (Section 5.5), applied directly to ORACLE's own PLABA test split, empirically confirms that ROUGE-L and BERTScore are sensitive to the informativeness, coherence, and simplification transformations this pipeline cares about, while FK and SMOG are not and instead capture a distinct property.

Self-retrieval evaluation was run on a stratified sample of 300 records per literacy band (1,200 total), rather than the full 36,664-record corpus, for computational feasibility; per-query retrieval time scales with candidate pool size, and a full-corpus run across all four bands was estimated at approximately 2.7 hours given the pipeline's current unoptimized similarity search.

![Figure: Self-retrieval Precision@1, Recall@10, and MRR by literacy band, as-deployed pool sizes](../figures/stage2/self_retrieval_precision_by_band.png)

**Figure.** Self-retrieval evaluation by literacy band using each band's true, as-deployed candidate pool size (low=4,264; medium=14,798; high=12,281; clinical=5,321).

To isolate genuine retrieval-quality differences from candidate-pool-size effects, a second evaluation fixes the candidate pool to 4,264 records per query, matching the smallest band, for every band, always including each query's own true source record.

![Figure: Self-retrieval scores under a pool-size-controlled comparison](../figures/stage2/self_retrieval_controlled_by_band.png)

**Figure.** Pool-size-controlled self-retrieval evaluation (fixed candidate pool = 4,264 per query). Low band's advantage persists after controlling for pool size, indicating a genuine, not merely size-driven, retrieval-quality difference by band.

The central empirical claim of this evaluation is precisely scoped, not sweeping: correct literacy-band routing measurably improves readability outcomes (Section 5.3), and produces no measurable effect on retrieval relevance or generation faithfulness by the metrics available here. Comprehension-outcome measurement against genuine human task success is identified as necessary future work (Section 6.4) and was not attempted in the current evaluation.
