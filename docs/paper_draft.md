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

Factual consistency is evaluated two ways: an adapted GPT-4o-mini method using source-abstract ground truth, and the official PlainQAFact metric (You & Guo, 2026), reported together with an explanation of what each measures rather than treated as interchangeable. Readability is measured via FK and SMOG, reported for comparability but not treated as the study's primary evidence on their own. Generation quality is measured via ROUGE-L and BERTScore. APPLS-based perturbation testing (Section 5.5), applied directly to ORACLE's own PLABA test split, empirically confirms that ROUGE-L and BERTScore are strongly sensitive to the informativeness, coherence, and simplification transformations this pipeline cares about; FK shows small, only statistically-significant-at-scale sensitivity to the same transformations and does not meet the threshold this evaluation treats as meaningful, capturing a substantially weaker signal than ROUGE-L or BERTScore for these transformations.

The central empirical claim of this evaluation is precisely scoped, not sweeping: correct literacy-band routing measurably improves readability outcomes (Section 5.3), and produces no measurable effect on retrieval relevance or generation faithfulness by the metrics available here. Comprehension-outcome measurement against genuine human task success is identified as necessary future work (Section 6.4) and was not attempted in the current evaluation.


## 5. Results

### 5.1 Corpus Composition and Literacy Distribution

Table 2 summarizes the corpus's literacy-band distribution.

| Band | Records | Corpus Share |
|---|---|---|
| Medium | 14,798 | 40.4% |
| High | 12,281 | 33.5% |
| Clinical | 5,321 | 14.5% |
| Low | 4,264 | 11.6% |

**Table 2.** Literacy-band distribution across the 36,664-record corpus.

The low band's composition reveals a documented, unmitigated failure mode of FK-based classification. MedMCQA, composed of short clinical exam-stem fragments, contributes 80.3% of all low-band-classified records despite containing dense medical terminology inappropriate for a genuinely low-literacy reader. PLABA, the corpus's one source of actual gold-standard plain-language writing, contributes zero low-band records; its content style does not register as "low" under a metric scoring only sentence length and syllable count.

![Figure 1: Source distribution across the corpus](../figures/stage2/eval_source_distribution.png)

**Figure 1.** Distribution of corpus records by source.

![Figure 2: FK grade distribution across the corpus](../figures/stage2/eval_fk_distribution.png)

**Figure 2.** Distribution of Flesch-Kincaid grade scores across all corpus records, the basis for literacy-band assignment.

### 5.2 Literacy-Band Routing Accuracy by Source

Routing accuracy, whether a query's estimated literacy band correctly matches its true source band, varies sharply by source rather than holding uniform across the corpus. On a 523-query evaluation set: MedMCQA and PLABA route correctly 100% of the time; MedQA routes correctly 73.0%; PubMedQA 44.7%; PubMed 42.7%; Mirage only 32.0%, the weakest of any source. This spread is large enough that a single blended routing-accuracy figure would misrepresent every individual source; Section 6 returns to what distinguishes the high- and low-accuracy sources.

Self-retrieval evaluation, using each record's own question as its query and its own record_id as ground truth, was run on a stratified sample of 300 records per literacy band (1,200 total) rather than the full corpus, for computational feasibility; a full-corpus run was estimated at approximately 2.7 hours given the pipeline's current unoptimized similarity search.

![Figure 3: Self-retrieval Precision@1, Recall@10, and MRR by literacy band, as-deployed pool sizes](../figures/stage2/self_retrieval_precision_by_band.png)

**Figure 3.** Self-retrieval evaluation by literacy band using each band's true, as-deployed candidate pool size (low=4,264; medium=14,798; high=12,281; clinical=5,321). Low band shows the strongest scores, but its candidate pool is also the smallest.

To isolate genuine retrieval-quality differences from candidate-pool-size effects, a second evaluation fixes the candidate pool to 4,264 records per query for every band, always including each query's own true source record.

![Figure 4: Self-retrieval scores under a pool-size-controlled comparison](../figures/stage2/self_retrieval_controlled_by_band.png)

**Figure 4.** Pool-size-controlled self-retrieval evaluation (fixed candidate pool = 4,264 per query). Low band's advantage persists after controlling for pool size (Precision@1 0.843 versus 0.687-0.727 for the other three bands), indicating a genuine, not merely size-driven, retrieval-quality difference by band. A natural hypothesis that MedMCQA's short, distinctive questions (79.7% of the low band) drive this advantage does not hold: MedMCQA's own within-band Precision@1 (0.816) falls slightly below the band average, while Mirage, a minority source in this band (19.3%), retrieves at 0.966. The low band's aggregate advantage is not explained by its dominant source; it is better explained, provisionally, by Mirage's unusually strong retrievability specifically, a pattern this evaluation surfaces but does not yet explain.

### 5.3 Routing Accuracy and Readability Outcomes

Correct literacy-band routing significantly improves generation readability. Comparing paired wrong-band versus correct-band ("upper bound") generation on identical retrieved context, FK reduction differs significantly by routing correctness (Wilcoxon, n=177, wrong-band mean=-4.102, correct-band mean=-2.969, p=0.0018). ROUGE-L (n=180, p=0.640) and BERTScore (n=180, p=0.937) show no significant difference between conditions. This is mechanistically expected: retrieved content is held identical across both conditions by design, so only the band-conditioned generation prompt differs; content-level metrics have no channel through which routing correctness could affect them, while readability, shaped directly by the prompt's literacy-band instruction, does.

![Figure 5: Wrong-band versus correct-band generation, paired comparison across three metrics](../figures/stage4/cross_dataset_routing_impact.png)

**Figure 5.** Wrong-band versus upper-bound (correct-band) generation on identical retrieved context, compared across FK reduction, ROUGE-L, and BERTScore.

![Figure 6: Misroute significance summary table](../figures/stage4/cross_dataset_misroute_significance.png)

**Figure 6.** Wilcoxon significance summary for the wrong-band versus correct-band comparison across all three metrics.

### 5.4 Factual Consistency Evaluation

Factual consistency is evaluated two ways on a 20-record test set. An adapted method, using each record's source abstract as ground truth and GPT-4o-mini as the scoring model, reports an overall consistency of 0.969. Consistent with PlainQAFact's own documented finding, claims classified as simplification score higher (0.983) than claims classified as elaboration (0.870), indicating the generation process preserves factual content more reliably when compressing information than when adding detail beyond the source.

The official PlainQAFact metric (You and Guo, 2026), which uses external knowledge-base retrieval rather than the source abstract as its ground truth, was run separately and reports a substantially different overall score of approximately 0.33 (external-knowledge-base mean approximately 0.26). This official evaluation requires a locally hosted Llama 3.1 8B Instruct model on 40GB or more of CUDA-capable GPU memory; it was run in a separate environment meeting that requirement, since the primary development machine (Apple Silicon, no CUDA) cannot run it directly. The two scores are not measuring the same thing and are not in tension: the adapted method asks whether generated content is consistent with its own cited source, while the official metric asks whether it is consistent with an external medical knowledge base, and a domain mismatch between that knowledge base's coverage (general medical education content) and this corpus's clinical-trial-specific claims plausibly explains much of the gap. Both scores are reported for this reason, rather than treating either as the single correct measurement.

![Figure 7: Adapted factual consistency scores by claim type](../figures/stage4/factual_consistency_by_claim_type.png)

**Figure 7.** Adapted-method factual consistency scores, simplification versus elaboration claims.

![Figure 8: Official PlainQAFact scores by claim type](../figures/stage4/official_plainqafact_by_claim_type.png)

**Figure 8.** Official PlainQAFact factual consistency scores, simplification versus elaboration claims.

### 5.5 APPLS-Based Metric Validation

Three perturbation types, informativeness (delete_sentence), coherence, and simplification, were applied to ORACLE's own data and each metric's correlation with the perturbation's severity was measured. ROUGE-L and BERTScore show large, consistent sensitivity across all three perturbation types (|r| = 0.63-0.98, p<0.001 in every case). FK's correlations are statistically significant in all three tests as well (p<0.003 in every case), a product of the large sample sizes involved (n=1,386-1,613), but its effect sizes are small (|r| = 0.08-0.28) and fall below the 0.3 threshold this evaluation treats as meaningful sensitivity, while ROUGE-L and BERTScore clear that threshold by a wide margin in every test. FK is therefore better described as largely, not completely, insensitive to these perturbations: technically detectable given enough data, but not sensitive enough to serve as a reliable signal for the transformations this pipeline cares about, in contrast to ROUGE-L and BERTScore's unambiguous sensitivity.

![Figure 9: ROUGE-L and BERTScore sensitivity to APPLS perturbations](../figures/stage4/appls_metric_sensitivity.png)

**Figure 9.** ROUGE-L and BERTScore correlation with APPLS perturbation severity across all three perturbation types.

![Figure 10: FK grade sensitivity to APPLS perturbations](../figures/stage4/appls_fk_sensitivity.png)

**Figure 10.** FK grade correlation with APPLS perturbation severity, shown separately to illustrate its substantially weaker sensitivity relative to Figure 9.

### 5.6 Limitations of the Literacy Proxy

Full-text (question plus answer) FK scoring changes routing correctness for 17.4% of queries overall compared to question-only scoring, but this overall figure obscures a sharply uneven distribution: PLABA (58.3%) and MedQA (42.0%) are far more sensitive to this scoring choice than MedMCQA (1.0%), Mirage (6.0%), PubMed (0.0%), or PubMedQA (0.0%). This limitation should be scoped to PLABA and MedQA specifically, not stated as a corpus-wide risk. The mechanism is confirmed directly: query length correlates with flip likelihood (point-biserial r=0.519, p<0.00001, n=523), so longer queries are substantially more likely to have their routing decision change under full-text versus question-only scoring. No evidence in this evaluation indicates that misrouted queries produce worse generation quality by the content metrics tested (Section 5.3); this is reported plainly as an absence of detected harm, not as confirmation that misrouting is harmless.

![Figure 11: Routing flip rate by source, full-text versus question-only FK scoring](../figures/stage4/fk_ablation_flip_rate_by_source.png)

**Figure 11.** Proportion of queries whose routing decision changes between full-text and question-only FK scoring, by source. PLABA and MedQA are substantially more sensitive to this scoring choice than the other four sources.

![Figure 12: Query length versus routing-flip likelihood](../figures/stage4/fk_ablation_length_correlation.png)

**Figure 12.** Relationship between query word count and the likelihood that full-text scoring flips the routing decision relative to question-only scoring (point-biserial r=0.519, p<0.00001).
