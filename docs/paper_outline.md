## Paper Outline, JBI Submission Target Sep 22 2026

---

## Abstract (250 words)
- RAG systems retrieve identically for all users regardless of literacy; a nurse and a newly diagnosed patient asking the same question get the same retrieved content
- No prior system conditions retrieval on user literacy; simplification is treated as post-hoc, which introduces factual errors (PlainQAFact, You & Guo 2025)
- ORACLE: literacy-conditioned dense retrieval, PEFT adapters per literacy band, comprehension-outcome evaluation over readability scores alone
- Evaluated across 5 corpus sources (36,664 records: MedMCQA, MedQA, MIRAGE, PubMedQA, PLABA)
- Key finding: correct literacy-band routing significantly improves generation readability (p=0.0012) but does not affect content-level metrics (ROUGE-L, BERTScore), a bounded, falsifiable claim about which stage of the pipeline literacy conditioning actually affects
- Factual consistency evaluated two ways: adapted GPT-4o-mini method and the official PlainQAFact metric; the two diverge substantially, traced to a genuine domain-mismatch in PlainQAFact's own knowledge base for clinical-trial-specific content, not a pipeline error
- APPLS-based empirical validation confirms ROUGE-L/BERTScore are sensitive to the transformations that matter for this pipeline; FK/SMOG are not, and serve a distinct, complementary role

---

## 1. Introduction (~800 words)

### 1.1 The Health Information Accessibility Gap
- Origin narrative: clinical NLP deployment for a healthcare client, retrieval accurate, system failing patients
- Four documented failure modes: readability mismatch, literacy drift across turns, factuality-accessibility inversion, post-processing as the wrong architecture
- Population affected extends beyond patients: family members navigating discharge instructions, non-native English speakers, high-stress caregivers

### 1.2 The Literacy-Agnostic Retrieval Problem
- Every major RAG system (Lewis et al. 2020, Karpukhin et al. 2020, Izacard & Grave 2021) retrieves on semantic similarity alone
- No notion of whether retrieved content is appropriate for the user's literacy level
- Simplification-as-post-processing introduces factual errors (You & Guo 2025); the right fix is upstream

### 1.3 ORACLE Contributions
- First system to condition dense retrieval on estimated literacy band
- First RAG evaluation to include accessibility/comprehension-outcome measurement alongside factual accuracy
- Empirical resolution of what literacy-band routing actually affects (readability, not content) versus what it does not

### 1.4 Paper Organization

---

## 2. Related Work (~600 words)

### 2.1 RAG Foundations
- Lewis et al. (2020): foundational RAG, literacy-agnostic by design
- Karpukhin et al. (2020): DPR backbone; ORACLE builds on this, adds literacy conditioning to the query encoder
- Izacard & Grave (2021): FiD multi-passage fusion; same literacy-agnostic limitation

### 2.2 Biomedical QA Benchmarks
- PubMedQA (Jin et al. 2019), MedMCQA (Pal et al. 2022), MedQA (Jin et al. 2021), MIRAGE (Xiong et al. 2024)
- All test clinical professional knowledge; none measure patient-facing comprehension
- MIRAGE is the strongest available RAG-specific medical benchmark but has no accessibility dimension

### 2.3 Plain Language and Health Literacy
- You & Guo (2025), PlainQAFact: documents factuality-accessibility inversion, anchors ORACLE's design
- Guo et al. (2024), Jargon: personalization required, not universal dictionaries; informs per-band PEFT design
- Guo et al. (2024), APPLS: standard metrics fail to predict comprehension; motivates comprehension-outcome measurement over readability scores alone
- Attal et al. (2023), PLABA: gold-standard plain language adaptations, ORACLE's primary evaluation dataset

### 2.4 Health Literacy Frameworks
- Nutbeam (2000): three-level literacy classification, informs ORACLE's band design (implemented as 4 discrete bands for finer granularity)
- Baker (2006): health literacy predicts outcomes independent of education/income
- Institute of Medicine (2004): 36% of US adults have basic or below-basic health literacy

---

## 3. Methodology (~1000 words)

### 3.1 Framework Overview
- Four-stage pipeline: document ingestion with literacy scoring, hybrid retrieval with literacy-band re-ranking, PEFT adapter generation per band, comprehension-outcome evaluation
- Post-processing simplification rejected in favor of upstream retrieval conditioning

### 3.2 Retrieval Corpus
- 36,664 records across 5 sources: MedMCQA (15,732), MedQA (11,431), MIRAGE (7,580), PubMedQA (1,000 expert-labeled subset), PLABA (921)
- MedMCQA capped from 193,155 (89.9% corpus share) to prevent single-source domination; final corpus balance 42.4% MedMCQA, no source above 50%
- MedQuAD (47,441 records, patient-facing) evaluated for inclusion but excluded: HuggingFace version's null-answer contamination affects 65.4% of records; a genuine 16,407-record usable subset exists but was not integrated in time, logged as future work
- See figures/corpus_composition.png for per-source breakdown

### 3.3 Literacy Band Classification
- Flesch-Kincaid grade level as primary literacy proxy, scored on ingestion
- Four discrete bands: low, medium, high, clinical
- Known limitation: FK measures surface features (sentence length, syllable count), not vocabulary difficulty; validated failure mode where jargon-heavy short text misclassifies as low-literacy

### 3.4 Retrieval and Generation
- DPR backbone (Karpukhin et al. 2020), chosen over BM25/hybrid/ColBERT for PEFT-adaptability per literacy band
- Literacy conditioning applied as re-ranking, not hard filtering, to preserve relevance
- PEFT adapter stack per literacy band rather than single fine-tuned model or continuous estimation
- gpt-4o-mini for Stage 4 generation, chosen for cost-efficiency and production-deployment realism

### 3.5 Evaluation Metrics
- Retrieval: standard IR metrics (Precision@K, Recall@K, MRR, NDCG), evaluated separately by literacy group
- Factual consistency: both an adapted GPT-4o-mini method and the official PlainQAFact metric (You & Guo 2025), reported together with an explanation of what each measures
- Readability: FK, SMOG, reported for comparability but not treated as primary evidence
- Generation quality: ROUGE-L, BERTScore
- APPLS-based empirical validation (Decision 15) confirms ROUGE-L/BERTScore are sensitive to informativeness/coherence/simplification perturbations on ORACLE's own data; FK/SMOG are not, and measure a distinct property
- Comprehension outcome: downstream task success rate via MedQuAD/PLABA QA pairs, the primary metric; readability scores are proxies, not the main claim

---

## 4. Experimental Setup (~400 words)

### 4.1 Implementation
- Python, DPR encoder, PEFT/LoRA adapters, gpt-4o-mini via OpenAI API
- Same PhysioNet credentials as FAPE for pending MIMIC-III access

### 4.2 Reproducibility
- Corpus construction, quality filtering (nan-string bug fix, duplicate removal), and all evaluation scripts committed
- Known API non-determinism at temperature=0 documented; qualitative patterns, not exact point estimates, are the reliable finding

---

## 5. Results (~1200 words)

### 5.1 Corpus Composition and Literacy Distribution
- Per-source breakdown, band distribution by source
- PLABA has zero "low"-band records despite being the one genuinely plain-language source; MedMCQA contributes 78% of "low"-band records despite being short clinical exam fragments, a documented, unmitigated failure mode of FK-based classification

### 5.2 Literacy-Band Routing Accuracy by Source
- Routing accuracy varies sharply by source (not uniform); reported per-source, not as a single blended percentage
- plaba and medqa show the highest sensitivity to full-text-vs-question-only FK scoring (58.3% and 42.0% flip rates respectively); mirage, medmcqa, pubmed, pubmedqa show minimal to zero sensitivity
- This limitation should be scoped to plaba/medqa specifically in the paper, not stated as a uniform corpus-wide risk
- See figures/stage4/cross_dataset_fk_by_band.png and cross_dataset_routing_impact.png

### 5.3 Routing Accuracy and Readability Outcomes
- Headline finding: correct band routing significantly improves FK reduction (independently recomputed directly from data/processed/cross_dataset_results.csv, Sep 7 2026: p=0.0012, n=181, wrong_mean=-4.157, upper_mean=-2.974); does not significantly change ROUGE-L or BERTScore
- Mechanistically expected: band-prompt controls surface style, which FK measures; retrieved content is fixed identically across compared conditions, so content metrics have no channel to be affected
- Bounded, falsifiable claim: routing governs readability outcomes specifically, not content relevance or faithfulness generally
- See figures/stage4/cross_dataset_misroute_significance.png

### 5.4 Factual Consistency Evaluation
- Adapted method (GPT-4o-mini, source-abstract ground truth): overall consistency ~0.96, simplification claims score higher than elaboration claims, consistent with PlainQAFact's own documented finding
- Official PlainQAFact (You & Guo 2025): overall ~0.33, a genuinely different measurement using external knowledge-base retrieval as ground truth
- Domain-mismatch confirmed at both mechanism level (direct retrieval inspection) and systematic level (234 claims): PlainQAFact's Textbooks knowledge base covers foundational medical education content, not clinical-trial-specific claims; StatPearls performs better for the same query type but does not fully compensate
- Both scores reported with this explanation; neither validates nor invalidates the other; they measure different things

### 5.5 APPLS-Based Metric Validation
- Three perturbation types applied to ORACLE's own PLABA test split (148 records): delete_sentence, coherent, simplification
- ROUGE-L and BERTScore strongly sensitive to all three (|r| > 0.6, p<0.001); FK shows no sensitivity to any
- Empirical, on-ORACLE's-own-data confirmation that the reported metric suite is appropriate, not just citation-based justification
- See figures/stage4/appls_metric_sensitivity.png

### 5.6 Limitations of the Literacy Proxy
- Full-text (question+answer) FK scoring changes routing correctness for roughly 1 in 6 queries versus question-only scoring, concentrated in plaba and medqa specifically
- Length mechanism confirmed directly (point-biserial r=0.519, p<0.0001): longer queries flip more often
- No evidence that misrouting produces worse generation quality by the two metrics tested; reported plainly, not interpreted as confirming harm
- See figures/stage4/fk_ablation_flip_rate_by_source.png and fk_ablation_length_correlation.png

---

## 6. Discussion (~600 words)

### 6.1 What Literacy Conditioning Actually Affects
- The p=0.0012 finding is precise about scope: routing affects readability output, not retrieval relevance or generation faithfulness
- This is a stronger, more defensible claim than an unbounded "routing accuracy matters"

### 6.2 Metric Suite Validation as a Contribution
- APPLS's own findings (no single metric captures all plain-language criteria) now have direct empirical support on ORACLE's own data, not just citation-based justification
- Reporting multiple metrics together, rather than one composite score, is methodologically necessary, not just cautious

### 6.3 Production Deployment Implications
- 93.6% of the retrieval corpus is clinical-professional content (MedMCQA, MedQA, MIRAGE combined); the paper must scope its claim accordingly: ORACLE improves retrieval quality across FK literacy levels, not necessarily for lay patient populations specifically
- FK's jargon-blindness (low-literacy band contaminated by short clinical fragments) is a real, unmitigated production risk, not a solved problem

### 6.4 Limitations
- FK as literacy proxy: not validated against actual patient comprehension; full_text scoring introduces a further, source-specific routing risk (Section 5.6)
- Literacy band estimated from short conversational turns is a fundamentally noisier task than the standardized-instrument literacy measurement it is based on (Baker 2006)
- MedQuAD, a genuinely patient-facing dataset, is excluded from the corpus; a usable 16,407-record subset exists but re-integration is future work, not attempted here
- PEFT adapter switching latency at inference time is not yet benchmarked; a systems assumption embedded in a research architecture
- Comprehension outcome measurement is simulated via QA-pair proxies, not human-subjects testing
- MIMIC-III access still pending PhysioNet credentialing at time of writing

---

## 7. Conclusion (~300 words)
- RAG's literacy-agnostic retrieval is a structural, not incidental, gap; ORACLE is the first system to condition retrieval on literacy rather than treating accessibility as post-processing
- The central empirical contribution is precise, not sweeping: literacy-band routing accuracy governs readability outcomes specifically, confirmed at p=0.0012, with no evidence of effect on content relevance or faithfulness
- Both adapted and official factual-consistency evaluation are reported, with the divergence between them itself a genuine, citable finding about knowledge-base domain-matching for clinical-trial-specific content
- Future work: MedQuAD re-integration, MIMIC-III clinical-to-patient evaluation once PhysioNet access clears, human-subjects comprehension validation

---

## References
- Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS
- Izacard & Grave (2021), Leveraging Passage Retrieval with Generative Models, EACL
- Karpukhin et al. (2020), Dense Passage Retrieval for Open-Domain QA, EMNLP
- Jin et al. (2019), PubMedQA: A Biomedical Research Question Answering Dataset, EMNLP
- Pal et al. (2022), MedMCQA: Large-Scale Multi-Subject Multi-Choice Medical QA, CHIL
- Jin et al. (2021), MedQA: USMLE Dataset, Applied Sciences
- Xiong et al. (2024), MIRAGE: Benchmarking RAG for Medicine, ACL Findings
- You & Guo (2025), PlainQAFact: Retrieval-Augmented Factual Consistency Evaluation for Biomedical Plain Language Summarization, arXiv
- Guo et al. (2024), Personalized Jargon Identification for Enhanced Interdisciplinary Communication, NAACL
- Guo et al. (2024), APPLS: Evaluating Evaluation Metrics for Plain Language Summarization, EMNLP
- Attal et al. (2023), PLABA: A Dataset for Plain Language Adaptation of Biomedical Abstracts, Scientific Data
- Ben Abacha & Demner-Fushman (2019), MedQuAD: A Manually Curated Question-Answer Dataset, BMC Bioinformatics
- Johnson et al. (2016), MIMIC-III Clinical Database, Scientific Data
- Nutbeam (2000), Health Literacy as a Public Health Goal, Health Promotion International
- Baker (2006), The Meaning and Measure of Health Literacy, Journal of General Internal Medicine
- Institute of Medicine (2004), Health Literacy: A Prescription to End Confusion, National Academies Press
