# ORACLE, Methodology Decisions Log
## Literacy-Conditioned Health RAG Pipeline, Methodology Decisions

---

## How to Read This Document

This is a decisions log, not a polished writeup. Every major methodological choice is documented here with the alternatives I considered and why I made the call I made. Some decisions I am confident about. A few remain uncertain, those are marked with a note.

The point of documenting decisions before writing code is to prevent the most common research mistake: making a decision implicitly during implementation and then justifying it post-hoc in the paper. Every decision here was made before Stage 1 pipeline implementation begins. If Stage 1-4 results contradict a decision, the decision gets updated, but the reasoning trail stays visible.

**Status:** Complete, decisions locked before Stage 1 pipeline implementation begins.

---

## Decision 1, Flesch-Kincaid as Primary Literacy Proxy

**Decision:** Use Flesch-Kincaid Grade Level (FK) as the primary readability proxy for literacy band assignment across the ORACLE retrieval corpus.

**Alternatives considered:**
- SMOG Index: better for medical content, requires 30+ sentences, most QA records too short
- Flesch Reading Ease (FRE): inverse of FK, same underlying formula, adds no information
- Coleman-Liau: character-based, handles medical jargon differently, less validated than FK for health content
- Human annotation: gold standard but infeasible at corpus scale (37,076 records)
- LLM-based readability scoring: GPT-4 can assess readability but adds API cost and latency to ingestion

**Why FK:**
FK is the most widely validated readability formula in health communication research. The National Institutes of Health recommends patient materials target FK Grade 6-8. FK requires only sentence length and syllable count, computable at ingestion time with zero additional infrastructure. textstat library provides reliable FK scoring with no external dependencies.

**Uncertainty note, documented open risk:** FK measures sentence length and syllable count, not vocabulary difficulty or conceptual density. A sentence using "myocardial infarction" scores the same as one using "heart attack" if both have the same length and syllable profile. FK will systematically misclassify medical jargon-heavy content as high-literacy when it may actually be low-comprehension for lay readers. This is the primary limitation of ORACLE's literacy conditioning approach and must be stated explicitly in the paper's limitations section. A validation step comparing FK bands to human readability judgments is deferred to Stage 4 evaluation.

---

## Decision 2, MedMCQA Capped at 20,000 Records Stratified by Subject

**Decision:** Cap MedMCQA from 193,155 records to 20,000 records, stratified by subject_name, before inclusion in the retrieval corpus.

**Alternatives considered:**
- Full 193,155 records: MedMCQA becomes 89.9% of the retrieval corpus, single-source domination
- Random 20,000 sample: loses subject distribution, some medical specialties over/underrepresented
- 50,000 stratified: reduces domination to ~60%, still too high
- Exclude MedMCQA entirely: loses 21 medical subjects, largest clinical coverage dataset available

**Why 20,000 stratified:**
At 193,155 records MedMCQA was 89.9% of the retrieval corpus before capping. A paper framed as literacy-conditioned health RAG for patients whose retrieval index is 90% medical-entrance-exam trivia contradicts its own premise. Capping at 20,000 stratified by subject reduces MedMCQA from 89.9% to 42.4% of the corpus, majority but not dominating. Stratification preserves subject coverage while controlling corpus balance. Verified: final corpus 37,076 records across 6 sources, zero null answers, zero unknown FK, zero duplicates.

---

## Decision 3, FK Scored on full_text, Not Question Alone

**Decision:** Score FK readability on the full_text field (question + answer concatenated) rather than on the question text alone.

**Alternatives considered:**
- Question-only FK scoring: captures reading difficulty of the prompt but ignores answer complexity
- Answer-only FK scoring: captures response complexity but ignores how the question is framed
- Separate FK scores for question and answer: doubles the feature set, complicates band assignment

**Why full_text:**
ORACLE's retrieval context is the full QA pair, both question and answer are returned to the user. Scoring FK on question alone would assign literacy bands based on prompt complexity while ignoring answer complexity. A simple question with a jargon-heavy answer would be incorrectly assigned to a low-literacy band. Scoring full_text ensures the literacy band reflects the complexity of the complete retrieved content the user receives. Consistent with MIRAGE benchmark's answer format, which concatenates question and answer options.

**Open risk:** FK on concatenated text is not validated as a patient literacy measure in the health communication literature. A 3-word answer appended to a 50-word question changes the FK score in ways that may not reflect actual reading difficulty. This limitation is documented and accepted for Stage 1. Stage 4 ablation will compare full_text FK versus question-only FK on retrieval quality metrics.

---

## Decision 4, MedQuAD Excluded from Retrieval Corpus

**Decision:** Exclude MedQuAD from the ORACLE retrieval corpus despite being a patient-facing health QA dataset.

**Alternatives considered:**
- Include MedQuAD via HuggingFace: HuggingFace version is questions-only, no answers available
- Re-parse from original XML: abachaa/MedQuAD GitHub contains full XML with answers, requires custom parser
- Include partial MedQuAD with null answers: introduces known null contamination into corpus

**Why excluded:**
The HuggingFace MedQuAD dataset (hf-datasets version) contains only questions with null answer fields, 47,441 records with no retrievable content. Including null-answer records would contaminate the retrieval corpus with empty context. Re-parsing from the original XML at abachaa/MedQuAD is the correct approach but was not implemented before corpus finalization. MedQuAD re-addition is documented as a future work item, if original XML is parsed before Stage 3, it will be added with a corpus update commit.

---

## Decision 5, DPR Over BM25 or Hybrid Retrieval

**Decision:** Use Dense Passage Retrieval (DPR) as the primary retrieval backbone for Stage 1, with BM25 as a baseline comparison.

**Alternatives considered:**
- BM25 only: sparse retrieval, no semantic understanding, fails on paraphrase and medical synonyms
- Hybrid BM25 + DPR: stronger than either alone but doubles infrastructure complexity
- ColBERT: late-interaction model, stronger than DPR on many benchmarks but computationally expensive at inference
- RAG with frozen retriever: simpler but cannot be adapted per literacy band

**Why DPR:**
DPR encodes semantic meaning rather than keyword overlap, critical for health QA where "heart attack" and "myocardial infarction" must retrieve the same documents. DPR's encoder architecture supports fine-tuning per literacy band via PEFT adapters, the core Stage 2 intervention. BM25 cannot be adapted per literacy band. Hybrid retrieval adds infrastructure complexity without a clear hypothesis for how it interacts with literacy conditioning. MIRAGE benchmark uses DPR-based retrieval as its primary setting, using DPR maintains direct comparability with ORACLE's primary evaluation framework.

---

## Decision 6, PEFT Adapters Per Literacy Band Over Single Fine-Tune

**Decision:** Apply PEFT (Parameter Efficient Fine-Tuning) adapters separately per literacy band rather than fine-tuning a single unified retrieval model across all literacy levels.

**Alternatives considered:**
- Single fine-tune across all bands: simpler, loses literacy-specific adaptation
- Full fine-tune per band: too computationally expensive, three separate full fine-tunes
- Prompt conditioning: prepend literacy band token to queries, lightweight but limited adaptation capacity
- No adaptation: baseline DPR without literacy conditioning, loses ORACLE's core contribution

**Why PEFT per band:**
ORACLE's contribution is literacy-conditioned retrieval, the claim that retrieval quality improves when the retrieval model is adapted to the user's literacy level. Single fine-tune loses this conditioning. Full fine-tune per band is computationally prohibitive for a research pipeline. PEFT adapters (LoRA specifically) allow lightweight per-band adaptation with shared base model weights, enabling literacy conditioning with manageable compute. Prompt conditioning has insufficient adaptation capacity for retrieval quality differences across literacy bands.

---

## Decision 7: PlainQAFact + APPLS as Primary Evaluation Metrics

**Decision:** Evaluate Stage 4 generation quality using PlainQAFact (factual consistency) and APPLS (plain language quality) as primary metrics alongside standard ROUGE and BERTScore.

**Alternatives considered:**
- ROUGE only: surface-level n-gram overlap, misses factual accuracy and readability
- BERTScore only: semantic similarity, misses plain language quality
- Human evaluation: gold standard but infeasible at scale
- MedQA accuracy: measures medical knowledge, not plain language accessibility

**Why PlainQAFact + APPLS:**
ORACLE's contribution is health accessibility, not just retrieval accuracy but generation quality for low-literacy users. PlainQAFact measures whether generated answers are factually consistent with retrieved context, which is critical for medical content where hallucination is harmful. APPLS measures plain language quality specifically. It directly evaluates ORACLE's core claim that literacy-conditioned retrieval produces more accessible answers. Standard metrics (ROUGE, BERTScore) measure generation quality generally but not accessibility specifically. Both PlainQAFact and APPLS are validated for health communication contexts.

---

## Decision 8, PubMed API for Retrieval Corpus Augmentation

**Decision:** Augment the retrieval corpus with PubMed abstracts fetched via NCBI E-utilities API rather than using a static PubMed snapshot.

**Alternatives considered:**
- Static PubMed snapshot: reproducible but requires large storage and periodic updates
- PubMed Central full-text: richer content but access restrictions and parsing complexity
- No PubMed augmentation: loses clinical evidence retrieval entirely

**Why PubMed API:**
NCBI E-utilities provides programmatic access to 35M+ PubMed abstracts with no storage requirements. API-based fetching allows targeted retrieval of abstracts relevant to the QA corpus via MeSH query expansion, more efficient than bulk download. 412 abstracts fetched in Stage 1 pilot (FK mean 13.4, confirming clinical professional level). PubMed augmentation adds clinical evidence retrieval to the corpus without dominating the patient-facing content balance. Reproducible via documented API parameters and seed PMIDs.

---

## Decision 9, Journal of Biomedical Informatics as Target Venue

**Decision:** Submit ORACLE to Journal of Biomedical Informatics (JBI), with arXiv preprint uploaded simultaneously.

**Alternatives considered:**
- ACL/SIGIR: original target, wrong fit. ACL wants novel NLP methods. SIGIR wants novel IR systems. ORACLE's contribution is health accessibility application, not novel methodology. Both venues would likely desk-reject or receive weak reviews citing "no methodological contribution."
- JAMIA: more clinical focus, requires IRB-approved studies and clinical validation. ORACLE has neither.
- AMIA Annual Symposium: good visibility but conference format limits paper depth
- npj Digital Medicine: Nature portfolio, open access, high impact, possible future target if JBI rejects

**Why JBI:**
JBI publishes computational approaches to biomedical informatics without requiring clinical trial validation. ORACLE's literacy-conditioned RAG framing maps directly to JBI's scope. Target faculty Yue Guo (UIUC iSchool) publishes in JBI, venue alignment strengthens PhD application signal. Rolling submission, no fixed deadline pressure. JBI review time 4-8 weeks, first decision expected before Dec 1 Informatics application deadline if submitted Sep 4.

**Submission target:** Sep 4, 2026 (shifted 14 days, sick leave Jul 13-25). arXiv preprint uploaded simultaneously.

---

## Decision 10, Two Open Architectural Risks Accepted, Not Fixed

**Decision:** Accept two known architectural risks as documented limitations rather than fixing them before Stage 1 implementation.

**Risk 1, FK on full_text validity:**
FK scored on question + answer concatenation is not validated as a patient literacy measure. Documented in Decision 1 and Decision 3. Accepted because: no validated alternative exists at corpus scale, FK is the standard health communication proxy, and Stage 4 ablation will empirically test whether this matters for retrieval quality.

**Risk 2, 93.6% clinical exam content:**
MedMCQA + MedQA + MIRAGE constitute 93.6% of the retrieval corpus. All three are clinical professional content (medical entrance exams, USMLE, clinical QA benchmarks), not patient-facing material. A paper framed as health accessibility for patients whose retrieval corpus is 94% clinical professional content contradicts its own premise. Accepted because: no sufficiently large patient-facing health QA corpus exists. PLABA (plain language adaptations) and MedQuAD (patient questions) are included but small. Paper must explicitly frame this as a limitation and scope the contribution accordingly, ORACLE improves retrieval quality for users at different FK literacy levels, not necessarily for lay patient populations specifically.

Both risks are documented here and in research_design_rationale.md. Both must appear in the paper's limitations section. Neither is hidden.

---

## Open Decisions, Not Yet Resolved

**Open 1, PEFT adapter type:**
LoRA vs adapter layers vs prefix tuning for per-band adaptation. To be determined empirically in Stage 2 based on parameter efficiency and retrieval quality tradeoff.

**Open 2, Literacy band boundaries:**
FK Grade < 6 (low), 6-10 (medium), > 10 (high), boundary values chosen based on NIH patient material guidelines but not validated for this corpus. Stage 1 EDA confirmed FK distribution across bands. Whether these boundaries produce meaningfully different retrieval behavior is unknown until Stage 3.

**Open 3, MedQuAD re-addition:**
If original XML from abachaa/MedQuAD is parsed before Stage 3, MedQuAD will be added with a corpus update commit. This would reduce MedMCQA's share and improve patient-facing content ratio. Depends on implementation time available before Stage 3 begins.

---

## References

All references as listed in literature_review.md and literature_analysis.md.

## Decision 11, Literacy Classification: Rule-Based FK Thresholds Rather Than ML Classifier

**Decision:** Use rule-based Flesch-Kincaid grade thresholds for query literacy routing rather than a trained ML classifier.

**What was tried:** Gradient Boosting classifier trained on corpus documents using FK grade, FRE score, and word count as features. Achieved 100% test accuracy and 100% 5-fold CV accuracy.

**Why rejected:** The 100% accuracy is circular, FK grade was used to create the literacy band labels (low ≤6, medium 7-10, high 11-14, clinical 15+) and then used as the primary classifier feature. The model is not learning anything, it is recovering the deterministic rule used to create the labels. This is data leakage from label construction.

**Why rule-based routing:** Rule-based FK thresholds are honest, interpretable, and directly implement the same logic used in corpus labeling. No model file needed. No training required.

**Known limitation:** FK grade is unreliable for short queries (fewer than 3 sentences). Single-sentence queries produce FK scores that may not reflect the reader's actual literacy level. Sample accuracy on 8 test queries: 4/8 (50%). This is a known open risk, see research_design_rationale.md.

**Refinement (Aug 2 2026):** A larger, dedicated evaluation in retrieval_evaluation.py (20 test queries, 5 per literacy band, live-reproduced during today's ORACLE audit) gives a more precise picture than the 8-query spot-check above: routing accuracy is not uniform across bands - low 80.0% (4/5), medium 60.0% (3/5), high 20.0% (1/5), clinical 40.0% (2/5), for an overall 50.0% (10/20) - essentially identical to the earlier 8-query figure, but this aggregate masks a real and important skew: routing is much more reliable for low/medium-literacy queries (80%, 60%) than for high/clinical queries (20%, 40%). The blended percentage alone would wrongly suggest routing fails uniformly; it actually fails far more for exactly the more complex queries where correct routing matters most. Worth stating the per-band breakdown in the paper rather than a single blended percentage.

**Word count gate tried and rejected:** A 20-word threshold defaulting short queries to medium was implemented but reduced accuracy to 2/8 (25%) because all test queries were under 20 words. Reverted to pure FK thresholds.

**Architectural resolution, Stage 3 handles literacy correction:**
Stage 2 routing is a best-effort FK approximation. The retrieval pipeline passes both the routed band AND routing confidence to Stage 3. Stage 3 health literacy adaptation corrects for routing errors by adapting the response to the user's actual literacy level regardless of which band was retrieved from. This separation of concerns, retrieve relevant documents (Stage 2) then adapt to literacy level (Stage 3), is architecturally cleaner than requiring perfect query-time routing.

**Implication for retrieval_pipeline.py:** Must pass routing_band, fk_grade, routing_confidence, and full query to Stage 3. Stage 3 cannot assume routing is correct.

**Implication for Stage 3 design:** Must include a literacy correction layer that operates on retrieved documents regardless of routing band. User-declared literacy level or session-level adaptation preferred over query-level FK classification.

## Decision 12, GPT-4o-mini as Stage 4 Generation Model

**Decision:** Use `gpt-4o-mini` via OpenAI API as the primary generation model for Stage 4 literacy-conditioned response generation.

**Alternatives considered:**
- GPT-4o: highest quality but 10x cost, prohibitive at evaluation scale
- Flan-T5-base: free, runs locally, but 250M parameters insufficient for health domain generation quality, literacy conditioning effect would be confounded by weak generation
- Flan-T5-large: better but still not health-specialized, same confounding risk
- BioGPT-Large: health domain fine-tuned but decoder-only, poor instruction following for summarization
- Llama 3.1 8B: strong but requires 16GB RAM + GPU for inference at research scale

**Why gpt-4o-mini:**
Instruction-tuned, health domain capable, strong summarization quality at evaluation scale. Cost-efficient at ~$2-3 for full evaluation set. Same API key as HyDMIS GPT-4 semantic verification, single credential covers both papers. Allows clean attribution of literacy conditioning effect to retrieval architecture rather than generation quality. Production-deployable framing: gpt-4o-mini is a realistic production choice for health information systems, strengthening external validity of ORACLE's claims.

**Cost estimate:** ~$2-3 for Stage 4 evaluation set (500-1000 queries across 4 literacy bands).

**Comfort level:** High. gpt-4o-mini validated for summarization and health QA. Cost fixed and predictable. Same key already budgeted for HyDMIS (~$3-4).

## Decision 13, Adapted Factual Consistency Evaluation Instead of Official PlainQAFact; Cloud GPU Path Deferred

**Decision:** Evaluate Stage 4 factual consistency using an adapted GPT-4o-mini-based methodology (src/factual_consistency_eval.py), not the official PlainQAFact metric (You & Guo, 2025, arXiv 2503.08890, accepted JBI) directly.

**Why not the official pip package:** The official plainqafact package requires Llama 3.1 8B Instruct locally (40GB+ GPU memory per the repo's own README) and a separate LERC scoring model - infeasible on this machine (Apple Silicon, no CUDA). A dry-run install was tested and revealed a further problem beyond compute: the package's dependency tree (torch 2.13.0, transformers 4.44.2, pyserini, faiss-cpu, nmslib, spacy) conflicts with versions already verified working elsewhere in this project (torch 2.10.0, transformers 4.57.6, confirmed working via HyDMIS's mbert_classifier.py). Installing it risked breaking a working environment for a package that would still hit the same GPU wall regardless.

**Why not a full local reimplementation either:** Considered rebuilding PlainQAFact's real algorithm locally (real BART question-generation model + real LERC scoring + GPT-4o-mini only for the Llama step). Rejected for tonight: this requires implementing 4 real components (sentence classifier, question generation, LERC scoring, weighted aggregation) from a paper's methods description without a reference implementation to validate against, carrying real risk of subtle, hard-to-catch divergence from the actual metric. Not worth the risk at this hour for a metric that will be run properly via cloud GPU instead.

**What was built instead:** factual_consistency_eval.py borrows PlainQAFact's real conceptual structure (classify claims as simplification vs. elaboration, judge factual consistency for each, weight by claim-type count) but substitutes GPT-4o-mini for every model in the original pipeline. Explicitly labeled as an adapted evaluation, not the official metric, in the script's own docstring and print output.

**First results (20-record Stage 4 lay-summarizer sample):** Mean overall consistency ~0.96 (0.957 on first run, 0.962 on immediate re-run with identical inputs and temperature=0). Simplification claims consistently scored notably higher (0.972 / 0.980 across the two runs) than elaboration claims (0.862 / 0.856, std ~0.30 both times) - consistent with PlainQAFact's own core finding that elaborative explanations are the harder, more hallucination-prone case, since they require the model to add context beyond direct restatement.

**Known limitation - run-to-run variance despite temperature=0:** Re-running the script on identical inputs produced close but not identical scores (see above). This is expected OpenAI API behavior - temperature=0 substantially reduces but does not fully eliminate non-determinism, due to floating-point behavior and routing in their serving infrastructure. Point estimates from a single run should not be treated as exactly reproducible; the qualitative pattern (elaboration scores notably lower and more variable than simplification scores) held consistently across both runs and is the reliable finding, not the specific decimal values.

**Figure:** factual_consistency_by_claim_type.png (figures/stage4/) - box plot with individual data points comparing simplification-claim scores (tight, 0.80-1.00) against elaboration-claim scores (full spread, 0.00-1.00), visually confirming elaboration claims are both lower-scoring and far more variable.

**Validation status (Aug 6 2026): environment built and verified working; proof-of-concept confirmed on 2 records before scaling to the full run below.**

Provisioned a real cloud GPU (RunPod, RTX PRO 6000, 96GB VRAM). Built the official PlainQAFact environment from scratch: isolated conda env (Python 3.9), cloned the real repository, resolved 7 genuine missing dependencies one at a time (transformers_old_tokenizer, tiktoken, faiss, python-liquid, nltk+punkt_tab, protobuf, hf_transfer), and correctly configured default_config.py - critically, switched answer_selection_strategy from the default 'llm-keywords' (which requires gated access to Llama 3.1 8B Instruct on Hugging Face) to 'gpt-keywords', an officially-supported alternative in the repo that uses GPT-4o-mini for answer extraction instead. This is a legitimate configuration choice documented in the original repo, not an approximation of our own.

Successfully ran the complete official pipeline end-to-end on 2 records: real BART question generation, real GPT-4o-mini answer extraction, real QAFactEval scoring (BERTScore + exact match). Result: internal_mean = 0.9025. This proves the full official pipeline works correctly.

**COMPLETE (Aug 6 2026): full 20-record official run finished successfully.**

Fixed a real upstream bug discovered during this run: PlainQAFact's own GitHub repository has broken Git LFS configuration (empty `.gitattributes`, `git lfs ls-files` returns nothing), meaning `git clone` + `git lfs pull` only ever retrieves ~130-byte LFS pointer stubs for the 18 Textbooks corpus files, not the real ~5-50MB text content - a bug independent of anything in our setup. Root cause confirmed via direct inspection: pointer files contained literal `git-lfs.github.com` pointer syntax instead of textbook content, causing a JSON decode error on the first real evaluation attempt. Fixed by cloning the same corpus directly and correctly from its authoritative source (`huggingface.co/datasets/MedRAG/textbooks`, properly LFS-configured), verified byte-for-byte size match against the original pointer files' stated sizes, and replacing the broken files in place. The pre-built embeddings/index (125,847 entries, exact match to the corpus's documented total) were confirmed already correct and did not need regenerating - only the raw text chunks were affected.

**Official result, all 20 records, run twice for stability check:**
- Run 1: internal_mean=0.6457, external_mean=0.2601, overall_mean=0.3260 (40 internal claims, 194 external claims)
- Run 2 (with full per-claim JSON logging): internal_mean=0.6629, external_mean=0.2567, overall_mean=0.3296 (42 internal claims, 192 external claims)
- Both runs used identical input data and temperature=0. The small differences between runs are expected OpenAI API non-determinism, the same documented behavior already noted for the adapted evaluation (C) above - not a data or configuration inconsistency. The two runs agree closely enough (all three metrics within ~0.02-0.04 of each other) to treat the finding as stable, not run-dependent.

**StatPearls verified as genuinely functional, not affected by the Textbooks LFS bug.** Before finalizing, checked whether StatPearls (the second half of the `combined` knowledge base) had the same broken-pointer problem as Textbooks. File sizes confirmed real content (18-33KB per article, 463MB total, not ~130-byte stubs). A direct retrieval test against StatPearls specifically, using the same Gabapentin query as the Textbooks test below, returned genuinely on-topic results: real passages specifically discussing gabapentin/pregabalin use in elderly patients, with higher similarity scores (69.9) than the equivalent Textbooks query (65.4). This confirms `knowledge_base='combined'` was correctly using both sources throughout, and StatPearls needed no fix.

**Systematic evidence of the domain-mismatch pattern, across all 234 claims (not one anecdote):**
- External (elaboration) claims: 64.1% scored below 0.3, only 15.6% scored above 0.8 (n=192, Run 2)
- Internal (simplification) claims: only 21.4% scored below 0.3, a clear majority (61.9%) scored above 0.8 (n=42, Run 2)
This is a substantial, consistent split across the full claim set, not a small number of unlucky examples - internal claims (checked directly against the source abstract) succeed at roughly the rate external claims (checked via knowledge-base retrieval) fail.

**Mechanism confirmed via direct retrieval inspection:** ran a standalone retrieval test on an actual claim from the dataset (a Gabapentin renal-clearance fact) against the Textbooks corpus specifically. Retrieval genuinely works - real similarity scores, real textbook content returned - but the top 3 results were about analgesic-induced kidney damage, antiepileptic drug liver metabolism, and opioid receptor mechanisms respectively: topically adjacent pharmacology content, not the specific Gabapentin renal-clearance fact needed to verify the claim. The equivalent StatPearls query, by contrast, returned genuinely on-topic gabapentin-specific content (see above) - showing the domain-mismatch is source-dependent, not a universal retrieval failure.

**Conclusion, evidence-backed at both the mechanism level (one worked example) and the systematic level (234 claims):** PlainQAFact's Textbooks knowledge base (18 USMLE board-exam textbooks) covers foundational medical education content and retrieves only topically-adjacent, not fact-specific, matches for ORACLE's clinical-trial-specific PLABA claims. StatPearls, a point-of-care clinical resource, retrieves more precisely for the same query type, but does not fully compensate - the combined external-claim pass rate remains low (64.1% below 0.3) because many claims still route to less-specific Textbooks matches or lack sufficiently specific coverage in either source. The low external_mean is an honest, correct, now-quantified signal of this mismatch, not a broken pipeline or a failure of our setup.

**Figure:** official_plainqafact_by_claim_type.png (figures/stage4/) - box plot with individual data points comparing internal (simplification) vs. external (elaboration) claim scores from the official PlainQAFact run, directly visualizing the 64.1%-vs-21.4% split described above.

**What this means for the paper, stated precisely:** the official PlainQAFact score (~0.33 overall) and C's adapted score (~0.96 overall) measure genuinely different things and are NOT directly comparable as "official vs. approximation of the same measurement." C's GPT-4o-mini-based approach judges factual consistency using the source abstract itself as ground truth (a well-matched, always-available reference). Official PlainQAFact judges factual consistency using external knowledge-base retrieval as ground truth (a reference that may or may not contain topically-specific matching content, and evidently often doesn't for ORACLE's clinical-trial-specific source material). This is itself a legitimate, citable methodological finding, now with systematic (not anecdotal) support: general-purpose medical knowledge bases are not a reliable factuality-verification ground truth for clinical-trial-specific health content, which has direct relevance to ORACLE's own thesis about literacy-conditioned retrieval needing source-appropriate grounding. The paper should report both scores with this explanation and the 64.1%-vs-21.4% systematic evidence, not treat one score as validating or invalidating the other.

## Decision 14: APPLS Metric Suite Justification (Extends Decision 7)

Decision: cite APPLS's published findings directly to justify why ORACLE reports a suite of metrics (PlainQAFact, FK, SMOG, ROUGE, BERTScore) rather than a single composite score, instead of building and running the full APPLS perturbation testbed on ORACLE's own PLABA data.

Two ways to use APPLS were considered. The first is to build the actual perturbation testbed from the APPLS GitHub repository, applying controlled perturbations to ORACLE's own source texts and checking whether ORACLE's chosen metrics respond correctly to each of the four criteria APPLS defines. This would be a genuine additional research project. The repository is small (3 stars, 21 commits, last meaningfully updated November 2024), requires formatting data into their specific CSV schema, running a multi stage pipeline (perturbation generation, then separate lexical or GPT based evaluation scripts), and includes a further separate scoring system called POMME with its own conda environment and reference dataset. One perturbation type, entity swap, requires installing an entirely different external repository. Given how much unplanned complexity a structurally similar situation produced during the PlainQAFact validation documented in Decision 13, this path is deferred as a separately scoped task rather than attempted today without a deliberate decision to do so.

The second, adopted here, is to use APPLS's own published results directly. The paper evaluated 14 existing metrics against four plain language summarization criteria: informativeness, simplification, coherence, and faithfulness. Their central finding is that no single existing metric captures all four criteria simultaneously. GPT based perplexity was the only metric sensitive to the simplification criterion specifically, while different metrics were needed for the other three. The paper's own recommendation, stated directly in the abstract, is that a suite of automated metrics should be used together rather than relying on any single method.

This finding provides direct, citable support for a choice ORACLE already made in Decision 7 for a different reason. ORACLE already reports PlainQAFact for factual consistency, FK and SMOG for readability, and ROUGE and BERTScore for general generation quality, rather than collapsing evaluation into one number. APPLS supplies independent published evidence that this multi metric approach is methodologically correct for plain language summarization specifically, not just a reasonable design choice made without external validation. This connection should be added to the paper's evaluation methodology section as direct support for the metric suite already in use.

Limitation, stated plainly: this decision cites APPLS's general findings about metric behavior on their own testbed. It does not test whether ORACLE's specific metrics behave correctly on ORACLE's own PLABA derived text, since that would require running the actual perturbation testbed described above. The two are not equivalent, and the paper should describe this as citing established evidence for the metric suite design, not as an APPLS evaluation of ORACLE's own outputs.

---

## Decision 15: APPLS Option 2 - Empirical Metric Sensitivity on ORACLE's PLABA Data

**Decision:** Run the APPLS perturbation testbed directly on ORACLE's own PLABA test split (148 records) to empirically verify that ORACLE's chosen metrics respond correctly to controlled perturbations, upgrading the metric suite justification from citation-based (Decision 14) to empirical.

**Date:** Aug 9 2026

**What we ran:**

Three perturbation types from the APPLS GitHub repository (LinguisticAnomalies/APPLS, Guo et al., EMNLP 2024) were applied to ORACLE's PLABA test split:

1. `delete_sentence` (informativeness criterion): Progressive sentence deletion using TextRank-based importance scoring. Produces 1,386 perturbed variants across 148 source texts.

2. `coherent` (coherence criterion): Sentence reordering via permutation sampling. Produces 1,613 perturbed variants.

3. `simplification` (simplification criterion): Lexical substitution using PLABA's expert plain language adaptations as the simple_text reference. Produces 1,476 perturbed variants. PLABA's labels (expert plain language adaptations) serve as the simple_text input, which is the correct pairing since ORACLE's task is exactly this: generate plain language from complex biomedical text.

**Perturbations skipped and why:**

- `add_definition`: Requires dbpedia.json external knowledge file, not included in the APPLS repository.
- `add_non_related_sentence` / `add_related_sentence`: Requires external sentence corpora (ACL-ARC, Cochrane abstracts) not included in the repository.
- `entity_swap` (faithfulness criterion): Requires AllenAI scientific-claim-generation repository, separate installation. Faithfulness is covered empirically by Decision 13 PlainQAFact evaluation.

**Evaluation method:**

Spearman correlation between perturbation percentage and metric score, measuring whether each metric tracks the perturbation magnitude monotonically. A metric is considered sensitive if |r| > 0.3 with p < 0.05.

**Results:**

| Metric | Informativeness (delete_sentence) | Coherence (coherent) | Simplification (simplification) |
|--------|----------------------------------|----------------------|--------------------------------|
| FK Grade | r=-0.080, not sensitive | r=0.281, not sensitive | r=-0.184, not sensitive |
| ROUGE-L | r=-0.982, SENSITIVE (p<0.001) | r=-0.838, SENSITIVE (p<0.001) | r=-0.859, SENSITIVE (p<0.001) |
| BERTScore | r=-0.973, SENSITIVE (p<0.001) | r=-0.628, SENSITIVE (p<0.001) | r=-0.875, SENSITIVE (p<0.001) |

**Interpretation:**

ROUGE-L and BERTScore are strongly sensitive to all three criteria on ORACLE's own PLABA data. FK grade shows no sensitivity to any of the three perturbation types. This is the expected and informative finding: FK measures surface readability (word length, sentence length) rather than information content, sentence order, or lexical substitution patterns. FK and SMOG serve a distinct role in ORACLE's metric suite: they measure the readability level of generated output, not sensitivity to the transformations tested here. This distinction strengthens the argument for reporting the full metric suite rather than any single metric.

The faithfulness criterion is covered empirically by Decision 13's PlainQAFact evaluation, giving ORACLE empirical coverage of all four APPLS criteria: informativeness and coherence and simplification via this decision, and faithfulness via Decision 13.

These results provide direct empirical confirmation on ORACLE's own data that ROUGE-L and BERTScore are appropriate metrics for evaluating ORACLE's generation quality, and that FK and SMOG serve a complementary and distinct measurement role.

**Limitation:** The simplification perturbation uses PLABA's expert plain language adaptations as the simple_text reference rather than LLM-generated simplifications as in the original APPLS paper. This is a valid alternative pairing since ORACLE's task is precisely the generation of expert-level plain language from biomedical text.

**Artifacts:**
- `src/appls_perturbation.py`: Perturbation script (delete_sentence + coherent + simplification)
- `src/appls_evaluation.py`: Metric sensitivity evaluation script
- `data/appls/plaba_test.csv`: PLABA test split in APPLS input format (148 records)
- `data/appls/plaba_test_with_simple.csv`: PLABA test split with simple_text column (148 records)
- `data/appls/delete_sentence_plaba_test_perturbation.csv`: 1,386 perturbed variants
- `data/appls/coherent_plaba_test_perturbation.csv`: 1,613 perturbed variants
- `data/appls/simplification_plaba_test_perturbation.csv`: 1,476 perturbed variants
- `data/appls/appls_oracle_results.csv`: Spearman correlation results (3 criteria)

---

## Decision 16: Cross-Dataset Evaluation Design and Routing-Readability Finding

**Decision:** Evaluate ORACLE's full retrieve-generate pipeline across all six corpus sources (medmcqa, medqa, mirage, plaba, pubmed, pubmedqa) rather than the fixed 20-query test set used in Stage 2 retrieval evaluation, to establish whether generation quality and literacy-band routing accuracy generalize across dataset types.

**Date:** Aug 10 2026

**Method:** For each source and each of its populated literacy bands, 5 queries were sampled (random_state=42) and passed through retrieval (top_k=5) then generation (gpt-4o-mini, temperature=0.3, seed=42). 144 total generations across 6 sources. PLABA uses `full_text` (the source abstract) as the query rather than `question`, since PLABA's question column contains only a PMID index, not free text; this was caught during a dry run when all 921 PLABA queries were silently filtered by the minimum-length check.

For every query where the FK-based router assigned an incorrect band, a second, paired generation was run using the correct target band with identical retrieved documents (condition: `upper_bound`). This isolates the effect of band-prompt choice from retrieval quality, since both runs share the same retrieved context and differ only in the literacy-band instruction given to the generator.

**Headline result:** FK reduction is negative across every source (system output is harder to read than the source text in every case), ranging from -1.78 (mirage) to -5.17 (pubmed). Band routing accuracy varies sharply by source: 100% (medmcqa, plaba), 75% (medqa), 43-44% (pubmedqa, pubmed), 25% (mirage).

**Routing-readability finding, with significance testing:** Paired Wilcoxon signed-rank tests (n=38 paired queries, misrouted-band vs correct-band-forced) show that correct band routing significantly improves FK reduction (wrong=-3.71, upper_bound=-2.31, p=0.032) but does not significantly change ROUGE-L (p=0.53) or BERTScore (p=0.61). This is not a null result to explain away; it is mechanistically expected given the design: the band prompt controls surface style (vocabulary, sentence length) and is exactly what FK measures, while retrieved content is fixed identically across both conditions, so content-level metrics (ROUGE-L, BERTScore) have no channel through which the band prompt could move them. The correct claim is bounded and specific: routing accuracy governs readability outcomes, not content relevance or faithfulness. This is a stronger, more falsifiable claim than an unbounded "routing accuracy matters," and the paper should state it this way rather than implying routing fixes generation quality generally.

**BERTScore added to evaluation suite:** ROUGE-L alone is uninformative for multiple-choice-derived sources (medqa ROUGE-L=0.031, mirage=0.048) where the reference answer is a short letter or phrase and ORACLE generates a full explanatory paragraph; low lexical overlap there is expected and not a quality signal. BERTScore (medqa=0.803, mirage=0.798) shows these generations are in fact semantically on-topic despite near-zero n-gram overlap, consistent with Decision 15's empirical validation that BERTScore is the more informative semantic metric for this pipeline's outputs.

**Limitation, stated plainly:** Sample size is 5 queries per source per band (n=144 total, n=38 for the paired routing comparison). This is a pilot-scale run; Aug 11-12 expand PubMedQA/MedMCQA and MIRAGE/MedQuAD/PLABA evaluation with larger samples, and the aggregation on Aug 13 should report whether the FK-reduction and routing-accuracy patterns found here hold at scale. Results here should be read as directional and statistically testable, not as final production numbers.

**Artifacts:**
- `src/cross_dataset_eval.py`: retrieval + generation pipeline across all sources, actual and upper_bound conditions, BERTScore scoring
- `src/cross_dataset_figures.py`: four figures, including paired Wilcoxon significance in the routing-impact comparison
- `data/processed/cross_dataset_results.csv`: 144 records, both conditions, all metrics
- `figures/stage4/cross_dataset_fk_comparison.png`, `cross_dataset_routing_rouge.png`, `cross_dataset_fk_by_band.png`, `cross_dataset_routing_impact.png`

**Update Aug 11 2026, scale-up of medmcqa and pubmedqa to n=25/band (from pilot n=5/band, capped at each band's actual pool size):** Paired set expanded from n=38 to n=71 misrouted queries. FK Reduction significance strengthened from p=0.032 to p=0.0094 (wrong=-3.991, upper_bound=-2.698). ROUGE-L (p=0.848) and BERTScore (p=0.977) remain non-significant, replicating the original finding that correct-band routing affects readability specifically, not semantic content or faithfulness. This result no longer carries the "thin sample" caveat from the original pilot and is stated as a confirmed finding, not a directional pilot observation.

**Correction, Aug 12 2026 — the "confirmed finding" framing above did not hold.** Scaling mirage and plaba to n=25/band (from their existing n=5/band pilot samples) brought the paired routing-comparison set to n=124 misrouted queries (up from n=71). At this larger, more representative n: FK reduction p=0.049 (wrong=-3.401, upper_bound=-2.546), ROUGE-L p=0.179, BERTScore p=0.072. The FK effect is barely significant, not more significant than the Aug 11 update reported — the Aug 11 claim that this result "no longer carries the thin sample caveat" and could be "stated as a confirmed finding" was premature. Read plainly, the trajectory across three sample sizes (n=38: p=0.032, n=71: p=0.0094, n=124: p=0.049) is not a monotonic strengthening; it moved toward significance and then back toward the threshold as more, more heterogeneous data was added. mirage now supplies the majority of misrouted pairs (68/124) and has markedly lower routing accuracy (32%, errors distributed across all bands, not concentrated at one boundary) than the other five sources; plaba by contrast shows 100% routing accuracy, verified as genuine (FK independently recomputed on raw text matches the corpus's pre-scored FK within 0.03 in spot checks, not a routing-fallback artifact) and attributable to plaba's narrow, FK-consistent writing register rather than any bug.

The paper should report the current numbers (p=0.049/0.179/0.072, n=121-124) as the finding, describe the sample as expanded across three stages as data collection continued rather than fixed in advance, and state the FK-reduction result as marginal rather than robust. The underlying mechanistic claim from the original Decision 16 write-up still holds and is arguably the more important one regardless of exact p-value: band-prompt choice affects readability output specifically (the channel FK measures) and has no mechanism to affect retrieval-fixed content metrics (ROUGE-L, BERTScore) -- that structural argument doesn't depend on which side of 0.05 the FK test happens to land on, and is more defensible to lead with than the p-value itself.

**Decision 17: MedQuAD excluded from corpus build.** `medquad_loader.py` and `eda_medquad.py` exist in `src/` and were built, but MedQuAD was never wired into `oracle_corpus.csv` -- zero rows with `source == 'medquad'` in the current 37,076-row corpus (confirmed Aug 12 2026). It is not included in the n=25/band scale-up for this reason; there is nothing to sample. If MedQuAD integration happens later, it should get its own scale-up pass rather than being retrofitted into this comparison after the fact.

**Artifacts (Aug 12 2026 update):**
- `src/scale_up_mirage_plaba.py`: extends mirage/plaba from n=5/band pilot to n=25/band, matching the Aug 11 medmcqa/pubmedqa pattern
- `src/cross_dataset_figures.py`: `plot_significance_progression()` (dead code, unreachable, hardcoded unreproducible p-values) replaced with `plot_misroute_significance()`, computed live from current `cross_dataset_results.csv` on every run
- `src/cross_dataset_eval.py`: write to `OUTPUT_PATH` guarded against silent overwrite of accumulated results
- `notebooks/cross_dataset_eval.ipynb`: markdown and figure-generation cell updated to match current numbers and current figure set
- `data/processed/cross_dataset_results.csv`: 510 records total (140 new actual rows, 53 new upper_bound backfill rows)

**Note, Aug 12 2026 — `tokens` column clarification.** The `tokens` field in `cross_dataset_results.csv` logs `response.usage.total_tokens` (prompt + completion combined), not completion length alone. Checked during today's end-to-end review because per-source token means looked unexpectedly high relative to `max_tokens=300`; confirmed via a live test call that actual generated output stays comfortably within the 300-token completion cap (a 181-word response measured at 344 total_tokens, consistent with prompt tokens dominating the count). No generation is being truncated and no metric in this analysis is affected. `completion_tokens` is available in the raw API response (`generation_pipeline.py` line 111) but was never the field logged to the results CSV.

**Update, Aug 12 2026 — APPLS perturbation scope, precise reasons.** Two of APPLS's four native perturbation types were not run: `add_definition` requires `external_knowledge_source/dbpedia.json`, an external DBpedia entity-description extract never published in the APPLS repo and not reconstructed for this evaluation; `entity_swap` depends on scispacy (AllenAI) with UMLS-linked entity resolution plus multiple transformer models, built against a Linux+CUDA environment incompatible with this project's macOS setup. Both gaps are informativeness/factual-consistency adjacent; informativeness is already empirically validated via `delete_sentence`, and factual consistency is covered independently via PlainQAFact (Decision 13). No open validation gap results from skipping these two.

**Note, Aug 13 2026 — duplicate row removal in cross_dataset_results.csv.** 3 duplicate source/query/condition/band pairs found, all plaba/medium/actual (6 rows). Each pair is two separate stochastic generations of the same input query, not a copy error -- fk_grade, rouge_l, and bertscore differ slightly between the two rows in each pair, confirming independent re-generation. First occurrence kept per pair; 3 rows dropped. Dataset: 706 -> 703 rows.

**Note, Aug 13 2026 — null fk_generated rows, root cause confirmed.** 25 rows (medmcqa: 13, mirage: 12) have null fk_generated, all with valid generated_text, rouge_l, and bertscore. Root cause, confirmed by direct inspection of `score_readability()` in generation_pipeline.py: the function requires >=30 words before computing Flesch-Kincaid grade, returning None below that threshold rather than an unreliable estimate. This is an intentional guard, not a bug -- FK and SMOG are calibrated for passages, not single short sentences, and a computed grade on e.g. a 7-word factual answer would not be a meaningful reading-level estimate. Affected rows are concentrated in medmcqa and mirage, consistent with those sources occasionally producing terse single-sentence factual answers. These 25 rows are correctly excluded from FK-based aggregates (mean, Wilcoxon) via pandas' default NaN handling; they remain fully valid for rouge_l and bertscore analysis.

## Decision 18: FK Ablation, Full-Text vs Question-Only (Closes Open Risk from Decision 1/3)

**Background:** Decision 1/3 flagged that ORACLE's corpus computes FK readability scores on concatenated question+answer text (full_text), not on the query alone, and noted this was unvalidated against retrieval quality.

**Method:** For all 523 records in the actual-condition cross-dataset evaluation set, recomputed literacy-band assignment using `classify_query()` (the real production classifier) on question text alone, and compared against the routing decision that used full_text FK at corpus build time.

**Result:** 82.6% agreement (432/523), 17.4% disagreement (91/523). Of the disagreements, 80 records were correctly routed under full_text FK but would be incorrectly routed under question-only FK; only 11 went the other direction, a roughly 7-to-1 asymmetry, not random disagreement.

**Contributing factor confirmed:** disagreeing records sit closer to a band threshold on average (mean margin 1.55) than agreeing records (mean margin 1.98), consistent with boundary-adjacent queries being more sensitive to which text is scored.

**Open, not yet confirmed:** the directional skew itself (why correct-to-wrong dominates over wrong-to-correct) is not fully explained by margin alone. A plausible mechanism is that appending answer text systematically shifts FK scores in one direction (e.g., via changed average sentence/word length), but this has not been directly tested. Flagged for future work rather than asserted as confirmed.

**Practical conclusion:** full_text FK is not neutral relative to question-only FK, it measurably changes routing correctness for roughly 1 in 6 queries in this evaluation set, concentrated among boundary-adjacent cases. This should be stated as a limitation in the paper, not treated as a validated design choice.

**Artifacts:** `src/fk_ablation.py`, `docs/fk_ablation_results.csv`, `figures/stage4/fk_ablation_full_text_vs_question_only.png`

**Note, Aug 14 2026, confirming current significance number.** Independently recomputed on the CSV as it exists today, after medqa and pubmed were added at n=25/band (703 total rows): FK reduction p=0.0018, n=177. ROUGE-L p=0.6398, BERTScore p=0.9374, n=180, both non-significant. This confirms the finding strengthened again beyond the Aug 12 correction's p=0.049/n=124 snapshot, as the dataset grew further; the Aug 12 note remains an accurate record of that intermediate state, not an error. This is the current number as of this entry and should be treated as authoritative unless a later dated note supersedes it.
