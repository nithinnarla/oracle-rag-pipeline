# ORACLE — Methodology Decisions Log
## Literacy-Conditioned Health RAG Pipeline — Methodology Decisions

---

## How to Read This Document

This is a decisions log, not a polished writeup. Every major methodological choice is documented here with the alternatives I considered and why I made the call I made. Some decisions I am confident about. A few remain uncertain — those are marked with a note.

The point of documenting decisions before writing code is to prevent the most common research mistake: making a decision implicitly during implementation and then justifying it post-hoc in the paper. Every decision here was made before Stage 1 pipeline implementation begins. If Stage 1-4 results contradict a decision, the decision gets updated — but the reasoning trail stays visible.

**Status:** Complete — decisions locked before Stage 1 pipeline implementation begins.

---

## Decision 1 — Flesch-Kincaid as Primary Literacy Proxy

**Decision:** Use Flesch-Kincaid Grade Level (FK) as the primary readability proxy for literacy band assignment across the ORACLE retrieval corpus.

**Alternatives considered:**
- SMOG Index: better for medical content, requires 30+ sentences — most QA records too short
- Flesch Reading Ease (FRE): inverse of FK, same underlying formula, adds no information
- Coleman-Liau: character-based, handles medical jargon differently — less validated than FK for health content
- Human annotation: gold standard but infeasible at corpus scale (37,076 records)
- LLM-based readability scoring: GPT-4 can assess readability but adds API cost and latency to ingestion

**Why FK:**
FK is the most widely validated readability formula in health communication research. The National Institutes of Health recommends patient materials target FK Grade 6-8. FK requires only sentence length and syllable count — computable at ingestion time with zero additional infrastructure. textstat library provides reliable FK scoring with no external dependencies.

**Uncertainty note — documented open risk:** FK measures sentence length and syllable count, not vocabulary difficulty or conceptual density. A sentence using "myocardial infarction" scores the same as one using "heart attack" if both have the same length and syllable profile. FK will systematically misclassify medical jargon-heavy content as high-literacy when it may actually be low-comprehension for lay readers. This is the primary limitation of ORACLE's literacy conditioning approach and must be stated explicitly in the paper's limitations section. A validation step comparing FK bands to human readability judgments is deferred to Stage 4 evaluation.

---

## Decision 2 — MedMCQA Capped at 20,000 Records Stratified by Subject

**Decision:** Cap MedMCQA from 193,155 records to 20,000 records, stratified by subject_name, before inclusion in the retrieval corpus.

**Alternatives considered:**
- Full 193,155 records: MedMCQA becomes 89.9% of the retrieval corpus — single-source domination
- Random 20,000 sample: loses subject distribution — some medical specialties over/underrepresented
- 50,000 stratified: reduces domination to ~60% — still too high
- Exclude MedMCQA entirely: loses 21 medical subjects, largest clinical coverage dataset available

**Why 20,000 stratified:**
At 193,155 records MedMCQA was 89.9% of the retrieval corpus before capping. A paper framed as literacy-conditioned health RAG for patients whose retrieval index is 90% medical-entrance-exam trivia contradicts its own premise. Capping at 20,000 stratified by subject reduces MedMCQA from 89.9% to 42.4% of the corpus — majority but not dominating. Stratification preserves subject coverage while controlling corpus balance. Verified: final corpus 37,076 records across 6 sources, zero null answers, zero unknown FK, zero duplicates.

---

## Decision 3 — FK Scored on full_text, Not Question Alone

**Decision:** Score FK readability on the full_text field (question + answer concatenated) rather than on the question text alone.

**Alternatives considered:**
- Question-only FK scoring: captures reading difficulty of the prompt but ignores answer complexity
- Answer-only FK scoring: captures response complexity but ignores how the question is framed
- Separate FK scores for question and answer: doubles the feature set, complicates band assignment

**Why full_text:**
ORACLE's retrieval context is the full QA pair — both question and answer are returned to the user. Scoring FK on question alone would assign literacy bands based on prompt complexity while ignoring answer complexity. A simple question with a jargon-heavy answer would be incorrectly assigned to a low-literacy band. Scoring full_text ensures the literacy band reflects the complexity of the complete retrieved content the user receives. Consistent with MIRAGE benchmark's answer format, which concatenates question and answer options.

**Open risk:** FK on concatenated text is not validated as a patient literacy measure in the health communication literature. A 3-word answer appended to a 50-word question changes the FK score in ways that may not reflect actual reading difficulty. This limitation is documented and accepted for Stage 1. Stage 4 ablation will compare full_text FK versus question-only FK on retrieval quality metrics.

---

## Decision 4 — MedQuAD Excluded from Retrieval Corpus

**Decision:** Exclude MedQuAD from the ORACLE retrieval corpus despite being a patient-facing health QA dataset.

**Alternatives considered:**
- Include MedQuAD via HuggingFace: HuggingFace version is questions-only — no answers available
- Re-parse from original XML: abachaa/MedQuAD GitHub contains full XML with answers — requires custom parser
- Include partial MedQuAD with null answers: introduces known null contamination into corpus

**Why excluded:**
The HuggingFace MedQuAD dataset (hf-datasets version) contains only questions with null answer fields — 47,441 records with no retrievable content. Including null-answer records would contaminate the retrieval corpus with empty context. Re-parsing from the original XML at abachaa/MedQuAD is the correct approach but was not implemented before corpus finalization. MedQuAD re-addition is documented as a future work item — if original XML is parsed before Stage 3, it will be added with a corpus update commit.

---

## Decision 5 — DPR Over BM25 or Hybrid Retrieval

**Decision:** Use Dense Passage Retrieval (DPR) as the primary retrieval backbone for Stage 1, with BM25 as a baseline comparison.

**Alternatives considered:**
- BM25 only: sparse retrieval, no semantic understanding, fails on paraphrase and medical synonyms
- Hybrid BM25 + DPR: stronger than either alone but doubles infrastructure complexity
- ColBERT: late-interaction model, stronger than DPR on many benchmarks but computationally expensive at inference
- RAG with frozen retriever: simpler but cannot be adapted per literacy band

**Why DPR:**
DPR encodes semantic meaning rather than keyword overlap — critical for health QA where "heart attack" and "myocardial infarction" must retrieve the same documents. DPR's encoder architecture supports fine-tuning per literacy band via PEFT adapters — the core Stage 2 intervention. BM25 cannot be adapted per literacy band. Hybrid retrieval adds infrastructure complexity without a clear hypothesis for how it interacts with literacy conditioning. MIRAGE benchmark uses DPR-based retrieval as its primary setting — using DPR maintains direct comparability with ORACLE's primary evaluation framework.

---

## Decision 6 — PEFT Adapters Per Literacy Band Over Single Fine-Tune

**Decision:** Apply PEFT (Parameter Efficient Fine-Tuning) adapters separately per literacy band rather than fine-tuning a single unified retrieval model across all literacy levels.

**Alternatives considered:**
- Single fine-tune across all bands: simpler, loses literacy-specific adaptation
- Full fine-tune per band: too computationally expensive — three separate full fine-tunes
- Prompt conditioning: prepend literacy band token to queries — lightweight but limited adaptation capacity
- No adaptation: baseline DPR without literacy conditioning — loses ORACLE's core contribution

**Why PEFT per band:**
ORACLE's contribution is literacy-conditioned retrieval — the claim that retrieval quality improves when the retrieval model is adapted to the user's literacy level. Single fine-tune loses this conditioning. Full fine-tune per band is computationally prohibitive for a research pipeline. PEFT adapters (LoRA specifically) allow lightweight per-band adaptation with shared base model weights — enabling literacy conditioning with manageable compute. Prompt conditioning has insufficient adaptation capacity for retrieval quality differences across literacy bands.

---

## Decision 7 — PlainQAFact + APPLS as Primary Evaluation Metrics

**Decision:** Evaluate Stage 4 generation quality using PlainQAFact (factual consistency) and APPLS (plain language quality) as primary metrics alongside standard ROUGE and BERTScore.

**Alternatives considered:**
- ROUGE only: surface-level n-gram overlap, misses factual accuracy and readability
- BERTScore only: semantic similarity, misses plain language quality
- Human evaluation: gold standard but infeasible at scale
- MedQA accuracy: measures medical knowledge, not plain language accessibility

**Why PlainQAFact + APPLS:**
ORACLE's contribution is health accessibility — not just retrieval accuracy but generation quality for low-literacy users. PlainQAFact measures whether generated answers are factually consistent with retrieved context — critical for medical content where hallucination is harmful. APPLS measures plain language quality specifically — directly evaluates ORACLE's core claim that literacy-conditioned retrieval produces more accessible answers. Standard metrics (ROUGE, BERTScore) measure generation quality generally but not accessibility specifically. Both PlainQAFact and APPLS are validated for health communication contexts.

---

## Decision 8 — PubMed API for Retrieval Corpus Augmentation

**Decision:** Augment the retrieval corpus with PubMed abstracts fetched via NCBI E-utilities API rather than using a static PubMed snapshot.

**Alternatives considered:**
- Static PubMed snapshot: reproducible but requires large storage and periodic updates
- PubMed Central full-text: richer content but access restrictions and parsing complexity
- No PubMed augmentation: loses clinical evidence retrieval entirely

**Why PubMed API:**
NCBI E-utilities provides programmatic access to 35M+ PubMed abstracts with no storage requirements. API-based fetching allows targeted retrieval of abstracts relevant to the QA corpus via MeSH query expansion — more efficient than bulk download. 412 abstracts fetched in Stage 1 pilot (FK mean 13.4, confirming clinical professional level). PubMed augmentation adds clinical evidence retrieval to the corpus without dominating the patient-facing content balance. Reproducible via documented API parameters and seed PMIDs.

---

## Decision 9 — Journal of Biomedical Informatics as Target Venue

**Decision:** Submit ORACLE to Journal of Biomedical Informatics (JBI), with arXiv preprint uploaded simultaneously.

**Alternatives considered:**
- ACL/SIGIR: original target — wrong fit. ACL wants novel NLP methods. SIGIR wants novel IR systems. ORACLE's contribution is health accessibility application, not novel methodology. Both venues would likely desk-reject or receive weak reviews citing "no methodological contribution."
- JAMIA: more clinical focus — requires IRB-approved studies and clinical validation. ORACLE has neither.
- AMIA Annual Symposium: good visibility but conference format limits paper depth
- npj Digital Medicine: Nature portfolio, open access, high impact — possible future target if JBI rejects

**Why JBI:**
JBI publishes computational approaches to biomedical informatics without requiring clinical trial validation. ORACLE's literacy-conditioned RAG framing maps directly to JBI's scope. Target faculty Yue Guo (UIUC iSchool) publishes in JBI — venue alignment strengthens PhD application signal. Rolling submission — no fixed deadline pressure. JBI review time 4-8 weeks — first decision expected before Dec 1 Informatics application deadline if submitted Sep 4.

**Submission target:** Sep 4, 2026 (shifted 14 days — sick leave Jul 13-25). arXiv preprint uploaded simultaneously.

---

## Decision 10 — Two Open Architectural Risks Accepted, Not Fixed

**Decision:** Accept two known architectural risks as documented limitations rather than fixing them before Stage 1 implementation.

**Risk 1 — FK on full_text validity:**
FK scored on question + answer concatenation is not validated as a patient literacy measure. Documented in Decision 1 and Decision 3. Accepted because: no validated alternative exists at corpus scale, FK is the standard health communication proxy, and Stage 4 ablation will empirically test whether this matters for retrieval quality.

**Risk 2 — 93.6% clinical exam content:**
MedMCQA + MedQA + MIRAGE constitute 93.6% of the retrieval corpus. All three are clinical professional content (medical entrance exams, USMLE, clinical QA benchmarks) — not patient-facing material. A paper framed as health accessibility for patients whose retrieval corpus is 94% clinical professional content contradicts its own premise. Accepted because: no sufficiently large patient-facing health QA corpus exists. PLABA (plain language adaptations) and MedQuAD (patient questions) are included but small. Paper must explicitly frame this as a limitation and scope the contribution accordingly — ORACLE improves retrieval quality for users at different FK literacy levels, not necessarily for lay patient populations specifically.

Both risks are documented here and in research_design_rationale.md. Both must appear in the paper's limitations section. Neither is hidden.

---

## Open Decisions — Not Yet Resolved

**Open 1 — PEFT adapter type:**
LoRA vs adapter layers vs prefix tuning for per-band adaptation. To be determined empirically in Stage 2 based on parameter efficiency and retrieval quality tradeoff.

**Open 2 — Literacy band boundaries:**
FK Grade < 6 (low), 6-10 (medium), > 10 (high) — boundary values chosen based on NIH patient material guidelines but not validated for this corpus. Stage 1 EDA confirmed FK distribution across bands. Whether these boundaries produce meaningfully different retrieval behavior is unknown until Stage 3.

**Open 3 — MedQuAD re-addition:**
If original XML from abachaa/MedQuAD is parsed before Stage 3, MedQuAD will be added with a corpus update commit. This would reduce MedMCQA's share and improve patient-facing content ratio. Depends on implementation time available before Stage 3 begins.

---

## References

All references as listed in literature_review.md and literature_analysis.md.

## Decision 11 — Literacy Classification: Rule-Based FK Thresholds Rather Than ML Classifier

**Decision:** Use rule-based Flesch-Kincaid grade thresholds for query literacy routing rather than a trained ML classifier.

**What was tried:** Gradient Boosting classifier trained on corpus documents using FK grade, FRE score, and word count as features. Achieved 100% test accuracy and 100% 5-fold CV accuracy.

**Why rejected:** The 100% accuracy is circular — FK grade was used to create the literacy band labels (low ≤6, medium 7-10, high 11-14, clinical 15+) and then used as the primary classifier feature. The model is not learning anything — it is recovering the deterministic rule used to create the labels. This is data leakage from label construction.

**Why rule-based routing:** Rule-based FK thresholds are honest, interpretable, and directly implement the same logic used in corpus labeling. No model file needed. No training required.

**Known limitation:** FK grade is unreliable for short queries (fewer than 3 sentences). Single-sentence queries produce FK scores that may not reflect the reader's actual literacy level. Sample accuracy on 8 test queries: 4/8 (50%). This is a known open risk — see research_design_rationale.md.

**Refinement (Aug 2 2026):** A larger, dedicated evaluation in retrieval_evaluation.py (20 test queries, 5 per literacy band, live-reproduced during today's ORACLE audit) gives a more precise picture than the 8-query spot-check above: routing accuracy is not uniform across bands -- low 80.0% (4/5), medium 60.0% (3/5), high 20.0% (1/5), clinical 40.0% (2/5), for an overall 50.0% (10/20) -- essentially identical to the earlier 8-query figure, but this aggregate masks a real and important skew: routing is much more reliable for low/medium-literacy queries (80%, 60%) than for high/clinical queries (20%, 40%). The blended percentage alone would wrongly suggest routing fails uniformly; it actually fails far more for exactly the more complex queries where correct routing matters most. Worth stating the per-band breakdown in the paper rather than a single blended percentage.

**Word count gate tried and rejected:** A 20-word threshold defaulting short queries to medium was implemented but reduced accuracy to 2/8 (25%) because all test queries were under 20 words. Reverted to pure FK thresholds.

**Architectural resolution — Stage 3 handles literacy correction:**
Stage 2 routing is a best-effort FK approximation. The retrieval pipeline passes both the routed band AND routing confidence to Stage 3. Stage 3 health literacy adaptation corrects for routing errors by adapting the response to the user's actual literacy level regardless of which band was retrieved from. This separation of concerns — retrieve relevant documents (Stage 2) then adapt to literacy level (Stage 3) — is architecturally cleaner than requiring perfect query-time routing.

**Implication for retrieval_pipeline.py:** Must pass routing_band, fk_grade, routing_confidence, and full query to Stage 3. Stage 3 cannot assume routing is correct.

**Implication for Stage 3 design:** Must include a literacy correction layer that operates on retrieved documents regardless of routing band. User-declared literacy level or session-level adaptation preferred over query-level FK classification.

## Decision 12 — GPT-4o-mini as Stage 4 Generation Model

**Decision:** Use `gpt-4o-mini` via OpenAI API as the primary generation model for Stage 4 literacy-conditioned response generation.

**Alternatives considered:**
- GPT-4o: highest quality but 10x cost — prohibitive at evaluation scale
- Flan-T5-base: free, runs locally, but 250M parameters insufficient for health domain generation quality — literacy conditioning effect would be confounded by weak generation
- Flan-T5-large: better but still not health-specialized — same confounding risk
- BioGPT-Large: health domain fine-tuned but decoder-only, poor instruction following for summarization
- Llama 3.1 8B: strong but requires 16GB RAM + GPU for inference at research scale

**Why gpt-4o-mini:**
Instruction-tuned, health domain capable, strong summarization quality at evaluation scale. Cost-efficient at ~$2-3 for full evaluation set. Same API key as HyDMIS GPT-4 semantic verification — single credential covers both papers. Allows clean attribution of literacy conditioning effect to retrieval architecture rather than generation quality. Production-deployable framing: gpt-4o-mini is a realistic production choice for health information systems, strengthening external validity of ORACLE's claims.

**Cost estimate:** ~$2-3 for Stage 4 evaluation set (500-1000 queries across 4 literacy bands).

**Comfort level:** High. gpt-4o-mini validated for summarization and health QA. Cost fixed and predictable. Same key already budgeted for HyDMIS (~$3-4).
