"""
ORACLE, Stage 4: Literacy-Conditioned Response Generation
Phase 4, Stage 4: GPT-4o-mini generation with literacy conditioning

Takes a medical query, retrieves relevant context via Stage 2 retrieval pipeline,
and generates a literacy-conditioned response using gpt-4o-mini.

Architecture:
1. Retrieve top-k documents using retrieval_pipeline.retrieve()
2. Build literacy-conditioned prompt based on routing band
3. Generate response using gpt-4o-mini
4. Score readability using Flesch-Kincaid + SMOG
5. Save results to data/processed/stage4_results.csv

Decision 12: gpt-4o-mini chosen for cost efficiency (~$2-3 for full eval set),
instruction-following quality, and production-deployable framing.

Pipeline/infrastructure script, no notebook.
"""

import os
import sys
import time
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from retrieval_pipeline import retrieve
from dpr_encoder import get_dpr_query_encoder
import textstat
from openai import OpenAI

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(REPO_ROOT, "data", "processed", "stage4_results.csv")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "stage4")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

client = OpenAI()

# Literacy-conditioned system prompts per band
SYSTEM_PROMPTS = {
    'low': (
        "You are a health information assistant helping someone with limited health literacy. "
        "Use simple words, short sentences, and avoid medical jargon. "
        "Explain medical terms if you must use them. "
        "Write at a 6th grade reading level or below."
    ),
    'medium': (
        "You are a health information assistant helping someone with moderate health literacy. "
        "Use clear language, define technical terms when needed, and provide helpful context. "
        "Write at an 8th to 10th grade reading level."
    ),
    'high': (
        "You are a health information assistant helping someone with strong health literacy. "
        "You may use medical terminology with brief clarifications. "
        "Provide detailed, accurate information. "
        "Write at a college reading level."
    ),
    'clinical': (
        "You are a health information assistant for a healthcare professional. "
        "Use precise clinical terminology, include relevant clinical details, "
        "and assume familiarity with medical concepts. "
        "Write at a professional clinical level."
    )
}


def build_prompt(query: str, retrieved_docs: list, band: str) -> str:
    """Build literacy-conditioned prompt from retrieved context."""
    context_parts = []
    for doc in retrieved_docs[:3]:  # top-3 documents
        text = doc.get('full_text', '')
        question = doc.get('question', '')
        if question and question != 'nan':
            context_parts.append(f"Q: {question}\nContext: {text}")
        else:
            context_parts.append(f"Context: {text}")

    context = "\n\n---\n\n".join(context_parts)

    return (
        f"Using the following medical information, answer the question below.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"Provide a clear, accurate answer based on the context above."
    )


def generate_response(query: str, band: str, retrieved_docs: list,
                      model: str = "gpt-4o-mini", max_tokens: int = 300) -> dict:
    """Generate literacy-conditioned response using gpt-4o-mini."""
    system_prompt = SYSTEM_PROMPTS.get(band, SYSTEM_PROMPTS['medium'])
    user_prompt = build_prompt(query, retrieved_docs, band)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            seed=42
        )
        generated_text = response.choices[0].message.content.strip()
        usage = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }
        return {'text': generated_text, 'usage': usage, 'error': None}

    except Exception as e:
        return {'text': '', 'usage': {}, 'error': str(e)}


def score_readability(text: str) -> dict:
    """Score readability using Flesch-Kincaid and SMOG."""
    if not text or len(text.split()) < 30:
        return {'fk_grade': None, 'smog_grade': None, 'fk_reading_ease': None}

    try:
        fk_grade = textstat.flesch_kincaid_grade(text)
        smog = textstat.smog_index(text)
        fk_ease = textstat.flesch_reading_ease(text)
        return {
            'fk_grade': round(fk_grade, 2),
            'smog_grade': round(smog, 2),
            'fk_reading_ease': round(fk_ease, 2)
        }
    except Exception:
        return {'fk_grade': None, 'smog_grade': None, 'fk_reading_ease': None}



def plot_fk_by_band(df, figures_dir):
    """Plot FK grade by target literacy band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band_order = ['low', 'medium', 'high', 'clinical']
    means = [df[df['target_band'] == b]['fk_grade'].mean() for b in band_order]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#2ca25f', '#66c2a4', '#fc8d59', '#b30000']
    bars = ax.bar(band_order, means, color=colors)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.1f}",
                ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Flesch-Kincaid Grade Level")
    ax.set_xlabel("Target Literacy Band")
    ax.set_title("Stage 4 Generation - FK Grade by Target Literacy Band (n=%d)" % len(df))
    plt.tight_layout()
    outpath = os.path.join(figures_dir, "fk_by_literacy_band.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {outpath}")


def run_stage4(queries: list = None, top_k: int = 5, dry_run: bool = False):
    """
    Run Stage 4 generation pipeline.

    Args:
        queries: list of (query, band) tuples. If None, uses default eval set.
        top_k: number of documents to retrieve per query
        dry_run: if True, skip API calls and simulate output
    """
    # Default evaluation queries, 2 per literacy band = 8 total
    if queries is None:
        queries = [
            ("What is diabetes and how does it affect the body?", "low"),
            ("How do I know if I have high blood pressure?", "low"),
            ("What are the treatment options for type 2 diabetes?", "medium"),
            ("How does metformin work to control blood sugar?", "medium"),
            ("What is the mechanism of insulin resistance in metabolic syndrome?", "high"),
            ("How does HbA1c reflect long-term glycemic control?", "high"),
            ("What are the pharmacokinetics of GLP-1 receptor agonists?", "clinical"),
            ("Describe the pathophysiology of diabetic nephropathy.", "clinical"),
        ]

    print("ORACLE, Stage 4: Literacy-Conditioned Response Generation")
    print("=" * 60)
    print(f"  Model: gpt-4o-mini")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  Queries: {len(queries)}")
    print(f"  Top-k retrieval: {top_k}")
    print()

    results = []
    total_tokens = 0

    # Pre-load DPR encoder once, avoids reloading on every query
    print("  Loading DPR encoder...")
    q_tokenizer, q_model = get_dpr_query_encoder()
    print("  DPR encoder ready\n")

    for i, (query, band) in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] Band: {band}, {query[:60]}...")

        # Step 1, Retrieve
        try:
            retrieval = retrieve(query, top_k=top_k, q_tokenizer=q_tokenizer, q_model=q_model)
            retrieved_docs = retrieval.get('retrieved', [])
            routing_band = retrieval.get('routing', {}).get('band', band)
        except Exception as e:
            print(f"    Retrieval error: {e}")
            retrieved_docs = []
            routing_band = band

        # Step 2, Generate
        if dry_run:
            generated = {
                'text': f"[DRY RUN] Simulated response for {band} literacy band.",
                'usage': {'total_tokens': 100},
                'error': None
            }
        else:
            generated = generate_response(query, band, retrieved_docs)
            time.sleep(0.5)  # rate limiting

        if generated['error']:
            print(f"    Generation error: {generated['error']}")

        # Step 3, Score readability
        readability = score_readability(generated['text'])

        # Step 4, Store result
        tokens = generated['usage'].get('total_tokens', 0)
        total_tokens += tokens

        result = {
            'query': query,
            'target_band': band,
            'routing_band': routing_band,
            'band_match': band == routing_band,
            'retrieved_count': len(retrieved_docs),
            'generated_text': generated['text'],
            'fk_grade': readability['fk_grade'],
            'smog_grade': readability['smog_grade'],
            'fk_reading_ease': readability['fk_reading_ease'],
            'total_tokens': tokens,
            'error': generated['error']
        }
        results.append(result)
        print(f"    FK grade: {readability['fk_grade']} | SMOG: {readability['smog_grade']} | Tokens: {tokens}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n  Results saved: {RESULTS_PATH}")
    plot_fk_by_band(df, FIGURES_DIR)
    print(f"  Total tokens used: {total_tokens}")
    print(f"  Estimated cost: ~${total_tokens * 0.00000015:.4f}")

    # Summary by band
    print("\n  Readability by literacy band:")
    for band in ['low', 'medium', 'high', 'clinical']:
        band_df = df[df['target_band'] == band]
        avg_fk = band_df['fk_grade'].mean()
        avg_smog = band_df['smog_grade'].mean()
        print(f"    {band:10s}: FK={avg_fk:.1f} SMOG={avg_smog:.1f}")

    return df


if __name__ == "__main__":
    # Run dry run first to verify pipeline
    run_stage4(dry_run=False)
