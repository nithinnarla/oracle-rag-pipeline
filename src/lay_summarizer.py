"""
ORACLE — Lay Language Summarization
Phase 4 — Stage 4: LLM-based plain language summarization

Takes medical abstracts from PLABA dataset and generates plain language
summaries using gpt-4o-mini, then scores readability using FK + SMOG
against both the source abstract and the expert PLABA adaptation.

Data structure note: load_plaba_all() returns {'train','val','test','metadata'},
each split is {'data': DataFrame[question,pmid,input_text,Question_Type],
'labels': Series named target_text, same index as data}.

Pipeline/infrastructure script — no notebook.
"""

import os
import sys
import time
import warnings
import pandas as pd
import textstat
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from plaba_loader import load_plaba_all
from openai import OpenAI

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(REPO_ROOT, "data", "processed", "lay_summarizer_results.csv")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "stage4")
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

client = OpenAI()

SYSTEM_PROMPT = (
    "You are a health communication specialist. "
    "Your job is to rewrite complex medical text in plain language "
    "that a patient with an 8th grade reading level can understand. "
    "Use short sentences, common words, and explain medical terms. "
    "Do not add information not in the original text."
)


def score_readability(text: str) -> dict:
    if not text or len(text.split()) < 30:
        return {'fk_grade': None, 'smog_grade': None, 'fk_reading_ease': None}
    try:
        return {
            'fk_grade': round(textstat.flesch_kincaid_grade(text), 2),
            'smog_grade': round(textstat.smog_index(text), 2),
            'fk_reading_ease': round(textstat.flesch_reading_ease(text), 2)
        }
    except Exception:
        return {'fk_grade': None, 'smog_grade': None, 'fk_reading_ease': None}


def generate_plain_summary(abstract: str, max_tokens: int = 250) -> dict:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Rewrite this in plain language:\n\n{abstract}"}
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            seed=42
        )
        text = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        return {'text': text, 'tokens': tokens, 'error': None}
    except Exception as e:
        return {'text': '', 'tokens': 0, 'error': str(e)}



def plot_fk_comparison(df_results, figures_dir):
    """Plot FK grade comparison: source vs generated vs expert, by question type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    means = [
        df_results['source_fk'].mean(),
        df_results['generated_fk'].mean(),
        df_results['expert_fk'].mean()
    ]
    labels = ['Source\n(abstract)', 'Generated\n(gpt-4o-mini)', 'Expert\n(PLABA)']
    colors = ['#7f7f7f', '#2c7fb8', '#31a354']
    bars = ax.bar(labels, means, color=colors)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.1f}",
                ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Flesch-Kincaid Grade Level")
    ax.set_title("FK Grade -- Source vs Generated vs Expert (n=%d)" % len(df_results))
    ax.axhline(8, color='red', linestyle='--', alpha=0.4, label='8th grade target')
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1]
    types = sorted(df_results['question_type'].unique())
    x = np.arange(len(types))
    width = 0.25
    for i, (col, color, label) in enumerate([
        ('source_fk', '#7f7f7f', 'Source'),
        ('generated_fk', '#2c7fb8', 'Generated'),
        ('expert_fk', '#31a354', 'Expert')
    ]):
        vals = [df_results[df_results['question_type'] == t][col].mean() for t in types]
        ax.bar(x + (i - 1) * width, vals, width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Type {t}" for t in types])
    ax.set_ylabel("Flesch-Kincaid Grade Level")
    ax.set_title("FK Grade by PLABA Adaptation Type")
    ax.legend(fontsize=8)

    plt.tight_layout()
    outpath = os.path.join(figures_dir, "fk_comparison_source_generated_expert.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {outpath}")


def run_lay_summarizer(n_samples: int = 20, dry_run: bool = False, split: str = 'train'):
    print("ORACLE — Lay Language Summarization")
    print("=" * 60)
    print(f"  Model: gpt-4o-mini")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  Dataset: PLABA ({split} split — expert plain language adaptations)")
    print(f"  Samples requested: {n_samples}")
    print()

    print("  Loading PLABA dataset...")
    plaba = load_plaba_all()
    df = plaba[split]['data'].copy()
    labels = plaba[split]['labels']
    df['target_text'] = labels.values
    print(f"  Loaded {split}: {len(df)} rows")

    n_per_type = max(1, n_samples // df['Question_Type'].nunique())
    df_sample = (
        df.groupby('Question_Type', group_keys=False)
          .apply(lambda x: x.sample(min(len(x), n_per_type), random_state=42))
    )
    df_sample = df_sample.head(n_samples).reset_index(drop=True)
    print(f"  Sampled: {len(df_sample)} instances across types {sorted(df_sample['Question_Type'].unique())}")
    print()

    results = []
    total_tokens = 0

    for i, row in df_sample.iterrows():
        source = str(row['input_text'])
        expert = str(row['target_text'])
        qtype = str(row['Question_Type'])

        print(f"  [{i+1}/{len(df_sample)}] Type {qtype} — {source[:60]}...")

        source_scores = score_readability(source)

        if dry_run:
            generated = {'text': '[DRY RUN] Simulated plain language summary.', 'tokens': 50, 'error': None}
        else:
            generated = generate_plain_summary(source)
            time.sleep(0.5)

        if generated['error']:
            print(f"    Error: {generated['error']}")

        gen_scores = score_readability(generated['text'])
        expert_scores = score_readability(expert)

        total_tokens += generated['tokens']

        fk_reduction = None
        if source_scores['fk_grade'] is not None and gen_scores['fk_grade'] is not None:
            fk_reduction = round(source_scores['fk_grade'] - gen_scores['fk_grade'], 2)

        result = {
            'question_type': qtype,
            'source_text': source[:200],
            'generated_summary': generated['text'],
            'expert_adaptation': expert[:200],
            'source_fk': source_scores['fk_grade'],
            'source_smog': source_scores['smog_grade'],
            'generated_fk': gen_scores['fk_grade'],
            'generated_smog': gen_scores['smog_grade'],
            'expert_fk': expert_scores['fk_grade'],
            'expert_smog': expert_scores['smog_grade'],
            'fk_reduction': fk_reduction,
            'tokens': generated['tokens'],
            'error': generated['error']
        }
        results.append(result)
        print(f"    Source FK: {source_scores['fk_grade']} -> Generated FK: {gen_scores['fk_grade']} (Expert: {expert_scores['fk_grade']})")

    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_PATH, index=False)
    print(f"\n  Results saved: {RESULTS_PATH}")
    plot_fk_comparison(df_results, FIGURES_DIR)
    print(f"  Total tokens: {total_tokens}")
    print(f"  Estimated cost: ~${total_tokens * 0.00000015:.4f}")

    print("\n  Readability summary (means, ignoring nulls):")
    print(f"    Source FK mean:    {df_results['source_fk'].mean():.1f}")
    print(f"    Generated FK mean: {df_results['generated_fk'].mean():.1f}")
    print(f"    Expert FK mean:    {df_results['expert_fk'].mean():.1f}")
    print(f"    FK reduction:      {df_results['fk_reduction'].mean():.1f} grades")

    return df_results


if __name__ == "__main__":
    run_lay_summarizer(n_samples=20, dry_run=False)
