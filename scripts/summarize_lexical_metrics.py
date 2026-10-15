"""
Table 6 (split by temperature) : Per-model evaluation results

Averages *_eval_summary.csv over the 2 datasets for each
(temperature, prompt_type, model) combination and prints console tables.

Metrics:
  perplexite_mean      PPL↓   surface fluency
  flesch_re_fr_mean    Fl↑    French Flesch reading ease
  gunning_fog_fr_mean  Fog↓   Gunning Fog index (complexity)
  msttr_100_mean       MSTTR↑ lexical richness
  self_bleu_4          SB4↓   corpus self-similarity
  distinct_1           D1↑    unique unigram ratio  → displayed as %
  distinct_2           D2↑    unique bigram ratio   → displayed as %
  mean_pairwise_cosine MPC↓   mean TF-IDF cosine similarity
  n_mots_mean          Words  average word count

Usage (from project root CAG/):
    python scripts/summarize_lexical_metrics.py \\
        --summaries outputs/metrics_results/*/*.csv \\
        --output    outputs/metrics_results/lexical_metrics_by_temperature.csv
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Model order (display name, size label)
# ---------------------------------------------------------------------------
MODEL_ORDER = [
    # (HuggingFace ID,                              display name,        size)
    ("Qwen/Qwen2.5-7B-Instruct",                   "Qwen2.5-7B",        "7B"),
    ("Qwen/Qwen2.5-32B-Instruct",                  "Qwen2.5-32B",       "32B"),
    ("mistralai/Mistral-7B-Instruct-v0.3",         "Mistral-7B-v0.3",   "7B"),
    ("mistralai/Mistral-Nemo-Instruct-2407",       "Mistral-Nemo-12B",  "12B"),
    ("mistralai/Mistral-Small-24B-Instruct-2501",  "Mistral-Small-24B", "24B"),
    ("swiss-ai/Apertus-8B-Instruct-2509",          "Apertus-8B",        "8B"),
    ("swiss-ai/Apertus-70B-Instruct-2509",         "Apertus-70B",       "70B"),
    ("google/medgemma-27b-it",                     "MedGemma-27B",      "27B"),
]

# ---------------------------------------------------------------------------
# Metric definitions: (csv_column, latex_label, direction)
#   direction "up"   → bold highest (better = larger)
#   direction "down" → bold lowest  (better = smaller)
#   direction ""     → no bold
# ---------------------------------------------------------------------------
METRICS = [
    ("perplexite_mean",      "PPL",   "down"),
    ("flesch_re_fr_mean",    "Fl",    "up"),
    ("gunning_fog_fr_mean",  "Fog",   "down"),   # Fog↓: lower = simpler text
    ("msttr_100_mean",       "MSTTR", "up"),
    ("self_bleu_4",          "SB4",   "down"),
    ("distinct_1",           "D1",    "up"),
    ("distinct_2",           "D2",    "up"),
    ("mean_pairwise_cosine", "MPC",   "down"),
    ("n_mots_mean",          "Words", ""),
]
METRIC_COLS = [m[0] for m in METRICS]

# Sampling presets: temperature → (label_key, preset_name)
PRESETS = [
    (0.3, "t03", "conservateur"),
    (0.5, "t05", "standard"),
    (0.8, "t08", "exploratoire"),
]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(col: str, val: float) -> str:
    if pd.isna(val):
        return "---"
    if col == "perplexite_mean":
        return f"{val:.1f}"
    if col in ("flesch_re_fr_mean", "gunning_fog_fr_mean", "self_bleu_4"):
        return f"{val:.1f}"
    if col == "msttr_100_mean":
        return f"{val:.3f}"
    if col == "distinct_1":
        return f"{val * 100:.2f}"
    if col == "distinct_2":
        return f"{val * 100:.1f}"
    if col == "mean_pairwise_cosine":
        return f"{val:.3f}"
    if col == "n_mots_mean":
        return f"{val:.0f}"
    return f"{val:.3f}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summaries(patterns: list) -> pd.DataFrame:
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    paths = sorted(set(paths))
    if not paths:
        sys.exit(f"[error] No summary CSV files matched: {patterns}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        dfs.append(df)
        print(f"  Loaded {len(df):>3} groups from {Path(p).name}")
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Averaging
# ---------------------------------------------------------------------------

def average_per_model(
    df: pd.DataFrame,
    prompt_type: str,
    temperature: float | None = None,
) -> pd.DataFrame:
    """
    Average each metric over datasets for a given (prompt_type, temperature).
    If temperature is None, averages over all temperatures.
    Returns a DataFrame indexed by display model name.
    """
    id_to_name = {hf: name for hf, name, _ in MODEL_ORDER}
    sub = df[df["prompt_type"] == prompt_type].copy()
    if temperature is not None:
        sub = sub[np.isclose(sub["temperature"], temperature)]
    sub["model_name"] = sub["model"].map(id_to_name)
    sub = sub.dropna(subset=["model_name"])
    available = [c for c in METRIC_COLS if c in sub.columns]
    if sub.empty:
        return pd.DataFrame(columns=available)
    return sub.groupby("model_name")[available].mean()


# ---------------------------------------------------------------------------
# Console output  (zero-shot and five-shot side by side, like the paper table)
# ---------------------------------------------------------------------------

def print_table(avg_0: pd.DataFrame, avg_5: pd.DataFrame, title: str) -> None:
    labels  = [m[1] for m in METRICS]
    n       = len(labels)
    col_w   = 7
    sep     = "  "

    metrics_hdr = sep.join(f"{l:>{col_w}}" for l in labels)
    bar         = "-" * (22 + 4 + 2 + n * (col_w + 2) + 4 + n * (col_w + 2))

    print(f"\n{'=' * len(bar)}")
    print(f"  {title}")
    print(f"{'=' * len(bar)}")

    # Group header
    block = n * (col_w + len(sep))
    print(
        f"{'Model':<22} {'Sz':>4}  "
        f"{'Zero-shot':^{block}}"
        f"    "
        f"{'Five-shot':^{block}}"
    )
    print(f"{'':22} {'':4}  {metrics_hdr}    {metrics_hdr}")
    print(bar)

    for _, name, size in MODEL_ORDER:
        def row_vals(avgs):
            if name not in avgs.index:
                return sep.join(f"{'---':>{col_w}}" for _ in METRICS)
            return sep.join(
                f"{fmt(col, avgs.loc[name, col]) if col in avgs.columns else '---':>{col_w}}"
                for col, _, _ in METRICS
            )
        print(f"{name:<22} {size:>4}  {row_vals(avg_0)}    {row_vals(avg_5)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--summaries",
        nargs="+",
        default=["outputs/metrics_results/*/*.csv"],
        help="glob pattern(s) for eval_summary CSVs (default: outputs/metrics_results/*/*.csv)",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="output CSV path for the per-temperature averages",
    )
    args = ap.parse_args()

    print("Loading summary files:")
    df = load_summaries(args.summaries)
    print(
        f"Total: {len(df)} groups  "
        f"({df['model'].nunique()} model(s), "
        f"{df['dataset'].nunique()} dataset(s), "
        f"{df['temperature'].nunique()} temperature(s))"
    )

    all_rows = []

    for temp, label_key, preset_name in PRESETS:
        avg_0 = average_per_model(df, "0shot", temperature=temp)
        avg_5 = average_per_model(df, "5shot", temperature=temp)

        print_table(avg_0, avg_5, f"T={temp} ({preset_name})")

        for _, name, size in MODEL_ORDER:
            for pt, avgs in [("0shot", avg_0), ("5shot", avg_5)]:
                row = {
                    "temperature": temp,
                    "preset":      preset_name,
                    "model_name":  name,
                    "size":        size,
                    "prompt_type": pt,
                }
                for col, label, _ in METRICS:
                    if col in avgs.columns and name in avgs.index:
                        val = avgs.loc[name, col]
                        row[label] = val * 100 if col in ("distinct_1", "distinct_2") else val
                    else:
                        row[label] = float("nan")
                all_rows.append(row)

    out = pd.DataFrame(all_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, float_format="%.4f")
    print(f"\n→ Per-temperature averages saved to: {args.output}")


if __name__ == "__main__":
    main()
