"""
Table 8 : BERTScore F1 per model and encoder

Prints one table averaged over all temperatures and one table per
temperature preset (T=0.3, T=0.5, T=0.8), both averaged over datasets.

Encoders:
  bertscore_max_base_mean   CamemBERT-base  (camembert-base)
  bertscore_max_bio_mean    CamemBERT-bio   (almanach/camembert-bio-base)
  bertscore_max_drbert_mean DrBERT          (Dr-BERT/DrBERT-7GB)

Columns per encoder:
  0-shot   mean BERTScore F1 in zero-shot prompting
  5-shot   mean BERTScore F1 in five-shot prompting
  Gap      5-shot − 0-shot  (positive = exemplars help)

Usage (from project root CAG/):
    python scripts/summarize_bertscore.py \\
        --summaries outputs/metrics_results/*/*.csv \\
        --output    outputs/metrics_results/bertscore_by_model.csv
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ORDER = [
    ("Qwen/Qwen2.5-7B-Instruct",                   "Qwen2.5-7B",        "7B"),
    ("Qwen/Qwen2.5-32B-Instruct",                  "Qwen2.5-32B",       "32B"),
    ("mistralai/Mistral-7B-Instruct-v0.3",         "Mistral-7B-v0.3",   "7B"),
    ("mistralai/Mistral-Nemo-Instruct-2407",       "Mistral-Nemo-12B",  "12B"),
    ("mistralai/Mistral-Small-24B-Instruct-2501",  "Mistral-Small-24B", "24B"),
    ("swiss-ai/Apertus-8B-Instruct-2509",          "Apertus-8B",        "8B"),
    ("swiss-ai/Apertus-70B-Instruct-2509",         "Apertus-70B",       "70B"),
    ("google/medgemma-27b-it",                     "MedGemma-27B",      "27B"),
    # TODO: replace placeholders with actual HuggingFace model IDs
    # ("PLACEHOLDER/medllama-8b",                  "MedLlama-8B",       "8B"),
    # ("PLACEHOLDER/openbiollm-8b",                "OpenBioLLM-8B",     "8B"),
    # ("PLACEHOLDER/openbiollm-70b",               "OpenBioLLM-70B",    "70B"),
]

ENCODERS = [
    ("bertscore_max_base_mean",   "CamemBERT-base"),
    ("bertscore_max_bio_mean",    "CamemBERT-bio"),
    ("bertscore_max_drbert_mean", "DrBERT"),
]


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


def print_table(df: pd.DataFrame, title: str = "Table 8 — BERTScore F1") -> None:
    id_to_name = {hf: name for hf, name, _ in MODEL_ORDER}
    df = df.copy()
    df["model_name"] = df["model"].map(id_to_name)
    df = df.dropna(subset=["model_name"])

    available = [(col, label) for col, label in ENCODERS if col in df.columns]
    if not available:
        print("[info] No BERTScore columns found — run add_bertscore_encoder.py first.")
        return

    enc_w = 26
    header_enc = "  ".join(f"{label:^{enc_w}}" for _, label in available)
    col_hdr    = "  ".join(f"{'0-shot':>7} {'5-shot':>7} {'Gap':>7}" for _ in available)
    bar = "-" * (27 + len(col_hdr) + 4 * len(available))

    print(f"\n{'=' * len(bar)}")
    print(f"  {title}")
    print(f"  Gap = 5-shot − 0-shot  |  ★ = smallest gap per encoder")
    print(f"{'=' * len(bar)}")
    print(f"{'Model':<22} {'Sz':>4}  {header_enc}")
    print(f"{'':22} {'':4}  {col_hdr}")
    print(bar)

    # Collect values to find best per encoder
    values = {}
    for _, name, _ in MODEL_ORDER:
        sub = df[df["model_name"] == name]
        values[name] = {}
        for col, _ in available:
            s0 = sub[sub["prompt_type"] == "0shot"][col].mean()
            s5 = sub[sub["prompt_type"] == "5shot"][col].mean()
            values[name][col] = (s0, s5)

    min_gap = {}
    for col, _ in available:
        gaps = [values[n][col][1] - values[n][col][0]
                for _, n, _ in MODEL_ORDER
                if not pd.isna(values[n][col][0])]
        min_gap[col] = min(gaps) if gaps else float("nan")

    for _, name, size in MODEL_ORDER:
        row = f"{name:<22} {size:>4}"
        for col, _ in available:
            s0, s5 = values[name][col]
            if pd.isna(s0):
                row += f"  {'---':>7} {'---':>7} {'---':>7}"
            else:
                gap = s5 - s0
                marker = "★" if np.isclose(gap, min_gap[col], atol=1e-4) else " "
                row += f"  {s0:>7.3f} {s5:>7.3f} {gap:>+6.3f}{marker}"
        print(row)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--summaries",
        nargs="+",
        default=["outputs/metrics_results/*/*.csv"],
        help="glob pattern(s) for eval_summary CSVs",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="output CSV path for the BERTScore averages",
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

    # Per-temperature tables
    PRESETS = [(0.3, "conservateur"), (0.5, "standard"), (0.8, "exploratoire")]

    id_to_name = {hf: name for hf, name, _ in MODEL_ORDER}
    id_to_size = {hf: size for hf, _, size in MODEL_ORDER}
    df2 = df.copy()
    df2["model_name"] = df2["model"].map(id_to_name)
    df2["size"] = df2["model"].map(id_to_size)
    df2 = df2.dropna(subset=["model_name"])

    rows = []
    for temp, preset_name in PRESETS:
        sub = df[np.isclose(df["temperature"], temp)]
        print_table(sub, title=f"BERTScore F1  —  T={temp} ({preset_name})")

        sub2 = df2[np.isclose(df2["temperature"], temp)]
        for _, name, size in MODEL_ORDER:
            m = sub2[sub2["model_name"] == name]
            for col, label in ENCODERS:
                if col not in m.columns:
                    continue
                s0 = m[m["prompt_type"] == "0shot"][col].mean()
                s5 = m[m["prompt_type"] == "5shot"][col].mean()
                rows.append({
                    "temperature": temp,
                    "preset": preset_name,
                    "model_name": name,
                    "size": size,
                    "encoder": label,
                    "zero_shot": s0,
                    "five_shot": s5,
                    "gap": s5 - s0,
                })

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, float_format="%.4f")
    print(f"\n→ BERTScore per-temperature saved to: {args.output}")


if __name__ == "__main__":
    main()
