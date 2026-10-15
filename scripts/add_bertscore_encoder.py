"""
Recompute BERTScore with a different encoder for already-evaluated models.

Reads existing per-CR CSVs and the original JSONL files, adds a new column
`bertscore_max_bio` (without touching the original `bertscore_max` column),
and updates the summary CSV accordingly.

Usage (one model at a time):
    python scripts/add_bertscore_encoder.py \\
        --input  outputs/generated_CR/qwen7b/*.jsonl \\
        --per_cr outputs/metrics_results/qwen7b/qwen7b_eval_per_cr.csv \\
        --summary outputs/metrics_results/qwen7b/qwen7b_eval_summary.csv \\
        --bertscore_model almanach/camembert-bio-base

Verify the model ID on HuggingFace before running.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

BASE_DIR   = Path(__file__).parent.parent
PROMPT_DIR = BASE_DIR / "prompts"

REF_PROMPT_5SHOT = {
    "metabric": PROMPT_DIR / "user_prompt_metabric_5shot.txt",
    "seer":     PROMPT_DIR / "user_prompt_seer_5shot.txt",
}


def extract_reference_examples(dataset: str) -> List[str]:
    path    = REF_PROMPT_5SHOT[dataset]
    text    = path.read_text(encoding="utf-8")
    pattern = re.compile(r"Exemple\s+\d+\s*:\s*\n(.*?)\n\s*\n---", re.S)
    examples = [ex.strip() for ex in pattern.findall(text)]
    if len(examples) != 5:
        print(f"[warn] found {len(examples)} example(s) in {path.name}, expected 5")
    return examples


def _get_num_layers(model_type: str) -> int:
    """Return the number of transformer layers for a HF model.
    bert_score only knows its built-in list; for any other model we read
    num_hidden_layers directly from the HuggingFace config.
    """
    from bert_score.utils import model2layers
    if model_type in model2layers:
        return model2layers[model_type]
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_type)
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None:
        raise ValueError(f"Cannot determine num_hidden_layers for {model_type}")
    print(f"[info] {model_type}: num_layers={n} (read from HF config)")
    return n


def bertscore_vs_references(
    texts: List[str],
    references: List[str],
    device: str,
    model_type: str,
) -> Tuple[List[float], List[int]]:
    from bert_score import score as bertscore_score

    num_layers = _get_num_layers(model_type)
    all_f1 = np.zeros((len(references), len(texts)))
    for ref_idx, ref_text in enumerate(references):
        _, _, f1 = bertscore_score(
            cands=texts,
            refs=[ref_text] * len(texts),
            model_type=model_type,
            num_layers=num_layers,
            device=device,
            verbose=False,
        )
        all_f1[ref_idx] = f1.numpy()

    return all_f1.max(axis=0).tolist(), all_f1.argmax(axis=0).tolist()


def main():
    ap = argparse.ArgumentParser(description="Recompute BERTScore with a biomedical encoder")
    ap.add_argument("--input",    nargs="+", required=True, help="JSONL files (same as original eval)")
    ap.add_argument("--per_cr",   required=True, help="existing per-CR CSV to update in place")
    ap.add_argument("--summary",  required=True, help="existing summary CSV to update in place")
    ap.add_argument("--bertscore_model", required=True,
                    help="HuggingFace encoder model ID (e.g. 'almanach/camembert-bio-base')")
    ap.add_argument("--out_col",  default="bertscore_max_bio",
                    help="name of the new column added to per_cr CSV (default: bertscore_max_bio)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Load JSONL records
    records = []
    for p in args.input:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    print(f"{len(records)} records loaded")

    # Load existing per-CR CSV
    df = pd.read_csv(args.per_cr)

    # Sanity check
    if len(df) != len(records):
        raise ValueError(
            f"Row count mismatch: per_cr has {len(df)} rows but JSONL has {len(records)} records. "
            "Make sure --input matches the original evaluation run."
        )

    # Compute BERTScore per dataset
    new_col    = args.out_col
    ref_col    = new_col + "_ref_idx"
    df[new_col] = float("nan")
    df[ref_col] = -1

    for dataset in df["dataset"].unique():
        mask  = df["dataset"] == dataset
        texts = [records[i]["cr_text"] for i in df[mask].index]
        refs  = extract_reference_examples(dataset)
        print(f"Computing {new_col} for dataset={dataset}, "
              f"n={len(texts)}, n_refs={len(refs)}, model={args.bertscore_model}")
        max_f1, ref_idx = bertscore_vs_references(texts, refs, args.device, args.bertscore_model)
        df.loc[mask, new_col] = max_f1
        df.loc[mask, ref_col] = ref_idx

    df.to_csv(args.per_cr, index=False)
    print(f"Updated per-CR CSV -> {args.per_cr}  (added columns: {new_col}, {ref_col})")

    # Update summary CSV: add mean/std of new column per group
    group_cols = ["dataset", "prompt_type", "model", "temperature", "top_p", "repetition_penalty"]
    summary = pd.read_csv(args.summary)
    agg = df.groupby(group_cols)[new_col].agg(["mean", "std"]).reset_index()
    agg.columns = list(group_cols) + [f"{new_col}_mean", f"{new_col}_std"]

    # Drop old bio columns if re-running
    for col in [f"{new_col}_mean", f"{new_col}_std"]:
        if col in summary.columns:
            summary.drop(columns=[col], inplace=True)

    summary = summary.merge(agg, on=group_cols, how="left")
    summary.to_csv(args.summary, index=False)
    print(f"Updated summary CSV  -> {args.summary}  (added columns: {new_col}_mean, {new_col}_std)")
    print("Done.")


if __name__ == "__main__":
    main()
