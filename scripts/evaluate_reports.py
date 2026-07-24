"""
Linguistic quality evaluation for a set of medical reports (.txt files).

Designed to evaluate real clinical reports and compare them to synthetic ones
produced by the CAG pipeline (see evaluate_cag.py for the synthetic side).

Metrics:
  n_words      word count
  msttr_100    lexical richness (length-corrected TTR, 100-word windows)
  flesch_re    readability 
  perplexity   surface fluency via a French causal LM 
  self_bleu_4  corpus-level diversity, higher means reports look too similar

The language model for perplexity (~700 MB) is downloaded automatically from
HuggingFace on first use.

Usage:
    python scripts/evaluate_reports.py \\
        --input_dir /path/to/reports/ \\
        --output results.csv \\
        --lm_model asi/gpt-fr-cased-base \\
        --device cuda
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)*")


def load_reports(input_dir: Path) -> dict:
    """Load all .txt files from a directory. Returns {filename: text}."""
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in: {input_dir}")
    reports = {}
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"Skipping empty file: {path.name}")
            continue
        reports[path.name] = text
    print(f"{len(reports)} report(s) loaded from {input_dir}")
    return reports


def tokenize_words(text: str) -> list:
    return [w.lower() for w in WORD_RE.findall(text)]


def ttr(tokens: list) -> float:
    return len(set(tokens)) / len(tokens) if tokens else float("nan")


def msttr_100(tokens: list, window: int = 100) -> float:
    """Mean Segmental TTR over non-overlapping 100-word windows.
    Avoids the length bias of raw TTR (Covington & McFall, 2010)."""
    if not tokens:
        return float("nan")
    if len(tokens) < window:
        return ttr(tokens)
    chunks = [tokens[i : i + window]
              for i in range(0, len(tokens), window)
              if i + window <= len(tokens)]
    return float(np.mean([ttr(c) for c in chunks]))


def flesch_reading_ease(text: str) -> float:
    """French Flesch Reading Ease via textstat. Clinical CRs typically score 20-50."""
    try:
        import textstat
        textstat.set_lang("fr")
        return round(float(textstat.flesch_reading_ease(text)), 2)
    except ImportError:
        print("textstat not installed — run: pip install textstat")
        return float("nan")
    except Exception as e:
        print(f"Flesch RE failed: {e}")
        return float("nan")


class FluencyScorer:
    """Perplexity scoring using a French causal language model."""

    def __init__(self, model_name: str, device: str):
        print(f"Loading fluency model: {model_name}")
        self.device    = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        """Texts over 1024 tokens are truncated."""
        enc      = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(self.device)
        if input_ids.shape[1] < 2:
            return float("nan")
        out = self.model(input_ids, labels=input_ids)
        return round(float(np.exp(out.loss.item())), 3)


def self_bleu_4(texts: list) -> float:
    """Mean corpus Self-BLEU-4: each report scored against all others as references.
    Higher = less diverse corpus (Zhu et al., 2018)."""
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        print("sacrebleu not installed — run: pip install sacrebleu")
        return float("nan")

    if len(texts) < 2:
        print("Self-BLEU requires at least 2 reports.")
        return float("nan")

    bleu   = BLEU(max_ngram_order=4, effective_order=True)
    scores = [
        bleu.sentence_score(hyp, texts[:i] + texts[i + 1:]).score
        for i, hyp in enumerate(texts)
    ]
    return round(float(np.mean(scores)), 2)


def main():
    parser = argparse.ArgumentParser(
        description="Linguistic quality evaluation of medical reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/evaluate_reports.py"
            " --input_dir ./reports/ --output results.csv"
            " --lm_model asi/gpt-fr-cased-base --device cuda"
        ),
    )
    parser.add_argument("--input_dir", required=True,
                        help="directory of .txt reports (UTF-8, one file per report)")
    parser.add_argument("--output", required=True,
                        help="output CSV path; a *_summary.csv is also written alongside it")
    parser.add_argument("--lm_model", default="asi/gpt-fr-cased-base",
                        help="HuggingFace causal LM for perplexity (default: asi/gpt-fr-cased-base)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="'cuda' (recommended) or 'cpu'")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Directory not found: {input_dir}")

    output_path  = Path(args.output)
    summary_path = output_path.with_name(output_path.stem + "_summary" + output_path.suffix)

    reports = load_reports(input_dir)
    texts   = list(reports.values())

    # Surface metrics (no GPU needed)
    rows = []
    for fname, text in reports.items():
        tokens = tokenize_words(text)
        rows.append({
            "file":      fname,
            "n_words":   len(tokens),
            "msttr_100": round(msttr_100(tokens), 4),
            "flesch_re": flesch_reading_ease(text),
        })
    df = pd.DataFrame(rows)

    # Perplexity
    print(f"Computing perplexity ({args.device}) ...")
    scorer   = FluencyScorer(args.lm_model, args.device)
    ppl_vals = []
    for i, (fname, text) in enumerate(reports.items()):
        ppl = scorer.perplexity(text)
        ppl_vals.append(ppl)
        print(f"  [{i+1}/{len(texts)}] {fname}  PPL={ppl}")
    df["perplexity"] = ppl_vals

    # Self-BLEU-4 is corpus-level : same value repeated in every row
    print("Computing Self-BLEU-4 ...")
    sb4 = self_bleu_4(texts)
    df["self_bleu_4"] = sb4

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Per-report detail  →  {output_path}")

    metric_cols = ["n_words", "msttr_100", "flesch_re", "perplexity"]
    summary = df[metric_cols].agg(["mean", "std"]).T.round(4)
    summary.columns = ["mean", "std"]
    summary.loc["self_bleu_4"] = [sb4, float("nan")]
    summary.to_csv(summary_path, encoding="utf-8")
    print(f"Corpus summary     →  {summary_path}")

    print(f"\nResults : {len(texts)} report(s)")
    labels = {
        "n_words":    "Word count (mean)",
        "msttr_100":  "Lexical richness  (MSTTR-100)",
        "flesch_re":  "Readability       (Flesch RE)",
        "perplexity": "Perplexity        (lower = more fluent)",
    }
    means = df[metric_cols].mean()
    stds  = df[metric_cols].std()
    for col, label in labels.items():
        if pd.isna(means[col]):
            print(f"  {label:<45}  n/a")
        else:
            print(f"  {label:<45}  {means[col]:>7.3f}  ± {stds[col]:.3f}")
    print(f"  {'Diversity (Self-BLEU-4)  (lower = more diverse)':<45}  {sb4:>7.2f}")


if __name__ == "__main__":
    main()
