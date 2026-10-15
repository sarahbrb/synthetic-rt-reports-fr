"""
Automatic evaluation of synthetic reports produced by generate_cag.py

Metrics computed per report:
  n_words         word count
  msttr_100       lexical richness (length-corrected TTR, 100-word windows)
  flesch_re_fr    readability (French Flesch Reading Ease; lower = more technical)
  gunning_fog_fr  readability (Gunning Fog index; higher = more complex text)
  perplexite      surface fluency via a French causal LM (asi/gpt-fr-cased-base)
  bertscore_max   max BERTScore F1 against the 5 real reports embedded in the
                  5-shot prompt

Metrics aggregated per group (dataset × prompt_type × model × sampling config):
  mean and std of all per-report metrics above
  self_bleu_4             corpus diversity (higher = reports look too similar to each other)
  distinct_1 / distinct_2 unique unigram / bigram ratio across the group (higher = more diverse)
  mean_pairwise_cosine    mean TF-IDF cosine similarity between all pairs in the group (lower = more diverse)

Dependencies:
    pip install textstat sacrebleu bert-score scikit-learn

Usage:
    python scripts/evaluate_synthetic_reports.py \\
        --input outputs/generated_CR/qwen7b/*.jsonl \\
        --out_per_cr  outputs/metrics_results/qwen7b/qwen7b_eval_per_cr.csv \\
        --out_summary outputs/metrics_results/qwen7b/qwen7b_eval_summary.csv
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
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR   = Path(__file__).parent.parent
PROMPT_DIR = BASE_DIR / "prompts"

# The 5-shot prompts embed anonymised reports as in-context examples
# We extract those examples at eval time to check that the model hasn't reproduced them verbatim (BERTScore anti-leakage check)
REF_PROMPT_5SHOT = {
    "metabric": PROMPT_DIR / "user_prompt_metabric_5shot.txt",
    "seer":     PROMPT_DIR / "user_prompt_seer_5shot.txt",
}

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)*")


def load_records(paths: List[Path]) -> List[Dict]:
    records = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def extract_reference_examples(dataset: str) -> List[str]:
    """Pull the 5 real reports out of the 5-shot prompt file.
    They are expected to appear as 'Exemple N :' blocks separated by '---'."""
    path    = REF_PROMPT_5SHOT[dataset]
    text    = path.read_text(encoding="utf-8")
    pattern = re.compile(r"Exemple\s+\d+\s*:\s*\n(.*?)\n\s*\n---", re.S)
    examples = [ex.strip() for ex in pattern.findall(text)]
    if len(examples) != 5:
        print(f"[warn] found {len(examples)} example(s) in {path.name}, expected 5")
    return examples


def tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def ttr(tokens: List[str]) -> float:
    if not tokens:
        return float("nan")
    return len(set(tokens)) / len(tokens)


def msttr_100(tokens: List[str], window: int = 100) -> float:
    """Mean Segmental TTR over non-overlapping 100-word windows.
    Corrects the length bias of raw TTR (Covington & McFall, 2010)."""
    if not tokens:
        return float("nan")
    if len(tokens) < window:
        return ttr(tokens)
    chunks = [tokens[i : i + window] for i in range(0, len(tokens), window)]
    chunks = [c for c in chunks if len(c) == window]
    return float(np.mean([ttr(c) for c in chunks]))


def flesch_re_fr(text: str) -> float:
    try:
        import textstat
        textstat.set_lang("fr")
        return float(textstat.flesch_reading_ease(text))
    except Exception as e:
        print(f"[warn] flesch_re_fr failed: {e}")
        return float("nan")


def gunning_fog_fr(text: str) -> float:
    try:
        import textstat
        textstat.set_lang("fr")
        return float(textstat.gunning_fog(text))
    except Exception as e:
        print(f"[warn] gunning_fog_fr failed: {e}")
        return float("nan")


class FluencyScorer:
    """Perplexity via a French causal LM. Lower perplexity = more natural text."""

    def __init__(self, model_name: str, device: str):
        print(f"Loading fluency model: {model_name}")
        self.device    = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        enc      = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(self.device)
        if input_ids.shape[1] < 2:
            return float("nan")
        out = self.model(input_ids, labels=input_ids)
        return float(np.exp(out.loss.item()))


def _get_num_layers(model_type: str) -> int:
    """Return num transformer layers for a HF model.
    bert_score's built-in list doesn't cover community models like camembert-bio;
    fall back to reading num_hidden_layers from the HuggingFace config.
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
    lang: str = "fr",
    model_type: str | None = None,
) -> Tuple[List[float], List[int]]:
    """For each generated text, return the max BERTScore F1 over the reference
    set and the index of the closest reference.
    Pass model_type to use a specific encoder (e.g. a biomedical CamemBERT);
    otherwise lang selects the default encoder for that language."""
    from bert_score import score as bertscore_score

    score_kwargs = dict(device=device, verbose=False)
    if model_type:
        score_kwargs["model_type"] = model_type
        score_kwargs["num_layers"] = _get_num_layers(model_type)
    else:
        score_kwargs["lang"] = lang

    all_f1 = np.zeros((len(references), len(texts)))
    for ref_idx, ref_text in enumerate(references):
        _, _, f1 = bertscore_score(
            cands=texts,
            refs=[ref_text] * len(texts),
            **score_kwargs,
        )
        all_f1[ref_idx] = f1.numpy()

    return all_f1.max(axis=0).tolist(), all_f1.argmax(axis=0).tolist()


def self_bleu(texts: List[str], max_ngram_order: int = 4) -> float:
    """Mean Self-BLEU-4: each text scored against all others as references.
    Higher score = corpus is less diverse"""
    from sacrebleu.metrics import BLEU

    if len(texts) < 2:
        return float("nan")
    bleu   = BLEU(max_ngram_order=max_ngram_order, effective_order=True)
    scores = [
        bleu.sentence_score(hyp, texts[:i] + texts[i + 1:]).score
        for i, hyp in enumerate(texts)
    ]
    return float(np.mean(scores))


def distinct_n(texts: List[str], n: int) -> float:
    """Corpus-level Distinct-n: unique n-grams / total n-grams (Li et al., 2016).
    Higher = more lexically diverse."""
    total, unique = 0, set()
    for text in texts:
        tokens = tokenize_words(text)
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        total += len(ngrams)
        unique.update(ngrams)
    if total == 0:
        return float("nan")
    return len(unique) / total


def mean_pairwise_cosine(texts: List[str]) -> float:
    """Mean pairwise TF-IDF cosine similarity over all pairs in the group.
    Lower = more semantically diverse."""
    if len(texts) < 2:
        return float("nan")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vecs = TfidfVectorizer().fit_transform(texts)
    sim  = cosine_similarity(vecs)
    n    = sim.shape[0]
    upper = sim[np.triu_indices(n, k=1)]
    return float(upper.mean())


def main():
    ap = argparse.ArgumentParser(description="CAG — automatic evaluation of generated reports")
    ap.add_argument("--input", nargs="+", required=True,
                    help="one or more .jsonl files from generate_cag.py")
    ap.add_argument("--out_per_cr",  required=True, help="per-report CSV")
    ap.add_argument("--out_summary", required=True, help="per-group aggregated CSV")
    ap.add_argument("--lm_model", default="asi/gpt-fr-cased-base",
                    help="French causal LM for perplexity scoring")
    ap.add_argument("--bertscore_alert_threshold", type=float, default=0.92,
                    help="BERTScore above this threshold is flagged as a possible copy")
    ap.add_argument("--bertscore_model", default=None,
                    help="HuggingFace model ID to use as BERTScore encoder "
                         "(e.g. 'almanach/camembert-bio-base'). "
                         "If omitted, the default French encoder is used (lang='fr').")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_perplexity", action="store_true")
    ap.add_argument("--skip_bertscore",  action="store_true")
    args = ap.parse_args()

    records = load_records([Path(p) for p in args.input])
    print(f"{len(records)} reports loaded from {len(args.input)} file(s)")

    # Per-report metrics (fast, no GPU needed)
    rows = []
    for rec in records:
        text   = rec["cr_text"]
        tokens = tokenize_words(text)
        rows.append({
            "idx":                rec["idx"],
            "dataset":            rec["dataset"],
            "prompt_type":        rec["prompt_type"],
            "model":              rec["model"],
            "temperature":        rec.get("temperature",        float("nan")),
            "top_p":              rec.get("top_p",              float("nan")),
            "repetition_penalty": rec.get("repetition_penalty", float("nan")),
            "n_mots":             len(tokens),
            "msttr_100":          msttr_100(tokens),
            "flesch_re_fr":       flesch_re_fr(text),
            "gunning_fog_fr":     gunning_fog_fr(text),
        })
    df = pd.DataFrame(rows)

    # Perplexity
    df["perplexite"] = float("nan")
    if not args.skip_perplexity:
        scorer = FluencyScorer(args.lm_model, args.device)
        df["perplexite"] = [scorer.perplexity(rec["cr_text"]) for rec in records]

    # BERTScore anti-leakage
    # Computed per dataset (0-shot and 5-shot pooled): we want to know whether any generated report copies the real examples embedded in the 5-shot prompt, regardless of whether the model was shown those examples or not
    df["bertscore_max"]    = float("nan")
    df["bertscore_ref_idx"] = -1
    df["bertscore_alerte"] = False
    if not args.skip_bertscore:
        for dataset in df["dataset"].unique():
            mask  = df["dataset"] == dataset
            texts = [rec["cr_text"] for rec in records if rec["dataset"] == dataset]
            refs  = extract_reference_examples(dataset)
            max_f1, ref_idx = bertscore_vs_references(
                texts, refs, device=args.device, model_type=args.bertscore_model
            )
            df.loc[mask, "bertscore_max"]     = max_f1
            df.loc[mask, "bertscore_ref_idx"] = ref_idx
            df.loc[mask, "bertscore_alerte"]  = (
                df.loc[mask, "bertscore_max"] > args.bertscore_alert_threshold
            )

    n_alerts = int(df["bertscore_alerte"].sum())
    if n_alerts:
        print(f"[alert] {n_alerts} report(s) suspiciously close to a real example "
              f"(BERTScore > {args.bertscore_alert_threshold}) — review manually.")

    df.to_csv(args.out_per_cr, index=False)
    print(f"Per-report detail  →  {args.out_per_cr}")

    # Aggregation per group
    group_cols  = ["dataset", "prompt_type", "model", "temperature", "top_p", "repetition_penalty"]
    metric_cols = ["n_mots", "msttr_100", "flesch_re_fr", "gunning_fog_fr", "perplexite", "bertscore_max"]

    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()

    # Corpus-level metrics: computed per group, not per report
    group_corpus_rows = []
    for keys, sub in df.groupby(group_cols):
        texts = [records[i]["cr_text"] for i in sub.index]
        group_corpus_rows.append({
            **dict(zip(group_cols, keys)),
            "self_bleu_4":          self_bleu(texts),
            "distinct_1":           distinct_n(texts, 1),
            "distinct_2":           distinct_n(texts, 2),
            "mean_pairwise_cosine": mean_pairwise_cosine(texts),
        })
    summary = summary.merge(pd.DataFrame(group_corpus_rows), on=group_cols, how="left")

    if n_alerts:
        alert_counts = (
            df.groupby(group_cols)["bertscore_alerte"].sum().rename("n_alertes_bertscore")
        )
        summary = summary.merge(alert_counts, on=group_cols, how="left")

    summary.to_csv(args.out_summary, index=False)
    print(f"Group summary      →  {args.out_summary}")
    print(f"Done. {len(df)} reports evaluated across {len(summary)} group(s).")


if __name__ == "__main__":
    main()
