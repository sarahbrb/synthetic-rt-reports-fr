"""
Évaluation automatique des CR synthétiques générés par generate_cag.py.

Métriques calculées :

  Par CR :
    n_mots          — nombre de mots
    msttr_100       — richesse lexicale (MSTTR, fenêtre 100 mots)
    flesch_re_fr    — lisibilité (Flesch Reading Ease, adaptation française)
    perplexite      — fluidité de surface via un LM causal français
    bertscore_max   — similarité max aux 5 exemples réels du prompt 5-shot
                      (garde-fou anti-leakage ; élevé = suspicion de recopie)

  Par groupe (dataset × prompt_type × model × sampling) :
    moyennes et écarts-types des métriques ci-dessus
    self_bleu_4     — diversité du corpus (élevé = les CR se ressemblent trop)

Dépendances :
    pip install textstat sacrebleu bert-score

Usage :
    conda activate synthgen
    python scripts/evaluate_cag.py \\
        --input outputs/metabric_0shot_qwen7b.jsonl outputs/metabric_5shot_qwen7b.jsonl \\
        --out_per_cr outputs/eval_per_cr.csv \\
        --out_summary outputs/eval_summary.csv
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

BASE_DIR = Path(__file__).parent.parent
PROMPT_DIR = BASE_DIR / "prompts"

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
    """Extrait les 5 CR réels intégrés dans le prompt 5-shot du dataset donné."""
    path = REF_PROMPT_5SHOT[dataset]
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"Exemple\s+\d+\s*:\s*\n(.*?)\n\s*\n---", re.S)
    examples = [ex.strip() for ex in pattern.findall(text)]
    if len(examples) != 5:
        print(f"[WARN] {len(examples)} exemple(s) trouvé(s) dans {path.name} (5 attendus)")
    return examples


def tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def ttr(tokens: List[str]) -> float:
    if not tokens:
        return float("nan")
    return len(set(tokens)) / len(tokens)


def msttr_100(tokens: List[str], window: int = 100) -> float:
    """TTR moyenné sur des fenêtres de `window` mots (corrige le biais de longueur du TTR brut)."""
    if not tokens:
        return float("nan")
    if len(tokens) < window:
        return ttr(tokens)
    chunks = [tokens[i:i + window] for i in range(0, len(tokens), window)]
    chunks = [c for c in chunks if len(c) == window]
    return float(np.mean([ttr(c) for c in chunks]))


def flesch_re_fr(text: str) -> float:
    try:
        import textstat
        textstat.set_lang("fr")
        return float(textstat.flesch_reading_ease(text))
    except Exception as e:
        print(f"[WARN] flesch_re_fr a échoué : {e}")
        return float("nan")


class FluencyScorer:
    """Calcule la perplexité d'un texte via un LM causal français."""

    def __init__(self, model_name: str, device: str):
        print(f"[LM]    chargement : {model_name}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(self.device)
        if input_ids.shape[1] < 2:
            return float("nan")
        out = self.model(input_ids, labels=input_ids)
        return float(np.exp(out.loss.item()))


def bertscore_vs_references(
    texts: List[str],
    references: List[str],
    device: str,
    lang: str = "fr",
) -> Tuple[List[float], List[int]]:
    """Pour chaque texte : BERTScore F1 max sur les références, et indice de la référence la plus proche."""
    from bert_score import score as bertscore_score

    all_f1 = np.zeros((len(references), len(texts)))
    for ref_idx, ref_text in enumerate(references):
        _, _, f1 = bertscore_score(
            cands=texts,
            refs=[ref_text] * len(texts),
            lang=lang,
            device=device,
            verbose=False,
        )
        all_f1[ref_idx] = f1.numpy()

    return all_f1.max(axis=0).tolist(), all_f1.argmax(axis=0).tolist()


def self_bleu(texts: List[str], max_ngram_order: int = 4) -> float:
    """Self-BLEU moyen du corpus : chaque texte scoré contre tous les autres.
    Un score élevé indique un manque de diversité."""
    from sacrebleu.metrics import BLEU

    if len(texts) < 2:
        return float("nan")
    bleu = BLEU(max_ngram_order=max_ngram_order, effective_order=True)
    scores = [
        bleu.sentence_score(hyp, texts[:i] + texts[i + 1:]).score
        for i, hyp in enumerate(texts)
    ]
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser(description="CAG — évaluation automatique des CR générés")
    ap.add_argument("--input", nargs="+", required=True,
                    help="un ou plusieurs fichiers .jsonl produits par generate_cag.py")
    ap.add_argument("--out_per_cr",  required=True, help="CSV détaillé, une ligne par CR")
    ap.add_argument("--out_summary", required=True, help="CSV agrégé, une ligne par groupe")
    ap.add_argument("--lm_model", default="asi/gpt-fr-cased-base",
                    help="LM causal français pour la perplexité")
    ap.add_argument("--bertscore_alert_threshold", type=float, default=0.92,
                    help="seuil BERTScore au-delà duquel une génération est flaggée comme suspecte")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_perplexity", action="store_true")
    ap.add_argument("--skip_bertscore",  action="store_true")
    args = ap.parse_args()

    records = load_records([Path(p) for p in args.input])
    print(f"[DATA]  {len(records)} CR chargés depuis {len(args.input)} fichier(s)")

    # métriques par CR
    rows = []
    for rec in records:
        text = rec["cr_text"]
        tokens = tokenize_words(text)
        rows.append({
            "idx":                rec["idx"],
            "dataset":            rec["dataset"],
            "prompt_type":        rec["prompt_type"],
            "model":              rec["model"],
            "temperature":        rec.get("temperature", float("nan")),
            "top_p":              rec.get("top_p", float("nan")),
            "repetition_penalty": rec.get("repetition_penalty", float("nan")),
            "n_mots":             len(tokens),
            "msttr_100":          msttr_100(tokens),
            "flesch_re_fr":       flesch_re_fr(text),
        })
    df = pd.DataFrame(rows)

    # perplexité
    df["perplexite"] = float("nan")
    if not args.skip_perplexity:
        scorer = FluencyScorer(args.lm_model, args.device)
        df["perplexite"] = [scorer.perplexity(rec["cr_text"]) for rec in records]

    # bertscore anti-leakage (calculé dataset par dataset, 0-shot et 5-shot confondus)
    df["bertscore_max"]    = float("nan")
    df["bertscore_ref_idx"] = -1
    df["bertscore_alerte"] = False
    if not args.skip_bertscore:
        for dataset in df["dataset"].unique():
            mask  = df["dataset"] == dataset
            texts = [rec["cr_text"] for rec in records if rec["dataset"] == dataset]
            refs  = extract_reference_examples(dataset)
            max_f1, ref_idx = bertscore_vs_references(texts, refs, device=args.device)
            df.loc[mask, "bertscore_max"]     = max_f1
            df.loc[mask, "bertscore_ref_idx"] = ref_idx
            df.loc[mask, "bertscore_alerte"]  = (
                df.loc[mask, "bertscore_max"] > args.bertscore_alert_threshold
            )

    n_alertes = int(df["bertscore_alerte"].sum())
    if n_alertes:
        print(f"[ALERTE] {n_alertes} CR anormalement proche(s) d'un exemple réel "
              f"(BERTScore > {args.bertscore_alert_threshold}) — à vérifier manuellement.")

    df.to_csv(args.out_per_cr, index=False)
    print(f"[SAVE]  détail par CR -> {args.out_per_cr}")

    # agrégation par groupe
    group_cols  = ["dataset", "prompt_type", "model", "temperature", "top_p", "repetition_penalty"]
    metric_cols = ["n_mots", "msttr_100", "flesch_re_fr", "perplexite", "bertscore_max"]

    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()

    # self-BLEU-4 par groupe
    sb4_rows = []
    for keys, sub in df.groupby(group_cols):
        texts = [records[i]["cr_text"] for i in sub.index]
        sb4_rows.append({**dict(zip(group_cols, keys)), "self_bleu_4": self_bleu(texts)})
    summary = summary.merge(pd.DataFrame(sb4_rows), on=group_cols, how="left")

    if n_alertes:
        alert_counts = (
            df.groupby(group_cols)["bertscore_alerte"].sum().rename("n_alertes_bertscore")
        )
        summary = summary.merge(alert_counts, on=group_cols, how="left")

    summary.to_csv(args.out_summary, index=False)
    print(f"[SAVE]  résumé par groupe -> {args.out_summary}")
    print(f"[DONE]  {len(df)} CR évalués — {len(summary)} groupe(s)")


if __name__ == "__main__":
    main()
