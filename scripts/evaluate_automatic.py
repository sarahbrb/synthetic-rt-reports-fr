"""
===================================================
AUTOMATIC EVALUATION OF RADIOTHERAPY REPORTS
===================================================

Metrics computed:
  Lexical richness  : TTR, MSTTR-100, Lexical Entropy, Hapax Ratio
  Readability       : Flesch Reading Ease (French adaptation)
  Fluency           : Pseudo-Perplexity (CamemBERT MLM only — no estimate fallback)
                      SLOR (only if perplexity is available)
  Diversity         : Self-BLEU-2, Self-BLEU-3, Self-BLEU-4
  Text statistics   : word_count, char_count, sentence_count,
                      avg_sentence_length, long_word_ratio

NOTE: Perplexity and SLOR require --use_gpu_ppl and a working GPU.
      If CamemBERT cannot be loaded, these two metrics are simply absent
      from the output (no estimation fallback).

Usage:
    python scripts/evaluate_automatic.py \
  --input outputs/test_mistral7b_gen.jsonl \
  --output evaluation/test_mistral7b_gen_metrics.json

Dependencies:
    pip install numpy torch transformers
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Optional heavy deps ───────────────────────────────────────────────────────
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import CamembertForMaskedLM, CamembertTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("[WARN] torch/transformers not available — perplexity and SLOR will be skipped.")


# =============================================================================
# 1. TOKENISATION
# =============================================================================

def tokenize_french(text: str) -> List[str]:
    """Lowercase French tokeniser — keeps accented characters, strips digits/units."""
    return re.findall(r"\b[a-zàâäéèêëïîôùûüÿçœæ]+\b", text.lower())


def count_syllables_french(word: str) -> int:
    """Approximate syllable count for French words."""
    word = word.lower()
    vowels = "aeiouyàâäéèêëïîôùûüÿœæ"
    count, prev_vowel = 0, False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    if word.endswith(("es", "ent")) and count > 1:
        count -= 1
    return max(1, count)


def split_sentences(text: str) -> List[str]:
    """Sentence boundary detection for clinical French text.

    Splits at sentence-ending punctuation followed by whitespace + uppercase letter,
    or at paragraph breaks (double newline).  Filters fragments shorter than 5 tokens
    to exclude header lines (date, reference code, recipient list) that would otherwise
    inflate sentence count and deflate average sentence length.
    """
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜŒÆ])|(?:\n\n+)', text)
    return [s.strip() for s in parts if s.strip() and len(s.split()) >= 5]


# =============================================================================
# 2. LEXICAL RICHNESS
# =============================================================================

def compute_ttr(text: str) -> float:
    """Type-Token Ratio.  >0.5 excellent | 0.4–0.5 good | <0.3 poor."""
    tokens = tokenize_french(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_msttr(text: str, window: int = 100) -> float:
    """Mean Segmental TTR (non-overlapping windows of `window` tokens).
    More robust than TTR for long texts.  >0.6 good."""
    tokens = tokenize_french(text)
    if len(tokens) < window:
        return compute_ttr(text)
    ttrs = [
        len(set(tokens[i : i + window])) / window
        for i in range(0, len(tokens) - window + 1, window)
    ]
    return float(np.mean(ttrs)) if ttrs else 0.0


def compute_lexical_entropy(text: str) -> float:
    """Shannon entropy H = -Σ p(w) log₂ p(w).  Typical range 5–10."""
    tokens = tokenize_french(text)
    if not tokens:
        return 0.0
    total = len(tokens)
    freq = Counter(tokens)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def compute_hapax_ratio(text: str) -> float:
    """Proportion of words appearing exactly once (hapax legomena)."""
    tokens = tokenize_french(text)
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    return sum(1 for c in freq.values() if c == 1) / len(tokens)


# =============================================================================
# 3. READABILITY
# =============================================================================

def compute_flesch_reading_ease_fr(text: str) -> float:
    """Flesch Reading Ease adapted for French (Kandel-Moles 1958).
    Formula: 207 − 1.015 × (words/sentence) − 73.6 × (syllables/word)
    Medical reports: target < 40 (complex).  Easy text ≈ 60–90."""
    sentences = split_sentences(text)
    tokens = tokenize_french(text)
    if not sentences or not tokens:
        return 0.0
    syllables = sum(count_syllables_french(w) for w in tokens)
    wps = len(tokens) / len(sentences)
    spw = syllables / len(tokens)
    return float(max(0.0, min(100.0, 207.0 - 1.015 * wps - 73.6 * spw)))


# =============================================================================
# 4. SELF-BLEU (inter-text diversity)
# =============================================================================

def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _bleu_pair(reference: List[str], hypothesis: List[str], max_n: int) -> float:
    if not hypothesis:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        ref_ng = Counter(_ngrams(reference, n))
        hyp_ng = Counter(_ngrams(hypothesis, n))
        if not hyp_ng:
            precisions.append(0.0)
            continue
        matches = sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
        total = sum(hyp_ng.values())
        precisions.append(matches / total if total else 0.0)
    if all(p > 0 for p in precisions):
        bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        bleu = 0.0
    bp = min(1.0, math.exp(1 - len(reference) / len(hypothesis))) if hypothesis else 0.0
    return bleu * bp


def compute_self_bleu(texts: List[str], max_n: int = 4) -> float:
    """Self-BLEU: each text as hypothesis, all others as references.
    Lower = more diverse.  <0.3 excellent | 0.3–0.5 good | >0.7 low diversity."""
    if len(texts) < 2:
        return 0.0
    tokenized = [tokenize_french(t) for t in texts]
    scores = []
    for i, hyp in enumerate(tokenized):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        scores.append(float(np.mean([_bleu_pair(r, hyp, max_n) for r in refs])))
    return float(np.mean(scores)) if scores else 0.0


# =============================================================================
# 5. PERPLEXITY (CamemBERT MLM) — no estimate fallback
# =============================================================================

class PerplexityCalculator:
    """Pseudo-perplexity via masked language modelling (CamemBERT).

    For each token position i:
        mask xᵢ, forward-pass, take − log p(xᵢ | x\i)
    PPL = exp( mean of all NLLs )

    Requires GPU for reasonable speed.  Returns None if model not loaded.
    """

    def __init__(self, model_name: str = "camembert-base"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self) -> bool:
        if not TRANSFORMERS_AVAILABLE:
            print("[WARN] torch/transformers not available — perplexity skipped.")
            return False
        try:
            print(f"[...] Loading {self.model_name} …")
            self.tokenizer = CamembertTokenizer.from_pretrained(self.model_name)
            self.model = CamembertForMaskedLM.from_pretrained(self.model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            print(f"[OK]  CamemBERT loaded on {self.device}")
            return True
        except Exception as exc:
            print(f"[WARN] Cannot load {self.model_name}: {exc} — perplexity skipped.")
            return False

    def compute(self, text: str, batch_size: int = 32) -> Optional[float]:
        """Return pseudo-perplexity, or None if model not available."""
        if self.model is None:
            return None
        try:
            enc = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            input_ids = enc.input_ids.to(self.device)
            seq_len = input_ids.size(1)
            if seq_len <= 2:
                return None

            nlls: List[float] = []
            for start in range(1, seq_len - 1, batch_size):
                end = min(start + batch_size, seq_len - 1)
                n = end - start
                batch = input_ids.repeat(n, 1)
                targets = []
                for k, pos in enumerate(range(start, end)):
                    batch[k, pos] = self.tokenizer.mask_token_id
                    targets.append(input_ids[0, pos].item())
                with torch.no_grad():
                    logits = self.model(batch).logits
                for k, pos in enumerate(range(start, end)):
                    log_probs = torch.log_softmax(logits[k, pos], dim=-1)
                    nlls.append(-log_probs[targets[k]].item())

            return float(math.exp(np.mean(nlls))) if nlls else None
        except Exception as exc:
            print(f"[WARN] PPL error: {exc}")
            return None


# =============================================================================
# 6. SLOR
# =============================================================================

def compute_slor(text: str, ppl: float, unigram_ppl: float = 50.0) -> float:
    """Syntactic Log-Odds Ratio — normalises log P(S) by length and unigram baseline.
    SLOR = (log P(S) − log P_unigram(S)) / |S|.  Higher = more fluent."""
    tokens = tokenize_french(text)
    n = len(tokens)
    if n == 0 or ppl <= 0:
        return 0.0
    log_ps = -math.log(ppl) * n
    log_pu = -math.log(unigram_ppl) * n
    return (log_ps - log_pu) / n


# =============================================================================
# 7. TEXT STATISTICS
# =============================================================================

def compute_text_statistics(text: str) -> Dict[str, Any]:
    """word_count, char_count, sentence_count, avg_sentence_length, long_word_ratio."""
    tokens = tokenize_french(text)
    sentences = split_sentences(text)
    long_words = [t for t in tokens if len(t) > 6]
    return {
        "word_count":          len(tokens),
        "char_count":          len(text),
        "sentence_count":      len(sentences),
        "avg_sentence_length": round(len(tokens) / len(sentences), 2) if sentences else 0.0,
        "long_word_ratio":     round(len(long_words) / len(tokens), 4) if tokens else 0.0,
    }


# =============================================================================
# 8. PER-TEXT EVALUATION
# =============================================================================

def evaluate_single_text(
    text: str,
    ppl_calc: Optional[PerplexityCalculator] = None,
) -> Dict[str, Any]:
    """Evaluate one text.  Perplexity/SLOR absent if ppl_calc not loaded."""
    results: Dict[str, Any] = {}

    # Lexical richness
    results["ttr"]             = round(compute_ttr(text), 4)
    results["msttr_100"]       = round(compute_msttr(text, 100), 4)
    results["lexical_entropy"] = round(compute_lexical_entropy(text), 4)
    results["hapax_ratio"]     = round(compute_hapax_ratio(text), 4)

    # Readability
    results["flesch_reading_ease_fr"] = round(compute_flesch_reading_ease_fr(text), 2)


    # Perplexity + SLOR (only real CamemBERT, no estimate)
    ppl = ppl_calc.compute(text) if (ppl_calc and ppl_calc.model is not None) else None
    if ppl is not None:
        results["perplexity"] = round(ppl, 2)
        results["slor"]       = round(compute_slor(text, ppl), 4)

    # Text statistics
    results.update(compute_text_statistics(text))

    return results


# =============================================================================
# 9. CORPUS EVALUATION
# =============================================================================

def evaluate_corpus(
    texts: List[str],
    records: Optional[List[Dict]] = None,
    use_gpu_ppl: bool = False,
) -> Dict[str, Any]:
    """Evaluate all texts and aggregate results."""

    # Perplexity calculator
    ppl_calc: Optional[PerplexityCalculator] = None
    if use_gpu_ppl:
        ppl_calc = PerplexityCalculator()
        if not ppl_calc.load():
            ppl_calc = None

    # Per-text metrics
    print("[...] Computing per-text metrics …")
    individual: List[Dict] = []
    for i, text in enumerate(texts):
        individual.append(evaluate_single_text(text, ppl_calc))
        if (i + 1) % 10 == 0 or (i + 1) == len(texts):
            print(f"      {i+1}/{len(texts)} done")

    # Aggregation — collect all keys that actually appear
    all_keys: set = set()
    for m in individual:
        all_keys.update(m.keys())

    aggregated: Dict[str, Any] = {}
    for key in sorted(all_keys):
        values = [
            m[key] for m in individual
            if key in m and m[key] is not None and isinstance(m[key], (int, float))
        ]
        if not values:
            continue
        aggregated[f"{key}_mean"] = round(float(np.mean(values)), 4)
        aggregated[f"{key}_std"]  = round(float(np.std(values)),  4)
        aggregated[f"{key}_min"]  = round(float(np.min(values)),  4)
        aggregated[f"{key}_max"]  = round(float(np.max(values)),  4)

    # Self-BLEU
    print("[...] Computing Self-BLEU …")
    aggregated["self_bleu_4"] = round(float(compute_self_bleu(texts, 4)), 4)
    aggregated["self_bleu_3"] = round(float(compute_self_bleu(texts, 3)), 4)
    aggregated["self_bleu_2"] = round(float(compute_self_bleu(texts, 2)), 4)

    # Core clinical rate (from JSONL records if available)
    if records:
        ccr = [r.get("core_clinical_rate", 0) for r in records]
        aggregated["core_clinical_rate_mean"] = round(float(np.mean(ccr)), 4)
        aggregated["core_clinical_rate_std"]  = round(float(np.std(ccr)),  4)

    # Clinical rule violations (from JSONL records — filled by check_clinical_coherence)
    _violation_keys = [
        "v_hormono_rh_neg",
        "v_ct_notation_post_chir",
        "v_n_plus_no_ganglion",
        "v_m1_not_palliatif",
        "v_recidive_no_history",
        "v_etalement_incoherent",
    ]
    if records and any(vk in r for r in records for vk in _violation_keys):
        print("[...] Aggregating clinical rule violations …")
        for vk in _violation_keys:
            vals = [int(bool(r.get(vk, False))) for r in records]
            aggregated[f"{vk}_rate"] = round(float(np.mean(vals)), 4)

        n_viol = [r.get("n_violations", 0) for r in records]
        aggregated["n_violations_mean"] = round(float(np.mean(n_viol)), 4)
        aggregated["n_violations_std"]  = round(float(np.std(n_viol)),  4)

        coh_scores = [r.get("coherence_score", 1.0) for r in records]
        aggregated["coherence_score_mean"] = round(float(np.mean(coh_scores)), 4)
        aggregated["coherence_score_std"]  = round(float(np.std(coh_scores)),  4)

        any_viol = [int(r.get("n_violations", 0) > 0) for r in records]
        aggregated["any_violation_rate"] = round(float(np.mean(any_viol)), 4)

    return {
        "individual":   individual,
        "aggregated":   aggregated,
        "corpus_size":  len(texts),
    }


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Automatic evaluation of radiotherapy reports (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",       required=True,          help="Input JSONL file")
    ap.add_argument("--output",      required=True,          help="Output JSON file")
    ap.add_argument("--text_field",  default="cr_text",      help="Field containing report text")
    ap.add_argument("--detailed",    action="store_true",    help="Include per-text metrics in output")
    ap.add_argument("--use_gpu_ppl", action="store_true",
                    help="Load CamemBERT for real pseudo-perplexity (requires GPU)")
    args = ap.parse_args()

    if not args.use_gpu_ppl:
        print("[INFO] --use_gpu_ppl not set: perplexity and SLOR will be absent from results.")

    # Load records
    in_path = Path(args.input)
    records: List[Dict] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[OK] {len(records)} records loaded from {in_path}")

    texts = [r[args.text_field] for r in records if args.text_field in r]
    print(f"[OK] {len(texts)} texts to evaluate")

    # Evaluate
    results = evaluate_corpus(texts, records, use_gpu_ppl=args.use_gpu_ppl)

    # Metadata
    results["metadata"] = {
        "input_file":    str(in_path),
        "text_field":    args.text_field,
        "num_texts":     len(texts),
        "used_gpu_ppl":  args.use_gpu_ppl,
        "version":       "3.0",
    }

    if not args.detailed:
        results.pop("individual", None)

    # Convert numpy types
    def _cvt(obj):
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        if isinstance(obj, dict):            return {k: _cvt(v) for k, v in obj.items()}
        if isinstance(obj, list):            return [_cvt(v) for v in obj]
        return obj
    results = _cvt(results)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Results saved → {out_path}")

    # ── Display ───────────────────────────────────────────────────────────────
    agg = results["aggregated"]

    def _p(label, key_mean, key_std=None, fmt=".4f"):
        v = agg.get(key_mean)
        if v is None:
            return
        s = f"  {label:<32} {v:{fmt}}"
        if key_std and agg.get(key_std) is not None:
            s += f"  (±{agg[key_std]:{fmt}})"
        print(s)

    print("\n" + "=" * 65)
    print("                   METRICS SUMMARY")
    print("=" * 65)

    print("\n LEXICAL RICHNESS:")
    _p("TTR",              "ttr_mean",             "ttr_std")
    _p("MSTTR-100",        "msttr_100_mean",        "msttr_100_std")
    _p("Lexical entropy",  "lexical_entropy_mean",  "lexical_entropy_std")
    _p("Hapax ratio",      "hapax_ratio_mean",      "hapax_ratio_std")

    print("\n READABILITY:")
    _p("Flesch RE (French)","flesch_reading_ease_fr_mean","flesch_reading_ease_fr_std", ".2f")


    if "perplexity_mean" in agg:
        print("\n FLUENCY:")
        _p("Perplexity (CamemBERT)", "perplexity_mean", "perplexity_std", ".2f")
        _p("SLOR",                   "slor_mean",        "slor_std")
    else:
        print("\n FLUENCY:  [skipped — run with --use_gpu_ppl to compute]")

    print("\n DIVERSITY (Self-BLEU, lower = more diverse):")
    for n in (4, 3, 2):
        k = f"self_bleu_{n}"
        if k in agg:
            print(f"  {'Self-BLEU-' + str(n):<32} {agg[k]:.4f}")

    if "core_clinical_rate_mean" in agg:
        print("\n CONFORMITY:")
        _p("Core clinical rate", "core_clinical_rate_mean", "core_clinical_rate_std")

    if "any_violation_rate" in agg:
        print("\n CLINICAL RULE VIOLATIONS (post-generation validator):")
        _viol_labels = [
            ("v_hormono_rh_neg_rate",          "Hormono. sur RH−"),
            ("v_ct_notation_post_chir_rate",   "cT notation après chirurgie"),
            ("v_n_plus_no_ganglion_rate",       "N+ sans irrad. ganglionnaire"),
            ("v_m1_not_palliatif_rate",         "M1 sans contexte palliatif"),
            ("v_recidive_no_history_rate",      "Récidive sans antécédents"),
            ("v_etalement_incoherent_rate",     "Étalement incohérent"),
            ("any_violation_rate",              "≥ 1 violation (taux global)"),
        ]
        for key, label in _viol_labels:
            v = agg.get(key)
            if v is not None:
                flag = " ✓" if v == 0.0 else (" !" if v <= 0.1 else " ✗")
                print(f"  {label:<38} {v:.1%}{flag}")
        _p("Score cohérence (moy)", "coherence_score_mean", "coherence_score_std")
        _p("Nb violations (moy)",   "n_violations_mean",    "n_violations_std", ".2f")

    print("\n TEXT STATISTICS:")
    _p("Words / text",            "word_count_mean",          "word_count_std",          ".1f")
    _p("Characters / text",       "char_count_mean",          "char_count_std",          ".1f")
    _p("Sentences / text",        "sentence_count_mean",      "sentence_count_std",      ".1f")
    _p("Avg sentence length",     "avg_sentence_length_mean", "avg_sentence_length_std", ".2f")
    _p("Long-word ratio (>6 ch)", "long_word_ratio_mean",     "long_word_ratio_std")

    print("\n" + "=" * 65)
    print("""
  INTERPRETATION REFERENCE
  ┌──────────────────────────┬────────────┬──────────┬────────────┐
  │ Metric                   │ Excellent  │ Good     │ Poor       │
  ├──────────────────────────┼────────────┼──────────┼────────────┤
  │ TTR                      │ > 0.50     │ 0.40–0.5 │ < 0.30     │
  │ MSTTR-100                │ > 0.65     │ 0.55–0.65│ < 0.50     │
  │ Self-BLEU-4              │ < 0.30     │ 0.30–0.50│ > 0.70     │
  │ Flesch RE (medical)      │ < 40       │ 40–60    │ > 60       │
  │ Perplexity (CamemBERT)   │ < 30       │ 30–50    │ > 50       │
  │ SLOR                     │ > -5       │ -5 to -10│ < -10      │
  │ Core clinical rate       │ > 90 %     │ 70–90 %  │ < 50 %     │
  │ Any violation rate       │ 0 %        │ ≤ 10 %   │ > 20 %     │
  │ Coherence score          │ = 1.0      │ ≥ 0.80   │ < 0.60     │
  └──────────────────────────┴────────────┴──────────┴────────────┘
""")
    print("=" * 65)


if __name__ == "__main__":
    main()