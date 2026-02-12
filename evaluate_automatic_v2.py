#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AUTOMATIC EVALUATION OF RADIOTHERAPY REPORTS
IMPROVED VERSION 2.0
=============================================================================

This script computes the following metrics:
- TTR (Type-Token Ratio): lexical richness
- MSTTR (Mean Segmental TTR): robust lexical richness
- Lexical entropy: vocabulary diversity
- Perplexity (PPL): linguistic fluency (with CamemBERT model)
- SLOR (Syntactic Log-Odds Ratio): normalized syntactic fluency
- Self-BLEU: inter-text diversity
- FRE (Flesch Reading Ease) adapted for French
- Medical-specific metrics (terminology, structure, doses)

IMPROVEMENTS V2:
- Batched perplexity computation (10-30x faster)
- Medical-specific specialized metrics
- Improved French syllable counting
- Detailed export with medical metrics

Usage:
    python evaluate_automatic_v2_en.py --input results.jsonl --output metrics.json

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
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Flag to check if transformers is available
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, CamembertForMaskedLM, CamembertTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("[WARN] transformers/torch not available. Perplexity will be estimated.")


# =============================================================================
# 1. TOKENIZATION
# =============================================================================

def tokenize_french(text: str) -> List[str]:
    """
    Simple tokenization for French.
    Preserves words with accents and French special characters.
    """
    text = text.lower()
    # Regular expression for French words (with accents)
    tokens = re.findall(r"\b[a-zàâäéèêëïîôùûüÿçœæ]+\b", text)
    return tokens


def count_syllables_french(word: str) -> int:
    """
    Approximate syllable count for French.
    Improved version with French rules.
    """
    word = word.lower()
    vowels = "aeiouyàâäéèêëïîôùûüÿœæ"
    count = 0
    prev_is_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    
    # Improved French rules
    # 1. Silent final 'e'
    if word.endswith('e') and count > 1:
        count -= 1
    
    # 2. Silent final '-es', '-ent'
    if word.endswith(('es', 'ent')) and count > 1:
        count -= 1
    
    return max(1, count)


# =============================================================================
# 2. LEXICAL RICHNESS METRICS
# =============================================================================

def compute_ttr(text: str) -> float:
    """
    Type-Token Ratio (TTR): measure of lexical richness.
    
    TTR = number of unique words / total number of words
    
    Interpretation:
    - > 0.5: Excellent (very rich vocabulary)
    - 0.4-0.5: Good
    - 0.3-0.4: Acceptable
    - < 0.3: Poor (repetitive vocabulary)
    """
    tokens = tokenize_french(text)
    if len(tokens) == 0:
        return 0.0
    types = set(tokens)
    return len(types) / len(tokens)


def compute_msttr(text: str, window_size: int = 100) -> float:
    """
    Mean Segmental TTR (MSTTR): TTR averaged over windows.
    
    More robust than simple TTR for long texts as simple TTR
    naturally decreases with text length.
    
    Args:
        text: Text to analyze
        window_size: Window size (default: 100 words)
    """
    tokens = tokenize_french(text)
    if len(tokens) < window_size:
        return compute_ttr(text)
    
    ttrs = []
    for i in range(0, len(tokens) - window_size + 1, window_size):
        window = tokens[i:i + window_size]
        ttrs.append(len(set(window)) / len(window))
    
    return np.mean(ttrs) if ttrs else 0.0


def compute_lexical_entropy(text: str) -> float:
    """
    Lexical entropy: measure of vocabulary unpredictability.
    
    H = -Σ p(w) * log2(p(w))
    
    Interpretation:
    - Higher = more varied/unpredictable vocabulary
    - Typically between 5 and 10 for well-written texts
    """
    tokens = tokenize_french(text)
    if len(tokens) == 0:
        return 0.0
    
    freq = Counter(tokens)
    total = len(tokens)
    
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy


def compute_hapax_ratio(text: str) -> float:
    """
    Ratio of hapax legomena (words appearing only once).
    
    Additional indicator of lexical richness.
    A high ratio indicates diversified vocabulary.
    """
    tokens = tokenize_french(text)
    if len(tokens) == 0:
        return 0.0
    
    freq = Counter(tokens)
    hapax = sum(1 for count in freq.values() if count == 1)
    return hapax / len(tokens)


# =============================================================================
# 3. READABILITY SCORE (Flesch Reading Ease adapted for French)
# =============================================================================

def compute_flesch_reading_ease_fr(text: str) -> float:
    """
    Flesch readability score adapted for French.
    
    Adapted formula: 207 - 1.015 * (words/sentences) - 73.6 * (syllables/words)
    
    Interpretation:
    - 90-100: Very easy
    - 60-90: Easy
    - 30-60: Difficult
    - 0-30: Very difficult (technical/academic text)
    
    Note: Medical texts are generally in the 20-40 range.
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        return 0.0
    
    # Tokenize
    tokens = tokenize_french(text)
    if len(tokens) == 0:
        return 0.0
    
    # Count syllables
    total_syllables = sum(count_syllables_french(w) for w in tokens)
    
    # Calculate ratios
    words_per_sentence = len(tokens) / len(sentences)
    syllables_per_word = total_syllables / len(tokens)
    
    # Flesch formula adapted for French
    fre = 207 - (1.015 * words_per_sentence) - (73.6 * syllables_per_word)
    
    return max(0, min(100, fre))  # Bound between 0 and 100


# =============================================================================
# 4. SELF-BLEU (Inter-text diversity)
# =============================================================================

def get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Extract n-grams from a token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_bleu_single(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:
    """
    Simplified BLEU calculation between two texts.
    """
    if len(hypothesis) == 0:
        return 0.0
    
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(get_ngrams(reference, n))
        hyp_ngrams = Counter(get_ngrams(hypothesis, n))
        
        if len(hyp_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        matches = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = sum(hyp_ngrams.values())
        precisions.append(matches / total if total > 0 else 0.0)
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        bleu = math.exp(log_avg)
    else:
        bleu = 0.0
    
    # Brevity penalty
    if len(hypothesis) > 0:
        bp = min(1.0, math.exp(1 - len(reference) / len(hypothesis)))
    else:
        bp = 0.0
    
    return bleu * bp


def compute_self_bleu(texts: List[str], max_n: int = 4) -> float:
    """
    Self-BLEU: measure of diversity between generated texts.
    
    For each text, calculates BLEU with all other texts.
    
    Interpretation:
    - < 0.3: Excellent diversity
    - 0.3-0.5: Good diversity
    - 0.5-0.7: Moderate diversity
    - > 0.7: Low diversity (very similar texts)
    
    Note: For standardized medical documents, a score of 0.4-0.5
    is acceptable as they must follow a common structure.
    """
    if len(texts) < 2:
        return 0.0
    
    tokenized = [tokenize_french(t) for t in texts]
    bleu_scores = []
    
    for i, hyp in enumerate(tokenized):
        # References = all other texts
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        
        # Average BLEU with each reference
        scores = [compute_bleu_single(ref, hyp, max_n) for ref in refs]
        if scores:
            bleu_scores.append(np.mean(scores))
    
    return np.mean(bleu_scores) if bleu_scores else 0.0


# =============================================================================
# 5. PERPLEXITY (PPL) AND SLOR - OPTIMIZED VERSION
# =============================================================================

class PerplexityCalculator:
    """
    Perplexity calculator using CamemBERT (French model).
    OPTIMIZED VERSION with batching.
    """
    
    def __init__(self, model_name: str = "camembert-base"):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.device = None
        
    def load_model(self):
        """Load CamemBERT model."""
        if not TRANSFORMERS_AVAILABLE:
            print("[WARN] transformers not available, perplexity will be estimated.")
            return False
        
        try:
            print(f"[...] Loading model {self.model_name}...")
            self.tokenizer = CamembertTokenizer.from_pretrained(self.model_name)
            self.model = CamembertForMaskedLM.from_pretrained(self.model_name)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[OK] Model loaded on {self.device}")
            return True
        except Exception as e:
            print(f"[WARN] Cannot load {self.model_name}: {e}")
            return False
    
    def compute_pseudo_perplexity(self, text: str, batch_size: int = 32) -> float:
        """
        Compute pseudo-perplexity for MLM (masked language model).
        OPTIMIZED VERSION with batching - 10-30x faster!
        
        For CamemBERT (MLM), we use the "pseudo-log-likelihood" technique
        where each token is masked in turn and we measure the probability
        of predicting it correctly.
        """
        if self.model is None:
            return self._estimate_perplexity(text)
        
        try:
            encodings = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            input_ids = encodings.input_ids.to(self.device)
            seq_len = input_ids.size(1)
            
            if seq_len <= 2:  # Just [CLS] and [SEP]
                return float('inf')
            
            nlls = []
            
            # Batch positions to mask (OPTIMIZATION)
            for start in range(1, seq_len - 1, batch_size):
                end = min(start + batch_size, seq_len - 1)
                batch_size_actual = end - start
                
                # Create batch of masked inputs
                masked_batch = input_ids.repeat(batch_size_actual, 1)
                targets = []
                
                for i, pos in enumerate(range(start, end)):
                    masked_batch[i, pos] = self.tokenizer.mask_token_id
                    targets.append(input_ids[0, pos].item())
                
                # Single forward pass on entire batch
                with torch.no_grad():
                    outputs = self.model(masked_batch)
                    logits = outputs.logits
                
                # Extract NLL for each position
                for i, pos in enumerate(range(start, end)):
                    log_probs = torch.log_softmax(logits[i, pos], dim=-1)
                    nll = -log_probs[targets[i]].item()
                    nlls.append(nll)
            
            # Perplexity = exp(mean of NLLs)
            ppl = math.exp(np.mean(nlls)) if nlls else float('inf')
            return ppl
            
        except Exception as e:
            print(f"[WARN] Error computing PPL: {e}")
            return self._estimate_perplexity(text)
    
    def _estimate_perplexity(self, text: str) -> float:
        """
        Perplexity estimation based on token entropy.
        Used when model is not available.
        """
        tokens = tokenize_french(text)
        if len(tokens) == 0:
            return float('inf')
        
        freq = Counter(tokens)
        total = len(tokens)
        entropy = -sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0)
        
        # Approximate conversion entropy -> perplexity
        return 2 ** entropy


def compute_slor(text: str, ppl: float, unigram_ppl: float = 50.0) -> float:
    """
    SLOR (Syntactic Log-Odds Ratio).
    
    Measures fluency by normalizing by length and unigram frequency.
    
    SLOR = (log P(S) - log P_unigram(S)) / |S|
    
    Interpretation:
    - Higher = better syntactic fluency
    - Typical values: -2 to 0 for well-formed texts
    """
    tokens = tokenize_french(text)
    length = len(tokens)
    
    if length == 0 or ppl <= 0 or ppl == float('inf'):
        return 0.0
    
    log_prob = -math.log(ppl) * length  # Approximation
    log_unigram = -math.log(unigram_ppl) * length
    
    slor = (log_prob - log_unigram) / length
    return slor


# =============================================================================
# 6. COMPLEMENTARY FLUENCY METRICS
# =============================================================================

def compute_fluency_metrics(text: str) -> Dict[str, float]:
    """
    Calculate additional fluency metrics.
    """
    tokens = tokenize_french(text)
    
    # Average sentence length
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sent_len = np.mean([len(tokenize_french(s)) for s in sentences]) if sentences else 0
    
    # Ratio of long words (>6 characters) - complexity indicator
    long_words = [t for t in tokens if len(t) > 6]
    long_word_ratio = len(long_words) / len(tokens) if tokens else 0
    
    # Logical connectors (narrative structure indicator)
    connectors = [
        "donc", "ainsi", "cependant", "toutefois", "néanmoins",
        "puis", "ensuite", "enfin", "également", "notamment",
        "ailleurs", "effet", "plus", "outre", "conséquent"
    ]
    connector_count = sum(1 for t in tokens if t in connectors)
    
    return {
        "avg_sentence_length": round(avg_sent_len, 2),
        "long_word_ratio": round(long_word_ratio, 4),
        "connector_count": connector_count,
        "total_tokens": len(tokens),
        "total_sentences": len(sentences),
    }


# =============================================================================
# 7. MEDICAL-SPECIFIC METRICS (NEW)
# =============================================================================

def compute_medical_terminology_density(text: str) -> Dict[str, float]:
    """
    Measure medical terminology density.
    
    Returns:
    - ratio_medical_terms: proportion of medical terms
    - unique_medical_terms: number of unique medical terms
    - medical_diversity: diversity of medical vocabulary
    """
    text_lower = text.lower()
    
    # Medical terms radiotherapy/oncology
    medical_terms = [
        # Anatomy
        "sein", "breast", "thorax", "ganglion", "node", "axillaire", "axillary", "chaîne", "chain", "mammaire", "mammary",
        "sternum", "clavicule", "clavicle", "sus-claviculaire", "supraclavicular", "pariétal", "parietal",
        
        # Pathology
        "carcinome", "carcinoma", "adénocarcinome", "adenocarcinoma", "tumeur", "tumor", "néoplasie", "neoplasia", "lésion", "lesion",
        "infiltrant", "infiltrating", "invasif", "invasive", "métastase", "metastasis", "récidive", "recurrence", "rechute",
        
        # Biology
        "récepteur", "receptor", "hormonal", "her2", "ki67", "triple",
        "positif", "positive", "négatif", "negative", "luminal", "basal",
        
        # Stages
        "tnm", "stade", "stage", "grade", "classification",
        
        # Treatments
        "radiothérapie", "radiotherapy", "chimiothérapie", "chemotherapy", "hormonothérapie",
        "chirurgie", "surgery", "mastectomie", "mastectomy", "tumorectomie", "lumpectomy", "curage",
        "adjuvant", "néoadjuvant", "neoadjuvant", "palliatif", "palliative",
        
        # RT techniques
        "photon", "electron", "irradiation", "dose", "fraction",
        "séance", "session", "gray", "gy", "centigray", "cgy",
        "conformationnelle", "conformal", "imrt", "arcthérapie", "vmat",
        "volume", "cible", "target", "ptv", "ctv", "gtv",
        "organe", "organ", "risque", "risk", "oar", "contrainte", "constraint",
        
        # Tolerance
        "toxicité", "toxicity", "effet", "effect", "secondaire", "secondary", "tolérance", "tolerance",
        "érythème", "erythema", "épidermite", "asthénie", "fatigue",
        
        # Follow-up
        "surveillance", "follow-up", "consultation", "contrôle", "control", "imagerie", "imaging",
        "mammographie", "mammography", "échographie", "ultrasound", "irm", "mri", "scanner", "ct",
    ]
    
    # Tokenize
    tokens = re.findall(r"\b[a-zàâäéèêëïîôùûüÿçœæ]+\b", text_lower)
    
    # Count medical terms
    medical_found = [t for t in tokens if any(mt in t for mt in medical_terms)]
    unique_medical = set(medical_found)
    
    ratio = len(medical_found) / len(tokens) if tokens else 0
    diversity = len(unique_medical) / len(medical_found) if medical_found else 0
    
    return {
        "medical_term_ratio": round(ratio, 4),
        "unique_medical_terms": len(unique_medical),
        "medical_diversity": round(diversity, 4),
    }


def compute_structure_completeness(text: str) -> Dict[str, any]:
    """
    Evaluate presence of expected sections in RT report.
    
    Expected sections:
    1. Patient identity
    2. Indication/Diagnosis
    3. Technical modalities
    4. Course of treatment
    5. Tolerance
    6. Conclusion
    """
    text_lower = text.lower()
    
    # Section markers
    section_markers = {
        "identity": ["patient", "madame", "monsieur", "né", "born", "date de naissance", "date of birth"],
        "indication": ["indication", "diagnostic", "diagnosis", "carcinome", "carcinoma", "cancer", "tumeur", "tumor"],
        "technique": [
            "technique", "modalité", "modality", "dose", "fraction", "séance", "session",
            "photon", "electron", "volume", "cible", "target"
        ],
        "course": [
            "déroulement", "course", "réalisation", "delivery", "réalisé", "performed", "effectué",
            "séance", "session", "fraction", "traitement", "treatment"
        ],
        "tolerance": [
            "tolérance", "tolerance", "toxicité", "toxicity", "effet", "effect", "secondaire", "side",
            "érythème", "erythema", "épidermite", "asthénie", "fatigue"
        ],
        "conclusion": [
            "conclusion", "synthèse", "summary", "bilan", "surveillance", "follow-up",
            "suivi", "contrôle", "control"
        ],
    }
    
    # Check presence of each section
    section_scores = {}
    for section, markers in section_markers.items():
        score = any(marker in text_lower for marker in markers)
        section_scores[f"has_{section}"] = bool(score)
    
    # Overall structural completeness score
    completeness = sum(section_scores.values()) / len(section_scores)
    section_scores["structure_completeness"] = round(completeness, 4)
    
    return section_scores


def compute_dose_mention_accuracy(text: str) -> Dict[str, any]:
    """
    Check presence and consistency of dose mentions.
    
    Searches for:
    - Total doses (e.g., 50 Gy, 45 gray)
    - Dose per fraction (e.g., 2 Gy/fr, 1.8 gray per session)
    - Number of fractions (e.g., 25 fractions, 30 sessions)
    """
    # Patterns for doses
    pattern_dose_total = r'(\d+(?:[.,]\d+)?)\s*(?:gy|gray|centigray|cgy)'
    pattern_dose_fraction = r'(\d+(?:[.,]\d+)?)\s*(?:gy|gray)(?:/fr|/fraction|par fraction|per fraction|par séance|per session)?'
    pattern_nb_fractions = r'(\d+)\s*(?:fraction|fractions|séance|séances|session|sessions)'
    
    text_lower = text.lower()
    
    # Extraction
    doses_total = re.findall(pattern_dose_total, text_lower, re.IGNORECASE)
    doses_fraction = re.findall(pattern_dose_fraction, text_lower, re.IGNORECASE)
    nb_fractions = re.findall(pattern_nb_fractions, text_lower, re.IGNORECASE)
    
    # Convert to floats
    doses_total_float = [float(d.replace(',', '.')) for d in doses_total if d]
    nb_fractions_int = [int(n) for n in nb_fractions if n]
    
    # Consistency check (total dose ≈ dose/fraction × nb fractions)
    coherent = False
    if doses_total_float and doses_fraction and nb_fractions_int:
        total = doses_total_float[0]
        per_frac = float(doses_fraction[0].replace(',', '.'))
        n_frac = nb_fractions_int[0]
        
        calculated_total = per_frac * n_frac
        # 10% tolerance
        if abs(calculated_total - total) / total < 0.10:
            coherent = True
    
    return {
        "has_dose_total": bool(doses_total),
        "has_dose_per_fraction": bool(doses_fraction),
        "has_nb_fractions": bool(nb_fractions),
        "dose_coherence": coherent,
        "dose_total_value": doses_total_float[0] if doses_total_float else None,
        "nb_fractions_value": nb_fractions_int[0] if nb_fractions_int else None,
    }


def evaluate_medical_quality(text: str) -> Dict[str, any]:
    """
    Complete evaluation of medical quality of a report.
    
    Combines all medical-specific metrics.
    """
    results = {}
    
    # Medical terminology
    results.update(compute_medical_terminology_density(text))
    
    # Structure
    results.update(compute_structure_completeness(text))
    
    # Doses
    results.update(compute_dose_mention_accuracy(text))
    
    # Overall medical quality score
    quality_components = [
        results.get("medical_term_ratio", 0) > 0.15,  # At least 15% medical terms
        results.get("structure_completeness", 0) > 0.67,  # At least 4/6 sections
        results.get("has_dose_total", False),  # Total dose mentioned
        results.get("has_nb_fractions", False),  # Number of fractions mentioned
    ]
    
    medical_quality_score = sum(quality_components) / len(quality_components)
    results["medical_quality_score"] = round(medical_quality_score, 4)
    
    return results


# =============================================================================
# 8. COMPLETE EVALUATION
# =============================================================================

def evaluate_single_text(
    text: str, 
    ppl_calculator: Optional[PerplexityCalculator] = None
) -> Dict[str, Any]:
    """
    Evaluate a single text with all metrics.
    IMPROVED VERSION with medical metrics.
    """
    results = {}
    
    # Lexical richness
    results["ttr"] = round(compute_ttr(text), 4)
    results["msttr_100"] = round(compute_msttr(text, 100), 4)
    results["lexical_entropy"] = round(compute_lexical_entropy(text), 4)
    results["hapax_ratio"] = round(compute_hapax_ratio(text), 4)
    
    # Readability
    results["flesch_reading_ease_fr"] = round(compute_flesch_reading_ease_fr(text), 2)
    
    # Perplexity
    if ppl_calculator and ppl_calculator.model is not None:
        results["perplexity"] = round(ppl_calculator.compute_pseudo_perplexity(text), 2)
    else:
        results["perplexity_estimate"] = round(
            PerplexityCalculator()._estimate_perplexity(text), 2
        )
    
    # SLOR
    ppl_value = results.get("perplexity", results.get("perplexity_estimate", 100))
    results["slor"] = round(compute_slor(text, ppl_value), 4)
    
    # Fluency
    fluency = compute_fluency_metrics(text)
    results.update({f"fluency_{k}": v for k, v in fluency.items()})
    
    # NEW: Medical metrics
    medical = evaluate_medical_quality(text)
    results.update({f"medical_{k}": v for k, v in medical.items()})
    
    # Length
    results["char_count"] = len(text)
    results["word_count"] = len(tokenize_french(text))
    
    return results


def evaluate_corpus(
    texts: List[str], 
    records: List[Dict] = None,
    use_gpu_ppl: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a complete corpus of texts.
    
    Args:
        texts: List of texts to evaluate
        records: Original records (for mention_rate)
        use_gpu_ppl: Use CamemBERT for perplexity (slower but accurate)
    """
    # Initialize perplexity calculator if requested
    ppl_calculator = None
    if use_gpu_ppl and TRANSFORMERS_AVAILABLE:
        ppl_calculator = PerplexityCalculator()
        ppl_calculator.load_model()
    
    # Individual metrics
    print("[...] Computing individual metrics...")
    individual_metrics = []
    for i, text in enumerate(texts):
        metrics = evaluate_single_text(text, ppl_calculator)
        individual_metrics.append(metrics)
        if (i + 1) % 10 == 0:
            print(f"      {i+1}/{len(texts)} texts processed")
    
    # Aggregation
    print("[...] Aggregating metrics...")
    aggregated = {}
    for key in individual_metrics[0].keys():
        # Skip None values and non-numeric values
        values = [m[key] for m in individual_metrics if m[key] is not None and isinstance(m[key], (int, float, bool))]
        if values:
            # Convert bool to int for calculations
            if isinstance(values[0], bool):
                values = [int(v) for v in values]
            
            aggregated[f"{key}_mean"] = round(float(np.mean(values)), 4)
            aggregated[f"{key}_std"] = round(float(np.std(values)), 4)
            aggregated[f"{key}_min"] = round(float(np.min(values)), 4)
            aggregated[f"{key}_max"] = round(float(np.max(values)), 4)
    
    # Self-BLEU (corpus diversity)
    print("[...] Computing Self-BLEU...")
    aggregated["self_bleu_4"] = round(float(compute_self_bleu(texts, max_n=4)), 4)
    aggregated["self_bleu_3"] = round(float(compute_self_bleu(texts, max_n=3)), 4)
    aggregated["self_bleu_2"] = round(float(compute_self_bleu(texts, max_n=2)), 4)
    
    # Mention rate if available
    if records:
        mention_rates = [r.get("mention_rate", r.get("core_clinical_rate", 0)) for r in records]
        aggregated["mention_rate_mean"] = round(float(np.mean(mention_rates)), 4)
        aggregated["mention_rate_std"] = round(float(np.std(mention_rates)), 4)
    
    return {
        "individual": individual_metrics,
        "aggregated": aggregated,
        "corpus_size": len(texts),
    }


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Automatic evaluation of radiotherapy reports - Improved Version 2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python evaluate_automatic_v2_en.py --input results.jsonl --output metrics.json
  python evaluate_automatic_v2_en.py --input results.jsonl --output metrics.json --use_gpu_ppl
  python evaluate_automatic_v2_en.py --input results.jsonl --output metrics.json --detailed
        """
    )
    ap.add_argument("--input", required=True, help="JSONL file of generations")
    ap.add_argument("--output", required=True, help="JSON file for results")
    ap.add_argument("--text_field", default="cr_text", help="Field containing text")
    ap.add_argument("--detailed", action="store_true", help="Include individual metrics")
    ap.add_argument("--use_gpu_ppl", action="store_true", 
                    help="Use CamemBERT for perplexity (requires GPU)")
    
    args = ap.parse_args()
    
    # Load data
    input_path = Path(args.input)
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"[OK] Loaded {len(records)} records from {input_path}")
    
    # Extract texts
    texts = [r[args.text_field] for r in records if args.text_field in r]
    print(f"[OK] {len(texts)} texts to evaluate")
    
    # Evaluation
    results = evaluate_corpus(texts, records, use_gpu_ppl=args.use_gpu_ppl)
    
    # Add metadata
    results["metadata"] = {
        "input_file": str(input_path),
        "text_field": args.text_field,
        "num_texts": len(texts),
        "used_gpu_ppl": args.use_gpu_ppl,
        "version": "2.0",
    }
    
    # Remove individual metrics if not requested
    if not args.detailed:
        del results["individual"]
    
    # Convert numpy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    results = convert_numpy(results)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Results saved to {output_path}")
    
    # Display summary
    print("\n" + "=" * 70)
    print("                    METRICS SUMMARY")
    print("=" * 70)
    
    agg = results["aggregated"]
    
    print(f"\n LEXICAL RICHNESS:")
    print(f"   Mean TTR:               {agg['ttr_mean']:.4f} (±{agg['ttr_std']:.4f})")
    print(f"   Mean MSTTR-100:         {agg['msttr_100_mean']:.4f} (±{agg['msttr_100_std']:.4f})")
    print(f"   Lexical entropy:        {agg['lexical_entropy_mean']:.4f} (±{agg['lexical_entropy_std']:.4f})")
    print(f"   Hapax ratio:            {agg['hapax_ratio_mean']:.4f} (±{agg['hapax_ratio_std']:.4f})")
    
    print(f"\n READABILITY:")
    print(f"   Flesch RE (French):     {agg['flesch_reading_ease_fr_mean']:.2f} (±{agg['flesch_reading_ease_fr_std']:.2f})")
    
    print(f"\n FLUENCY:")
    ppl_key = "perplexity_mean" if "perplexity_mean" in agg else "perplexity_estimate_mean"
    if ppl_key in agg:
        print(f"   Perplexity:             {agg[ppl_key]:.2f} (±{agg[ppl_key.replace('_mean', '_std')]:.2f})")
    print(f"   SLOR:                   {agg['slor_mean']:.4f} (±{agg['slor_std']:.4f})")
    
    print(f"\n DIVERSITY (Self-BLEU, lower = more diverse):")
    print(f"   Self-BLEU-2:            {agg['self_bleu_2']:.4f}")
    print(f"   Self-BLEU-3:            {agg['self_bleu_3']:.4f}")
    print(f"   Self-BLEU-4:            {agg['self_bleu_4']:.4f}")
    
    print(f"\n MEDICAL QUALITY (NEW):")
    print(f"   Medical term density:   {agg['medical_medical_term_ratio_mean']:.2%} (±{agg['medical_medical_term_ratio_std']:.2%})")
    print(f"   Structure completeness: {agg['medical_structure_completeness_mean']:.2%} (±{agg['medical_structure_completeness_std']:.2%})")
    print(f"   Total dose mention:     {agg['medical_has_dose_total_mean']:.2%}")
    print(f"   Dose coherence:         {agg['medical_dose_coherence_mean']:.2%}")
    print(f"   Overall quality score:  {agg['medical_medical_quality_score_mean']:.2%} (±{agg['medical_medical_quality_score_std']:.2%})")
    
    if "mention_rate_mean" in agg:
        print(f"\n CONFORMITY:")
        print(f"   Mention rate:           {agg['mention_rate_mean']:.2%} (±{agg['mention_rate_std']:.2%})")
    
    print(f"\n LENGTH:")
    print(f"   Words/text:             {agg['word_count_mean']:.0f} (±{agg['word_count_std']:.0f})")
    print(f"   Characters/text:        {agg['char_count_mean']:.0f} (±{agg['char_count_std']:.0f})")
    print(f"   Sentences/text:         {agg['fluency_total_sentences_mean']:.1f} (±{agg['fluency_total_sentences_std']:.1f})")
    
    print("\n" + "=" * 70)
    print("                    INTERPRETATION GRID")
    print("=" * 70)
    print("""
    | Metric                 | Excellent | Good      | Acceptable | Poor      |
    |------------------------|-----------|-----------|------------|-----------|
    | TTR                    | > 0.5     | 0.4-0.5   | 0.3-0.4    | < 0.3     |
    | Self-BLEU              | < 0.3     | 0.3-0.5   | 0.5-0.7    | > 0.7     |
    | Mention Rate           | > 90%     | 70-90%    | 50-70%     | < 50%     |
    | Flesch RE              | > 60      | 40-60     | 20-40      | < 20      |
    | Medical Density        | > 20%     | 15-20%    | 10-15%     | < 10%     |
    | Structure Complete     | > 85%     | 70-85%    | 50-70%     | < 50%     |
    | Medical Quality        | > 85%     | 70-85%    | 50-70%     | < 50%     |
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()