"""
======================================================================================
AUTOMATIC EVALUATION OF RADIOTHERAPY REPORTS
===================================================================================================

This script calculates the following metrics:

- TTR (Type-Token Ratio): lexical richness
- MSTTR (Mean Segmental TTR): robust lexical richness
- Lexical entropy: vocabulary diversity
- Perplexity (PPL): Linguistic fluency (using the CamemBERT model)

- SLOR (Syntactic Log-Odds Ratio): normalized syntactic fluency
- Self-BLEU: inter-text diversity
- FRE (Flesch Reading Ease) adapted for French

Usage:
python evaluate_automatic.py --input results.jsonl --output metrics.json

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

# Flag to check if transformer is available 
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, CamembertForMaskedLM, CamembertTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("[WARN] transformers/torch non available. Perplexité calculated by estimation.")


# =============================================================================
# 1. TOKENISATION
# =============================================================================

def tokenize_french(text: str) -> List[str]:
    """
    Tokenisation simple pour le français.
    Conserve les mots avec accents et caractères spéciaux français.
    """
    text = text.lower()
    # Expression régulière pour les mots français (avec accents)
    tokens = re.findall(r"\b[a-zàâäéèêëïîôùûüÿçœæ]+\b", text)
    return tokens


def count_syllables_french(word: str) -> int:
    """
    Compte approximatif des syllabes en français.
    Basé sur les voyelles avec quelques règles de base.
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
    
    # Règle du 'e' muet final
    if word.endswith('e') and count > 1:
        count -= 1
    
    return max(1, count)


# =============================================================================
# 2. MÉTRIQUES DE RICHESSE LEXICALE
# =============================================================================

def compute_ttr(text: str) -> float:
    """
    Type-Token Ratio (TTR) : mesure de richesse lexicale.
    
    TTR = nombre de mots uniques / nombre total de mots
    
    Interprétation:
    - > 0.5 : Excellent (vocabulaire très riche)
    - 0.4-0.5 : Bon
    - 0.3-0.4 : Acceptable
    - < 0.3 : Faible (vocabulaire répétitif)
    """
    tokens = tokenize_french(text)
    if len(tokens) == 0:
        return 0.0
    types = set(tokens)
    return len(types) / len(tokens)


def compute_msttr(text: str, window_size: int = 100) -> float:
    """
    Mean Segmental TTR (MSTTR) : TTR moyenné sur des fenêtres.
    
    Plus robuste que TTR pour les textes longs car le TTR simple
    diminue naturellement avec la longueur du texte.
    
    Args:
        text: Texte à analyser
        window_size: Taille de la fenêtre (défaut: 100 mots)
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
    Entropie lexicale : mesure de l'imprévisibilité du vocabulaire.
    
    H = -Σ p(w) * log2(p(w))
    
    Interprétation:
    - Plus élevé = vocabulaire plus varié/imprévisible
    - Typiquement entre 5 et 10 pour des textes bien écrits
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
    Ratio de hapax legomena (mots n'apparaissant qu'une fois).
    
    Indicateur supplémentaire de richesse lexicale.
    Un ratio élevé indique un vocabulaire diversifié.
    """
    tokens = tokenize_french(text)
    if len(tokens) == 0:
        return 0.0
    
    freq = Counter(tokens)
    hapax = sum(1 for count in freq.values() if count == 1)
    return hapax / len(tokens)


# =============================================================================
# 3. SCORE DE LISIBILITÉ (Flesch Reading Ease adapté français)
# =============================================================================

def compute_flesch_reading_ease_fr(text: str) -> float:
    """
    Score de lisibilité Flesch adapté au français.
    
    Formule adaptée : 207 - 1.015 * (mots/phrases) - 73.6 * (syllabes/mots)
    
    Interprétation:
    - 90-100 : Très facile
    - 60-90 : Facile
    - 30-60 : Difficile
    - 0-30 : Très difficile (texte technique/académique)
    
    Note: Les textes médicaux sont généralement dans la zone 20-40.
    """
    # Découper en phrases
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        return 0.0
    
    # Tokeniser
    tokens = tokenize_french(text)
    if len(tokens) == 0:
        return 0.0
    
    # Compter les syllabes
    total_syllables = sum(count_syllables_french(w) for w in tokens)
    
    # Calculer les ratios
    words_per_sentence = len(tokens) / len(sentences)
    syllables_per_word = total_syllables / len(tokens)
    
    # Formule Flesch adaptée au français
    fre = 207 - (1.015 * words_per_sentence) - (73.6 * syllables_per_word)
    
    return max(0, min(100, fre))  # Borner entre 0 et 100


# =============================================================================
# 4. SELF-BLEU (Diversité inter-textes)
# =============================================================================

def get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Extrait les n-grammes d'une liste de tokens."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_bleu_single(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:
    """
    Calcul simplifié du BLEU entre deux textes.
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
    
    # Moyenne géométrique des précisions
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
    Self-BLEU : mesure de diversité entre les textes générés.
    
    Pour chaque texte, calcule le BLEU avec tous les autres textes.
    
    Interprétation:
    - < 0.3 : Excellente diversité
    - 0.3-0.5 : Bonne diversité
    - 0.5-0.7 : Diversité modérée
    - > 0.7 : Faible diversité (textes très similaires)
    
    Note: Pour des documents médicaux standardisés, un score de 0.4-0.5
    est acceptable car ils doivent suivre une structure commune.
    """
    if len(texts) < 2:
        return 0.0
    
    tokenized = [tokenize_french(t) for t in texts]
    bleu_scores = []
    
    for i, hyp in enumerate(tokenized):
        # Références = tous les autres textes
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        
        # BLEU moyen avec chaque référence
        scores = [compute_bleu_single(ref, hyp, max_n) for ref in refs]
        if scores:
            bleu_scores.append(np.mean(scores))
    
    return np.mean(bleu_scores) if bleu_scores else 0.0


# =============================================================================
# 5. PERPLEXITÉ (PPL) ET SLOR
# =============================================================================

class PerplexityCalculator:
    """
    Calculateur de perplexité utilisant CamemBERT (modèle français).
    """
    
    def __init__(self, model_name: str = "camembert-base"):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.device = None
        
    def load_model(self):
        """Charge le modèle CamemBERT."""
        if not TRANSFORMERS_AVAILABLE:
            print("[WARN] transformers non disponible, perplexité estimée.")
            return False
        
        try:
            print(f"[...] Chargement du modèle {self.model_name}...")
            self.tokenizer = CamembertTokenizer.from_pretrained(self.model_name)
            self.model = CamembertForMaskedLM.from_pretrained(self.model_name)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[OK] Modèle chargé sur {self.device}")
            return True
        except Exception as e:
            print(f"[WARN] Impossible de charger {self.model_name}: {e}")
            return False
    
    def compute_pseudo_perplexity(self, text: str, stride: int = 256) -> float:
        """
        Calcule la pseudo-perplexité pour un modèle MLM (masked language model).
        
        Pour CamemBERT (MLM), on utilise la technique du "pseudo-log-likelihood"
        où chaque token est masqué tour à tour et on mesure la probabilité
        de le prédire correctement.
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
            
            nlls = []
            seq_len = input_ids.size(1)
            
            # Masquer chaque token et calculer la perte
            for i in range(1, seq_len - 1):  # Ignorer [CLS] et [SEP]
                masked_input = input_ids.clone()
                masked_input[0, i] = self.tokenizer.mask_token_id
                
                with torch.no_grad():
                    outputs = self.model(masked_input)
                    logits = outputs.logits
                
                # Log-probabilité du token correct
                log_probs = torch.log_softmax(logits[0, i], dim=-1)
                nll = -log_probs[input_ids[0, i]].item()
                nlls.append(nll)
            
            # Perplexité = exp(moyenne des NLL)
            ppl = math.exp(np.mean(nlls)) if nlls else float('inf')
            return ppl
            
        except Exception as e:
            print(f"[WARN] Erreur calcul PPL: {e}")
            return self._estimate_perplexity(text)
    
    def _estimate_perplexity(self, text: str) -> float:
        """
        Estimation de la perplexité basée sur l'entropie des tokens.
        Utilisée quand le modèle n'est pas disponible.
        """
        tokens = tokenize_french(text)
        if len(tokens) == 0:
            return float('inf')
        
        freq = Counter(tokens)
        total = len(tokens)
        entropy = -sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0)
        
        # Conversion approximative entropie -> perplexité
        return 2 ** entropy


def compute_slor(text: str, ppl: float, unigram_ppl: float = 50.0) -> float:
    """
    SLOR (Syntactic Log-Odds Ratio).
    
    Mesure la fluence en normalisant par la longueur et la fréquence unigramme.
    
    SLOR = (log P(S) - log P_unigram(S)) / |S|
    
    Interprétation:
    - Plus élevé = meilleure fluence syntaxique
    - Valeurs typiques : -2 à 0 pour des textes bien formés
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
# 6. MÉTRIQUES DE FLUENCE COMPLÉMENTAIRES
# =============================================================================

def compute_fluency_metrics(text: str) -> Dict[str, float]:
    """
    Calcule des métriques de fluence supplémentaires.
    """
    tokens = tokenize_french(text)
    
    # Longueur de phrase moyenne
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sent_len = np.mean([len(tokenize_french(s)) for s in sentences]) if sentences else 0
    
    # Ratio de mots longs (>6 caractères) - indicateur de complexité
    long_words = [t for t in tokens if len(t) > 6]
    long_word_ratio = len(long_words) / len(tokens) if tokens else 0
    
    # Connecteurs logiques (indicateur de structure narrative)
    connecteurs = [
        "donc", "ainsi", "cependant", "toutefois", "néanmoins",
        "puis", "ensuite", "enfin", "également", "notamment",
        "ailleurs", "effet", "plus", "outre", "conséquent"
    ]
    connecteur_count = sum(1 for t in tokens if t in connecteurs)
    
    return {
        "avg_sentence_length": round(avg_sent_len, 2),
        "long_word_ratio": round(long_word_ratio, 4),
        "connecteur_count": connecteur_count,
        "total_tokens": len(tokens),
        "total_sentences": len(sentences),
    }


# =============================================================================
# 7. ÉVALUATION COMPLÈTE
# =============================================================================

def evaluate_single_text(
    text: str, 
    ppl_calculator: Optional[PerplexityCalculator] = None
) -> Dict[str, Any]:
    """
    Évalue un seul texte avec toutes les métriques.
    """
    results = {}
    
    # Richesse lexicale
    results["ttr"] = round(compute_ttr(text), 4)
    results["msttr_100"] = round(compute_msttr(text, 100), 4)
    results["lexical_entropy"] = round(compute_lexical_entropy(text), 4)
    results["hapax_ratio"] = round(compute_hapax_ratio(text), 4)
    
    # Lisibilité
    results["flesch_reading_ease_fr"] = round(compute_flesch_reading_ease_fr(text), 2)
    
    # Perplexité
    if ppl_calculator and ppl_calculator.model is not None:
        results["perplexity"] = round(ppl_calculator.compute_pseudo_perplexity(text), 2)
    else:
        results["perplexity_estimate"] = round(
            PerplexityCalculator()._estimate_perplexity(text), 2
        )
    
    # SLOR
    ppl_value = results.get("perplexity", results.get("perplexity_estimate", 100))
    results["slor"] = round(compute_slor(text, ppl_value), 4)
    
    # Fluence
    fluency = compute_fluency_metrics(text)
    results.update({f"fluency_{k}": v for k, v in fluency.items()})
    
    # Longueur
    results["char_count"] = len(text)
    results["word_count"] = len(tokenize_french(text))
    
    return results


def evaluate_corpus(
    texts: List[str], 
    records: List[Dict] = None,
    use_gpu_ppl: bool = False
) -> Dict[str, Any]:
    """
    Évalue un corpus complet de textes.
    
    Args:
        texts: Liste des textes à évaluer
        records: Enregistrements originaux (pour mention_rate)
        use_gpu_ppl: Utiliser CamemBERT pour la perplexité (plus lent mais précis)
    """
    # Initialiser le calculateur de perplexité si demandé
    ppl_calculator = None
    if use_gpu_ppl and TRANSFORMERS_AVAILABLE:
        ppl_calculator = PerplexityCalculator()
        ppl_calculator.load_model()
    
    # Métriques individuelles
    print("[...] Calcul des métriques individuelles...")
    individual_metrics = []
    for i, text in enumerate(texts):
        metrics = evaluate_single_text(text, ppl_calculator)
        individual_metrics.append(metrics)
        if (i + 1) % 10 == 0:
            print(f"      {i+1}/{len(texts)} textes traités")
    
    # Agrégation
    print("[...] Agrégation des métriques...")
    aggregated = {}
    for key in individual_metrics[0].keys():
        values = [m[key] for m in individual_metrics]
        aggregated[f"{key}_mean"] = round(float(np.mean(values)), 4)
        aggregated[f"{key}_std"] = round(float(np.std(values)), 4)
        aggregated[f"{key}_min"] = round(float(np.min(values)), 4)
        aggregated[f"{key}_max"] = round(float(np.max(values)), 4)
    
    # Self-BLEU (diversité du corpus)
    print("[...] Calcul du Self-BLEU...")
    aggregated["self_bleu_4"] = round(float(compute_self_bleu(texts, max_n=4)), 4)
    aggregated["self_bleu_3"] = round(float(compute_self_bleu(texts, max_n=3)), 4)
    aggregated["self_bleu_2"] = round(float(compute_self_bleu(texts, max_n=2)), 4)
    
    # Mention rate si disponible
    if records:
        mention_rates = [r.get("mention_rate", 0) for r in records]
        aggregated["mention_rate_mean"] = round(float(np.mean(mention_rates)), 4)
        aggregated["mention_rate_std"] = round(float(np.std(mention_rates)), 4)
    
    return {
        "individual": individual_metrics,
        "aggregated": aggregated,
        "corpus_size": len(texts),
    }


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Évaluation automatique des CR de radiothérapie",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python evaluate_automatic.py --input results.jsonl --output metrics.json
  python evaluate_automatic.py --input results.jsonl --output metrics.json --use_gpu_ppl
  python evaluate_automatic.py --input results.jsonl --output metrics.json --detailed
        """
    )
    ap.add_argument("--input", required=True, help="Fichier JSONL des générations")
    ap.add_argument("--output", required=True, help="Fichier JSON des résultats")
    ap.add_argument("--text_field", default="cr_text", help="Champ contenant le texte")
    ap.add_argument("--detailed", action="store_true", help="Inclure métriques individuelles")
    ap.add_argument("--use_gpu_ppl", action="store_true", 
                    help="Utiliser CamemBERT pour perplexité (nécessite GPU)")
    
    args = ap.parse_args()
    
    # Chargement des données
    input_path = Path(args.input)
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"[OK] Chargé {len(records)} enregistrements depuis {input_path}")
    
    # Extraction des textes
    texts = [r[args.text_field] for r in records if args.text_field in r]
    print(f"[OK] {len(texts)} textes à évaluer")
    
    # Évaluation
    results = evaluate_corpus(texts, records, use_gpu_ppl=args.use_gpu_ppl)
    
    # Ajout de métadonnées
    results["metadata"] = {
        "input_file": str(input_path),
        "text_field": args.text_field,
        "num_texts": len(texts),
        "used_gpu_ppl": args.use_gpu_ppl,
    }
    
    # Suppression des métriques individuelles si pas demandées
    if not args.detailed:
        del results["individual"]
    
    # Convertir les types numpy en types Python natifs
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
    
    # Sauvegarde
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Résultats sauvegardés dans {output_path}")
    
    # Affichage résumé
    print("\n" + "=" * 70)
    print("                    RÉSUMÉ DES MÉTRIQUES")
    print("=" * 70)
    
    agg = results["aggregated"]
    
    print(f"\nRICHESSE LEXICALE:")
    print(f"   TTR moyen:              {agg['ttr_mean']:.4f} (±{agg['ttr_std']:.4f})")
    print(f"   MSTTR-100 moyen:        {agg['msttr_100_mean']:.4f} (±{agg['msttr_100_std']:.4f})")
    print(f"   Entropie lexicale:      {agg['lexical_entropy_mean']:.4f} (±{agg['lexical_entropy_std']:.4f})")
    print(f"   Ratio hapax:            {agg['hapax_ratio_mean']:.4f} (±{agg['hapax_ratio_std']:.4f})")
    
    print(f"\nLISIBILITÉ:")
    print(f"   Flesch RE (français):   {agg['flesch_reading_ease_fr_mean']:.2f} (±{agg['flesch_reading_ease_fr_std']:.2f})")
    
    print(f"\n📈 FLUENCE:")
    ppl_key = "perplexity_mean" if "perplexity_mean" in agg else "perplexity_estimate_mean"
    if ppl_key in agg:
        print(f"   Perplexité:             {agg[ppl_key]:.2f} (±{agg[ppl_key.replace('_mean', '_std')]:.2f})")
    print(f"   SLOR:                   {agg['slor_mean']:.4f} (±{agg['slor_std']:.4f})")
    
    print(f"\n🔄 DIVERSITÉ (Self-BLEU, plus bas = plus diversifié):")
    print(f"   Self-BLEU-2:            {agg['self_bleu_2']:.4f}")
    print(f"   Self-BLEU-3:            {agg['self_bleu_3']:.4f}")
    print(f"   Self-BLEU-4:            {agg['self_bleu_4']:.4f}")
    
    if "mention_rate_mean" in agg:
        print(f"\n✅ CONFORMITÉ:")
        print(f"   Mention rate:           {agg['mention_rate_mean']:.2%} (±{agg['mention_rate_std']:.2%})")
    
    print(f"\n📏 LONGUEUR:")
    print(f"   Mots/texte:             {agg['word_count_mean']:.0f} (±{agg['word_count_std']:.0f})")
    print(f"   Caractères/texte:       {agg['char_count_mean']:.0f} (±{agg['char_count_std']:.0f})")
    print(f"   Phrases/texte:          {agg['fluency_total_sentences_mean']:.1f} (±{agg['fluency_total_sentences_std']:.1f})")
    
    print("\n" + "=" * 70)
    print("                    GRILLE D'INTERPRÉTATION")
    print("=" * 70)
    print("""
    | Métrique      | Excellent | Bon       | Acceptable | Faible    |
    |---------------|-----------|-----------|------------|-----------|
    | TTR           | > 0.5     | 0.4-0.5   | 0.3-0.4    | < 0.3     |
    | Self-BLEU     | < 0.3     | 0.3-0.5   | 0.5-0.7    | > 0.7     |
    | Mention Rate  | > 90%     | 70-90%    | 50-70%     | < 50%     |
    | Flesch RE     | > 60      | 40-60     | 20-40      | < 20      |
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
