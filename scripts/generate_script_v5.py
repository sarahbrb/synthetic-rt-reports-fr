"""
Batch generation of synthetic end-of-radiotherapy reports (breast cancer).
VERSION 6 changes:
- Stage pools: cT stages for néoadjuvant/palliatif, pT stages for adjuvant/récidive
- 7 clinical conditioning rules (see light_breast_profile_v4 docstring)
- Derived fields: her2_score, rh_status, n_status, irradiation_ganglionnaire,
                  prior_treatment, hormono_rule
- Post-generation validator check_clinical_coherence() with 6 violation flags:
  v_hormono_rh_neg, v_ct_notation_post_chir, v_n_plus_no_ganglion,
  v_m1_not_palliatif, v_recidive_no_history, v_etalement_incoherent

to launch:
cd ~/Data_Generation_2
conda activate synthgen

python scripts/generate_script_v6.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --mode chat \
  --system_prompt prompts/system_prompt.txt \
  --user_prompt prompts/user_prompt_v6.txt \
  --out outputs/test_mistral7b_gen.jsonl \
  --n 3 \
  --seed0 1000 \
  --dtype fp16 \
  --temperature 0.4 \
  --top_p 0.9 \
  --repetition_penalty 1.1 \
  --max_new_tokens 1600

"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import re
import time
from typing import Dict, Tuple, Any, Optional, List
from datetime import datetime, timedelta


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from faker import Faker


# 1. PATIENT IDENTITY GENERATION WITH FAKER


def generate_patient_identity_faker(faker_instance: Faker, rng: random.Random) -> Dict[str, Any]:
    """Generate patient identity using Faker library for more realistic data."""
    faker_instance.seed_instance(rng.randint(0, 999999))
    
    is_female = rng.random() < 0.80
    
    if is_female:
        prenom = faker_instance.first_name_female()
        nom = faker_instance.last_name()
        
        if rng.random() < 0.60:
            nom_jeune_fille = faker_instance.last_name()
            while nom_jeune_fille == nom:
                nom_jeune_fille = faker_instance.last_name()
        else:
            nom_jeune_fille = None
        
        civilite = "Madame"
        sex = "F"
    else:
        prenom = faker_instance.first_name_male()
        nom = faker_instance.last_name()
        nom_jeune_fille = None
        civilite = "Monsieur"
        sex = "M"
    
    age = rng.randint(40, 80)
    birth_date = faker_instance.date_of_birth(minimum_age=age, maximum_age=age)
    birth_date_str = birth_date.strftime("%d/%m/%Y")
    
    if nom_jeune_fille:
        patient_name = f"{nom.upper()} {prenom}, née {nom_jeune_fille.upper()}"
    else:
        patient_name = f"{nom.upper()} {prenom}"
    
    return {
        "civilite": civilite,
        "prenom": prenom,
        "nom": nom,
        "nom_jeune_fille": nom_jeune_fille,
        "patient_name": patient_name,
        "sex": sex,
        "birth_date": birth_date_str,
    }



# 2. CLINICAL CONDITIONING

# ── Stage pools — context-aware (Règle 1 : cT→néoadjuvant/palliatif, pT→adjuvant/récidive) ──
_CT_NEOADJ = ["cT1N1", "cT2N0", "cT2N1", "cT2N2", "cT3N0", "cT3N1", "cT3N2"]
# cT1N0 excluded (neoadjuvant for T1N0 is very rare — trial setting only)
# For RH+/HER2- neoadjuvant: large tumour required — restrict to cT2N1+/cT3 (Règle 5)
_CT_NEOADJ_LUMINAL = ["cT2N1", "cT3N0", "cT3N1", "cT3N2"]
# For palliatif: M1 is the main indication (4/6 = 67%); non-M1 cT3 locally advanced is rare (2/6)
# cT2 without M1 excluded (curative intent) — max stage = cT3 (Règle 2 : M1 → palliatif only)
# M0 omitted for non-M1 cases (clinically implicit, never written in French reports)
_CT_PALLIATIF = ["cT3N0M1", "cT3N1M1", "cT3N1M1", "cT3N2M1", "cT3N0", "cT3N2"]
# pT1cN0 weighted 2/6 (dominant in screening era); pT3N0 removed (pT3 → usually N+)
_PT_ADJUVANT  = ["pT1cN0", "pT1cN0", "pT1cN1", "pT2N0", "pT2N1", "pT3N1"]
# Récidive locale: pT1N0 included (small early nodule — clinically realistic)
_PT_RECIDIVE  = ["pT1N0", "pT2N0", "pT2N1", "pT3N1"]

# Subtypes allowed for male patients (Règle 4 : Monsieur → no TN, no RH-/HER2+)
_MALE_SUBTYPES = ["RH+/HER2-", "RH+/HER2+"]

# Prior treatment descriptions for récidive / néoadjuvant contexts
_PRIOR_TX_RECIDIVE = {
    "RH+/HER2-":    ["tumorectomie du {side} avec hormonothérapie adjuvante",
                     "mastectomie du {side} avec radiothérapie et hormonothérapie"],
    "RH+/HER2+":    ["tumorectomie du {side} avec chimiothérapie adjuvante et trastuzumab",
                     "mastectomie du {side} avec chimiothérapie et thérapie ciblée anti-HER2"],
    "RH-/HER2+":    ["tumorectomie du {side} avec chimiothérapie adjuvante et trastuzumab",
                     "mastectomie du {side} avec chimiothérapie et pertuzumab-trastuzumab"],
    "triple négatif":["tumorectomie du {side} avec chimiothérapie adjuvante",
                     "mastectomie du {side} avec chimiothérapie adjuvante"],
}
_PRIOR_TX_NEOADJ = {
    "RH+/HER2-":    "chimiothérapie néoadjuvante",
    "RH+/HER2+":    "chimiothérapie néoadjuvante associée à une thérapie ciblée anti-HER2 (trastuzumab + pertuzumab)",
    "RH-/HER2+":    "chimiothérapie néoadjuvante associée à une thérapie ciblée anti-HER2 (trastuzumab + pertuzumab)",
    "triple négatif":"chimiothérapie néoadjuvante (protocole AC-T)",
}


def _sample_stage(rng: random.Random, context: str, subtype: str) -> str:
    """Sample a clinically coherent stage for the given context and subtype."""
    if context == "adjuvant":
        return rng.choice(_PT_ADJUVANT)
    elif context == "récidive locale":
        return rng.choice(_PT_RECIDIVE)
    elif context == "néoadjuvant":
        pool = _CT_NEOADJ_LUMINAL if subtype == "RH+/HER2-" else _CT_NEOADJ
        return rng.choice(pool)
    else:  # palliatif
        return rng.choice(_CT_PALLIATIF)


def _derive_fields(rng: random.Random, identity: Dict, side: str,
                   subtype: str, stage: str, context: str) -> Dict[str, Any]:
    """Derive all clinical fields from the primary conditioning variables."""
    # HER2 score (1+/2+/3+ scale — Résolution erreur 10)
    if subtype in ["RH+/HER2+", "RH-/HER2+"]:
        her2_score = "3+"
    else:
        her2_score = rng.choice(["0", "1+"])

    # RH status
    rh_positive = subtype in ["RH+/HER2-", "RH+/HER2+"]
    rh_status = "positifs" if rh_positive else "négatifs"

    # N status and irradiation ganglionnaire (Résolution erreur 5)
    n_match = re.search(r'N(\d+)', stage, re.IGNORECASE)
    n_status = int(n_match.group(1)) if n_match else 0
    irradiation_ganglionnaire = n_status >= 1

    # Prior treatment for récidive locale / néoadjuvant (Résolution erreur 9)
    if context == "récidive locale":
        template = rng.choice(_PRIOR_TX_RECIDIVE.get(subtype, ["traitement initial du {side}"]))
        prior_treatment = template.format(side=side)
    elif context == "néoadjuvant":
        prior_treatment = _PRIOR_TX_NEOADJ.get(subtype, "traitement systémique néoadjuvant")
    else:
        prior_treatment = "premier traitement"

    # Hormono rule — injected explicitly in prompt (Résolution erreur 3)
    if rh_positive:
        hormono_rule = (
            "Le sous-type est RH+. Une hormonothérapie adjuvante est OBLIGATOIRE "
            "(anti-aromatase si ménopausée, Tamoxifène sinon)."
        )
    else:
        hormono_rule = (
            f"Le sous-type est {subtype} (RH–). "
            "AUCUNE hormonothérapie ne doit apparaître dans ce compte rendu "
            "(Tamoxifène, Létrozole, Anastrozole et tout autre anti-hormonal sont INTERDITS). "
            "Mentionner explicitement l'absence d'indication d'hormonothérapie."
        )

    # Irradiation ganglionnaire instruction (Résolution erreur 5)
    if irradiation_ganglionnaire:
        irradiation_ganglionnaire_text = (
            "oui — irradiation des chaînes axillaires et sus-claviculaires REQUISE "
            "(patient N1 : décrire explicitement le volume d'irradiation ganglionnaire)"
        )
    else:
        irradiation_ganglionnaire_text = (
            "non — patient N0, irradiation mammaire seule (ne pas mentionner d'irradiation ganglionnaire)"
        )

    return {
        "her2_score":                  her2_score,
        "rh_status":                   rh_status,
        "n_status":                    n_status,
        "irradiation_ganglionnaire":   irradiation_ganglionnaire_text,
        "prior_treatment":             prior_treatment,
        "hormono_rule":                hormono_rule,
    }


def light_breast_profile_v4(seed: int, faker_fr: Faker) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Generate a clinically coherent patient profile.

    Clinical rules applied (from Mme Brun's feedback analysis):
      Rule 1 — cT stages → néoadjuvant / palliatif only
               pT stages → adjuvant / récidive locale only
      Rule 2 — M1 stage  → palliatif only
      Rule 3 — Early stage (cT1) + palliatif → impossible (filtered)
      Rule 4 — Monsieur → subtype ∈ {RH+/HER2-, RH+/HER2+} only
      Rule 5 — RH+/HER2- + néoadjuvant → cT3 / cT2N1 only (early stages implausible)
    Derived fields added: her2_score, rh_status, irradiation_ganglionnaire,
                          prior_treatment, hormono_rule.
    """
    rng = random.Random(seed)
    identity = generate_patient_identity_faker(faker_fr, rng)

    side = rng.choice(["sein gauche", "sein droit"])
    context = rng.choice(["adjuvant", "néoadjuvant", "récidive locale", "palliatif"])

    # Rule 4: Monsieur → restricted subtype pool
    if identity["civilite"] == "Monsieur":
        subtype = rng.choice(_MALE_SUBTYPES)
    else:
        subtype = rng.choice(["RH+/HER2-", "RH+/HER2+", "RH-/HER2+", "triple négatif"])

    # Rules 1, 2, 3, 5: context-aware + subtype-aware stage sampling
    stage = _sample_stage(rng, context, subtype)

    # Derived clinical fields
    derived = _derive_fields(rng, identity, side, subtype, stage, context)

    fields_raw = {
        # Identity
        "civilite":                  identity["civilite"],
        "patient_name":              identity["patient_name"],
        "birth_date":                identity["birth_date"],
        # Primary clinical fields
        "side":                      side,
        "subtype":                   subtype,
        "stage":                     stage,
        "context":                   context,
        # Derived fields (injected into prompt)
        "her2_score":                derived["her2_score"],
        "rh_status":                 derived["rh_status"],
        "irradiation_ganglionnaire": derived["irradiation_ganglionnaire"],
        "prior_treatment":           derived["prior_treatment"],
        "hormono_rule":              derived["hormono_rule"],
    }

    fields_str = {
        "nom":                       identity["nom"].upper(),
        "prenom":                    identity["prenom"],
        "civilite":                  identity["civilite"],
        "birth_date":                identity["birth_date"],
        "side":                      side,
        "subtype":                   subtype,
        "stage":                     stage,
        "context":                   context,
        "her2_score":                derived["her2_score"],
        "irradiation_ganglionnaire": str(derived["n_status"] >= 1),
        "n_status":                  str(derived["n_status"]),
    }

    return fields_raw, fields_str



# 3. IMPROVED MENTION CHECKING (relaxed matching extended : I added some variants for clinical fields since they could be mentioned in several ways)

def normalize_text(text: str) -> str:
    # Normalizing text for comparison: lowercase, removing extra spaces
    return re.sub(r'\s+', ' ', text.lower()).strip()


def _check_subtype_ok(subtype_value: str, t_lower: str) -> bool:
    """Checks if the correct molecular subtype is mentioned in the text.

    Two-pass strategy:
    1. Direct notation (e.g. "rh+/her2-") — fast, unambiguous.
    2. Composite AND check: both RH status AND HER2 status must be explicitly
       mentioned to avoid false positives from descriptors shared across subtypes.
       (e.g. "récepteurs hormonaux positifs" appears for both RH+/HER2- and RH+/HER2+
        → requires HER2 qualifier to discriminate.)
    """
    _DIRECT: Dict[str, List[str]] = {
        "RH+/HER2-":     ["rh+/her2-",      "rh+ her2-",    "rh + her2 -",   "rh+her2-",   "rh +/her2-"],
        "RH+/HER2+":     ["rh+/her2+",      "rh+ her2+",    "rh + her2 +",   "rh+her2+",   "rh +/her2+"],
        "RH-/HER2+":     ["rh-/her2+",      "rh- her2+",    "rh - her2 +",   "rh-her2+",   "rh -/her2+"],
        "triple négatif": ["triple négatif", "triple-négatif", "triple negatif"],
    }
    if any(v in t_lower for v in _DIRECT.get(subtype_value, [subtype_value.lower()])):
        return True

    # Composite: BOTH RH and HER2 status must be explicitly present
    rh_pos = any(v in t_lower for v in [
        "récepteurs hormonaux positifs", "exprimant les récepteurs hormonaux",
        "rh positifs", "récepteurs hormonaux (rh) positifs",
    ])
    rh_neg = any(v in t_lower for v in [
        "récepteurs hormonaux négatifs", "n'exprimant pas les récepteurs hormonaux",
        "rh négatifs", "récepteurs hormonaux (rh) négatifs",
    ])
    her2_pos = any(v in t_lower for v in [
        "her2 positif", "her2 surexprimé", "her2 amplifié",
        "surexpression de her2", "surexpression her2", "her2 3+",
    ])
    her2_neg = any(v in t_lower for v in [
        "her2 négatif", "her2 non surexprimé", "her2 non amplifié",
        "her2 0", "her2 1+", "her2 score 0", "her2 score 1+",
    ])

    if subtype_value == "RH+/HER2-":
        return rh_pos and her2_neg
    if subtype_value == "RH+/HER2+":
        return rh_pos and her2_pos
    if subtype_value == "RH-/HER2+":
        return rh_neg and her2_pos
    if subtype_value == "triple négatif":
        return (rh_neg and her2_neg) or "basal-like" in t_lower
    return subtype_value.lower() in t_lower


def check_mentions_v4_extended(text: str, fields_str: Dict[str, str]) -> Dict[str, Any]:
    """
    MATCHING STRATEGIES:
    - EXACT:       nom, prenom, civilite, birth_date
    - RELAXED:     side (compound terms only), subtype (via _check_subtype_ok),
                   stage (TNM components), context (specific variants only)
    - CONTEXTUAL:  her2_score (regex for ambiguous "0"), irradiation_ganglionnaire (N+ only)
    """
    t_normalized = normalize_text(text)
    t_lower = text.lower()

    # a. EXACT MATCHES
    nom_ok      = fields_str["nom"].lower() in t_lower
    prenom_ok   = fields_str["prenom"].lower() in t_lower
    civilite_ok = fields_str["civilite"].lower() in t_lower

    birth_date_normalized = fields_str["birth_date"].replace("/", "").replace("-", "").replace(".", "")
    birth_date_ok = (
        fields_str["birth_date"] in text or
        birth_date_normalized in text.replace("/", "").replace("-", "").replace(".", "")
    )

    # b. RELAXED MATCHES

    # b.1. SIDE — compound anatomical terms only
    # Bare "gauche"/"droit" removed: too ambiguous in French ("il a le droit de…")
    side_value = fields_str["side"].lower()
    side_variants = {
        "sein gauche": ["sein gauche", "hémithorax gauche", "côté gauche",
                        "mammaire gauche", "du sein gauche"],
        "sein droit":  ["sein droit",  "hémithorax droit",  "côté droit",
                        "mammaire droit",  "du sein droit"],
    }
    if side_value in side_variants:
        side_ok = any(variant in t_lower for variant in side_variants[side_value])
    else:
        side_ok = side_value in t_lower

    # b.2. SUBTYPE — two-pass: direct notation → composite AND check
    subtype_ok = _check_subtype_ok(fields_str["subtype"], t_lower)

    # b.3. STAGE — numeric TNM components (cT vs pT prefix handled by check_clinical_coherence)
    # Fix: [a-d]? instead of [a-z]* — prevents greedily capturing N/M letters with IGNORECASE.
    # (N=14th letter, M=13th letter — both outside [a-d] range → safe delimiter)
    stage_value = fields_str["stage"]
    t_match = re.search(r'[pcy]?T(\d+[a-d]?)', stage_value, re.IGNORECASE)
    n_match = re.search(r'[pcy]?N(\d+[a-d]?)', stage_value, re.IGNORECASE)
    m_match = re.search(r'M(\d+[a-d]?)',        stage_value, re.IGNORECASE)

    stage_components: List[str] = []
    if t_match:
        tval = t_match.group(1).lower()
        stage_components += [f"t{tval}", f"t {tval}"]
    if n_match:
        nval = n_match.group(1).lower()
        stage_components += [f"n{nval}", f"n {nval}"]
    # M0 is clinically implicit → never written in French reports for non-metastatic cases.
    # Only require M in the text when it is M1+ (metastatic).
    if m_match and m_match.group(1) != "0":
        mval = m_match.group(1).lower()
        stage_components += [f"m{mval}", f"m {mval}"]

    stage_exact = stage_value.lower() in t_lower
    stage_components_match = (
        all(
            any(comp in t_lower for comp in [stage_components[i], stage_components[i + 1]])
            for i in range(0, len(stage_components), 2)
        )
        if stage_components else False
    )
    stage_ok = stage_exact or stage_components_match

    # b.4. CONTEXT — specific variants only
    # Removed: "locale" (matches "anesthésie locale"), "avancé"/"généralisé" (too generic),
    #           "induction"/"première ligne" (too ambiguous), "complément" (too short)
    context_value = fields_str["context"].lower()
    context_variants: Dict[str, List[str]] = {
        "adjuvant": [
            "adjuvant", "adjuvante",
            "post-opératoire", "post opératoire",
            "après chirurgie", "après intervention",
            "traitement complémentaire", "complément de traitement", "complémentaire",
        ],
        "néoadjuvant": [
            "néoadjuvant", "néo-adjuvant", "néo adjuvant",
            "pré-opératoire", "pré opératoire", "préopératoire",
            "avant chirurgie", "avant intervention",
            "chimiothérapie d'induction", "traitement d'induction",
        ],
        "récidive locale": [
            "récidive", "récurrence", "rechute",
            "récidive locale", "locorégionale", "locorégional",
            "reprise évolutive", "progression locale",
        ],
        "palliatif": [
            "palliatif", "palliative",
            "métastatique", "stade iv", "stade 4",
            "soins de confort", "à visée palliative",
        ],
    }
    if context_value in context_variants:
        context_ok = any(variant in t_lower for variant in context_variants[context_value])
    else:
        context_ok = context_value in t_lower

    # b.5. HER2 SCORE — "0" requires HER2 context (bare "0" matches any digit in text)
    her2_score_value = fields_str.get("her2_score", "")
    if not her2_score_value:
        her2_score_ok = False
    elif her2_score_value == "0":
        her2_score_ok = bool(re.search(r'her2[\s:]*0\b', t_lower))
    else:
        # "1+", "2+", "3+" are specific enough to match directly
        her2_score_ok = her2_score_value in t_lower

    # b.6. IRRADIATION GANGLIONNAIRE — only meaningful when N+
    n_status = int(fields_str.get("n_status", "0") or "0")
    if n_status >= 1:
        ganglion_terms = ["axillaire", "sus-claviculaire", "ganglionnaire", "mammaire interne",
                          "chaîne", "creux axillaire"]
        irradiation_ganglionnaire_ok = any(t in t_lower for t in ganglion_terms)
    else:
        irradiation_ganglionnaire_ok = True   # N0: irradiation ganglionnaire not expected → no violation

    # AGGREGATE FLAGS
    flags = {
        "nom_ok":                     bool(nom_ok),
        "prenom_ok":                  bool(prenom_ok),
        "civilite_ok":                bool(civilite_ok),
        "birth_date_ok":              bool(birth_date_ok),
        "side_ok":                    bool(side_ok),
        "subtype_ok":                 bool(subtype_ok),
        "stage_ok":                   bool(stage_ok),
        "context_ok":                 bool(context_ok),
        "her2_score_ok":              bool(her2_score_ok),
        "irradiation_ganglionnaire_ok": bool(irradiation_ganglionnaire_ok),
    }

    # Core clinical fields (patient name excluded — clinical fields only)
    core_clinical_flags = [side_ok, subtype_ok, stage_ok, context_ok]
    core_clinical_rate = sum(core_clinical_flags) / len(core_clinical_flags)

    # All fields (comprehensive completeness)
    all_flags = [nom_ok, prenom_ok, civilite_ok, birth_date_ok,
                 side_ok, subtype_ok, stage_ok, context_ok]
    completeness_rate = sum(all_flags) / len(all_flags)

    flags["core_clinical_rate"] = round(float(core_clinical_rate), 3)
    flags["completeness_rate"]  = round(float(completeness_rate), 3)

    return flags


def assess_quality_threshold(flags: Dict[str, Any], 
                            threshold_core: float = 0.80,
                            threshold_complete: float = 0.70) -> Dict[str, Any]:
    """Assess if generation meets quality thresholds."""
    assessment = {
        "is_good_quality": flags["core_clinical_rate"] >= threshold_core,
        "is_excellent_quality": flags["completeness_rate"] >= threshold_complete,
        "quality_tier": "poor"
    }
    
    if flags["completeness_rate"] >= threshold_complete:
        assessment["quality_tier"] = "excellent"
    elif flags["core_clinical_rate"] >= threshold_core:
        assessment["quality_tier"] = "good"
    elif flags["core_clinical_rate"] >= 0.60:
        assessment["quality_tier"] = "acceptable"
    else:
        assessment["quality_tier"] = "poor"
    
    return assessment

# Quality tiers:
#   excellent  : completeness_rate ≥ 0.70
#   good       : core_clinical_rate ≥ 0.80
#   acceptable : core_clinical_rate ≥ 0.60
#   poor       : otherwise


# 3b. POST-GENERATION CLINICAL COHERENCE VALIDATOR

_HORMONO_TERMS = [
    # Specific drug names only — never appear legitimately in negation contexts
    "tamoxifène", "tamoxifen", "nolvadex",
    "létrozole", "letrozole", "femara",
    "anastrozole", "arimidex", "anastrazole",
    "fulvestrant", "faslodex",
    "examestane", "aromasine",
    # NOTE: "hormonothérapie", "anti-aromatase", "antiestrogène" intentionally excluded:
    # the prompt requires RH− CRs to explicitly mention absence of hormono → these words
    # appear in negation context ("Aucune hormonothérapie n'est indiquée") and would
    # trigger a false positive.
]
_GANGLION_TERMS = [
    "axillaire", "sus-claviculaire", "susclaviculaire",
    "ganglionnaire", "mammaire interne", "chaîne", "creux axillaire",
]


def check_clinical_coherence(text: str, fields_str: Dict[str, str]) -> Dict[str, Any]:
    """Post-generation validator for clinical coherence rules.

    Returns a dict with one boolean per rule (True = VIOLATION detected).
    Also returns a total violation count and a composite coherence_score.
    """
    t_lower = text.lower()
    subtype  = fields_str.get("subtype", "")
    stage    = fields_str.get("stage", "")
    context  = fields_str.get("context", "")
    n_status = int(fields_str.get("n_status", "0") or "0")

    rh_negative = subtype in ["triple négatif", "RH-/HER2+"]
    is_pT_stage  = stage.lower().startswith("pt")

    # Rule A: RH− → no hormonothérapie (Résolution erreur 3)
    hormono_in_text = any(t in t_lower for t in _HORMONO_TERMS)
    violation_hormono = rh_negative and hormono_in_text

    # Rule B: pT stage → text must not use cT notation (Résolution erreur 1)
    ct_in_text = bool(re.search(r'\bct\d', t_lower))
    violation_ct_notation = is_pT_stage and ct_in_text

    # Rule C: N+ → irradiation ganglionnaire must be mentioned (Résolution erreur 5)
    ganglion_in_text = any(t in t_lower for t in _GANGLION_TERMS)
    violation_n_plus_no_ganglion = (n_status >= 1) and not ganglion_in_text

    # Rule D: M1 stage → should describe metastatic/palliatif intent
    is_m1 = "m1" in stage.lower()
    palliatif_terms = ["palliatif", "palliative", "métastatique", "soins de confort"]
    palliatif_in_text = any(t in t_lower for t in palliatif_terms)
    violation_m1_not_palliatif = is_m1 and not palliatif_in_text

    # Rule E: récidive locale → prior treatment must be mentioned
    prior_terms = ["traitement antérieur", "précédent", "initiale", "récidive", "rechute",
                   "radiothérapie antérieure", "chirurgie antérieure", "a bénéficié",
                   "avait été traitée", "avait eu"]
    violation_recidive_no_history = (
        context == "récidive locale" and not any(t in t_lower for t in prior_terms)
    )

    # Rule F: étalement / n_fractions ratio must be in [1.2, 2.5]
    # étalement = calendar days from first to last fraction (inclusive)
    # 5 séances/week → ratio ≈ 1.4–1.5 with no interruption; up to 2.0 with pauses
    etalement_match = re.search(r'étalement\s+de\s+(\d+)\s+jour', t_lower)
    fractions_match  = re.search(r'(\d{1,3})\s+fractions?', t_lower)
    violation_etalement = False
    if etalement_match and fractions_match:
        etalement_days = int(etalement_match.group(1))
        n_frac = int(fractions_match.group(1))
        if n_frac > 0:
            ratio = etalement_days / n_frac
            violation_etalement = not (1.2 <= ratio <= 2.5)

    violations = {
        "v_hormono_rh_neg":         bool(violation_hormono),
        "v_ct_notation_post_chir":  bool(violation_ct_notation),
        "v_n_plus_no_ganglion":     bool(violation_n_plus_no_ganglion),
        "v_m1_not_palliatif":       bool(violation_m1_not_palliatif),
        "v_recidive_no_history":    bool(violation_recidive_no_history),
        "v_etalement_incoherent":   bool(violation_etalement),
    }
    n_violations = sum(1 for v in violations.values() if v)
    violations["n_violations"]    = n_violations
    violations["coherence_score"] = round(1.0 - n_violations / len([k for k in violations if k.startswith("v_")]), 3)
    return violations


# 4. MODEL LOADING

def _auto_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, minor = torch.cuda.get_device_capability()
    if major >= 8:
        return torch.bfloat16
    return torch.float16


def build_model(model_name: str, dtype: str = "auto"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception as e_fast:
        print(f"[warn] Fast tokenizer load failed: {type(e_fast).__name__}: {e_fast}")
        print("[warn] Falling back to slow tokenizer (use_fast=False).")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if dtype == "auto":
        torch_dtype = _auto_dtype()
    else:
        dtype_map = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported --dtype '{dtype}'. Use: auto|fp16|bf16|fp32")
        torch_dtype = dtype_map[dtype]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


# 5. PROMPT RENDERING

def render_user_prompt(template: str, fields_raw: Dict[str, Any]) -> str:
    """Replace placeholders."""
    try:
        return template.format(**fields_raw)
    except KeyError as e:
        raise KeyError(f"Missing placeholder in user prompt template: {e}") from e


def build_messages(system_content: str, user_content: str) -> List[Dict[str, str]]:
    system_content = (system_content or "").strip()
    user_content = (user_content or "").strip()
    messages: List[Dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    return messages


# 6. GENERATION
@torch.inference_mode()
def generate_one(
    tokenizer,
    model,
    mode: str,
    system_content: str,
    user_content: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if mode == "chat":
        messages = build_messages(system_content, user_content)

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)
            except Exception:
                # Fallback for models without system role support (e.g. BioMistral/Mistral-7B)
                merged = [{"role": "user", "content": f"{system_content}\n\n{user_content}".strip()}]
                input_ids = tokenizer.apply_chat_template(
                    merged,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)
        else:
            flat = ""
            if system_content:
                flat += f"[SYSTEM]\n{system_content}\n\n"
            flat += f"[USER]\n{user_content}\n\n[ASSISTANT]\n"
            input_ids = tokenizer(flat, return_tensors="pt").input_ids.to(model.device)
    else:
        flat = ""
        if system_content:
            flat += system_content.strip() + "\n\n"
        flat += user_content.strip()
        input_ids = tokenizer(flat, return_tensors="pt").input_ids.to(model.device)

    prompt_len = int(input_ids.shape[-1])

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    gen_ids = out[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    text = re.sub(r"^(assistant|assistant:)\s*", "", text, flags=re.IGNORECASE).strip()
    return text



# 7. STATISTICS COMPUTATION
def compute_batch_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics across all generated records."""
    n = len(records)
    if n == 0:
        return {}
    
    quality_tiers = {"poor": 0, "acceptable": 0, "good": 0, "excellent": 0}
    total_core_rate = 0.0
    total_completeness_rate = 0.0
    
    field_counts = {
        "nom_ok": 0,
        "prenom_ok": 0,
        "civilite_ok": 0,
        "birth_date_ok": 0,
        "side_ok": 0,
        "subtype_ok": 0,
        "stage_ok": 0,
        "context_ok": 0,
        "her2_score_ok": 0,
    }
    
    for record in records:
        if "quality_tier" in record:
            quality_tiers[record["quality_tier"]] += 1
        
        total_core_rate += record.get("core_clinical_rate", 0)
        total_completeness_rate += record.get("completeness_rate", 0)
        
        for field in field_counts:
            if record.get(field, False):
                field_counts[field] += 1
    
    stats = {
        "n_samples": n,
        "avg_core_clinical_rate": round(total_core_rate / n, 3),
        "avg_completeness_rate": round(total_completeness_rate / n, 3),
        "quality_distribution": {
            tier: round(count / n * 100, 1) for tier, count in quality_tiers.items()
        },
        "field_success_rates": {
            field: round(count / n * 100, 1) for field, count in field_counts.items()
        },
    }
    
    return stats


# 8. MAIN
def main():
    ap = argparse.ArgumentParser(
        description="V6: Clinically conditioned breast radiotherapy report generation"
    )
    ap.add_argument("--model", required=True, help="HuggingFace model name or path")
    ap.add_argument("--mode", choices=["chat", "plain"], required=True, help="Generation mode")

    ap.add_argument("--system_prompt", required=True, help="Path to system prompt file")
    ap.add_argument("--user_prompt", required=True, help="Path to user prompt template with placeholders")

    ap.add_argument("--out", required=True, help="Output JSONL file path")
    ap.add_argument("--n", type=int, default=30, help="Number of samples to generate")
    ap.add_argument("--seed0", type=int, default=1000, help="Starting seed")

    ap.add_argument("--max_new_tokens", type=int, default=1600)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--dtype", default="auto", help="auto|fp16|bf16|fp32")

    ap.add_argument("--threshold_core", type=float, default=0.80)
    ap.add_argument("--threshold_complete", type=float, default=0.70)

    ap.add_argument("--print_messages", action="store_true")
    ap.add_argument("--print_every", type=int, default=1)
    ap.add_argument("--faker_locale", default="fr_FR")

    args = ap.parse_args()

    # Load prompts
    system_path = Path(args.system_prompt)
    user_path = Path(args.user_prompt)
    system_base = system_path.read_text(encoding="utf-8").strip()
    user_template = user_path.read_text(encoding="utf-8").strip()

    # Prepare output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize Faker
    faker_fr = Faker(args.faker_locale)
    print(f"[OK] Faker initialized with locale: {args.faker_locale}", flush=True)

    # Load model
    tokenizer, model = build_model(args.model, dtype=args.dtype)
    print(f"[OK] Model loaded: {args.model} | mode={args.mode} | dtype={args.dtype}", flush=True)

    # Storage for statistics
    all_records = []

    # Generation loop
    for i in range(args.n):
        seed = args.seed0 + i

        fields_raw, fields_str = light_breast_profile_v4(seed, faker_fr)
        user_content = render_user_prompt(user_template, fields_raw)
        system_content = system_base

        if args.print_messages and i == 0:
            print("\n" + "=" * 80)
            print("[SYSTEM]")
            print(system_content)
            print("\n[USER]")
            print(user_content)
            print("\n[CONDITIONING]")
            print(json.dumps(fields_str, ensure_ascii=False, indent=2))
            print("=" * 80 + "\n", flush=True)

        t0 = time.time()
        text = generate_one(
            tokenizer=tokenizer,
            model=model,
            mode=args.mode,
            system_content=system_content,
            user_content=user_content,
            seed=seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        dt = time.time() - t0

        # Mention matching
        mention_flags = check_mentions_v4_extended(text, fields_str)
        # Clinical coherence validation
        coherence = check_clinical_coherence(text, fields_str)

        quality_assessment = assess_quality_threshold(
            mention_flags,
            threshold_core=args.threshold_core,
            threshold_complete=args.threshold_complete
        )

        record = {
            "model": args.model,
            "prompt_file": {
                "system": system_path.name,
                "user": user_path.name,
            },
            "seed": seed,
            "gen_seconds_total": round(dt, 3),
            "system_content": system_content,
            "user_content": user_content,
            "fields": {
                "civilite":   fields_str["civilite"],
                "nom":        fields_str["nom"],
                "prenom":     fields_str["prenom"],
                "birth_date": fields_str["birth_date"],
                "side":       fields_str["side"],
                "subtype":    fields_str["subtype"],
                "stage":      fields_str["stage"],
                "context":    fields_str["context"],
                "her2_score": fields_str.get("her2_score", ""),
                "n_status":   fields_str.get("n_status", "0"),
            },
            **mention_flags,
            **coherence,
            **quality_assessment,
            "cr_text": text,
        }

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        all_records.append(record)

        if args.print_every > 0 and ((i + 1) % args.print_every == 0 or (i + 1) == args.n):
            patient_display = f"{fields_str['civilite']} {fields_str['nom']}"
            quality_display = quality_assessment["quality_tier"].upper()
            print(
                f"[OK] {i+1}/{args.n} | {dt:.1f}s | "
                f"core={mention_flags['core_clinical_rate']:.2f} | "
                f"complete={mention_flags['completeness_rate']:.2f} | "
                f"coh={coherence['coherence_score']:.2f} ({coherence['n_violations']}v) | "
                f"quality={quality_display} | {patient_display}",
                flush=True,
            )

    # Compute and display statistics
    print("\n" + "=" * 80)
    print("BATCH STATISTICS")
    print("=" * 80)
    
    stats = compute_batch_statistics(all_records)
    print(f"\nSamples generated: {stats['n_samples']}")
    print(f"Average core clinical rate: {stats['avg_core_clinical_rate']:.3f}")
    print(f"Average completeness rate: {stats['avg_completeness_rate']:.3f}")
    
    print("\nQuality distribution:")
    for tier, pct in stats['quality_distribution'].items():
        print(f"  {tier:12s}: {pct:5.1f}%")
    
    print("\nField success rates:")
    for field, pct in stats['field_success_rates'].items():
        print(f"  {field:15s}: {pct:5.1f}%")
    
    print("\n" + "=" * 80)
    print(f"[DONE] Results written to: {out_path}")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()