"""
Batch generation of synthetic end-of-radiotherapy reports (breast cancer).
VERSION 5 changes:
- Extended relaxed matching: side, subtype, stage, context (not just context)
- Improved mention checking with semantic variants for all clinical fields
- Maintains exact matching for: nom, prenom, civilite, birth_date
- Completeness scoring with configurable thresholds

to launch : 
python ~/Data_Generation/scripts/generate_script_v5.py \
  --model "mistralai/Mistral-7B-Instruct-v0.2" \      
  --mode chat \
  --system_prompt ~/Data_Generation/prompts/system_prompt.txt \
  --user_prompt ~/Data_Generation/prompts/user_prompt_v5.txt \
  --out ~/Data_Generation/datasets/mistral_5shot_v5.jsonl \
  --n 30 \
  --print_messages

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


def light_breast_profile_v4(seed: int, faker_fr: Faker) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Generate patient profile with Faker integration."""
    rng = random.Random(seed)
    
    identity = generate_patient_identity_faker(faker_fr, rng)
    
    side = rng.choice(["sein gauche", "sein droit"])
    subtype = rng.choice(["RH+/HER2-", "RH+/HER2+", "RH-/HER2+", "triple négatif"])
    stage = rng.choice(["cT1cN0M0", "cT2N0M0", "cT2N1M0", "cT3N0M0", "cT3N0M1", "cT3N1M0"])
    context = rng.choice(["adjuvant", "néoadjuvant", "récidive locale", "palliatif"])

    fields_raw = {
        "civilite": identity["civilite"],
        "patient_name": identity["patient_name"],
        "birth_date": identity["birth_date"],
        "side": side,
        "subtype": subtype,
        "stage": stage,
        "context": context,
    }
    
    fields_str = {
        "nom": identity["nom"].upper(),
        "prenom": identity["prenom"],
        "civilite": identity["civilite"],
        "birth_date": identity["birth_date"],
        "side": side,
        "subtype": subtype,
        "stage": stage,
        "context": context,
    }
    
    return fields_raw, fields_str



# 3. IMPROVED MENTION CHECKING (relaxed matching extended : I added some variants for clinical fields since they could be mentioned in several ways)

def normalize_text(text: str) -> str:
    # Normalizing text for comparison: lowercase, removing extra spaces
    return re.sub(r'\s+', ' ', text.lower()).strip()


def check_mentions_v4_extended(text: str, fields_str: Dict[str, str]) -> Dict[str, Any]:
    """    
    MATCHING STRATEGIES:
    - EXACT: nom, prenom, civilite, birth_date
    - RELAXED: side, subtype, stage, context
    """
    t_normalized = normalize_text(text)
    t_lower = text.lower()
    

    # a. EXACT MATCHES (must appear as-is)
    nom_ok = fields_str["nom"].lower() in t_lower
    prenom_ok = fields_str["prenom"].lower() in t_lower
    civilite_ok = fields_str["civilite"].lower() in t_lower
    
    birth_date_normalized = fields_str["birth_date"].replace("/", "").replace("-", "").replace(".", "")
    birth_date_ok = (
        fields_str["birth_date"] in text or
        birth_date_normalized in text.replace("/", "").replace("-", "").replace(".", "")
    )

    # b. RELAXED MATCHES
    
    # b.1. SIDE
    side_value = fields_str["side"].lower()
    side_variants = {
        "sein gauche": ["sein gauche", "gauche", "hémithorax gauche", "côté gauche"],
        "sein droit": ["sein droit", "droit", "hémithorax droit", "côté droit"],
    }
    
    if side_value in side_variants:
        side_ok = any(variant in t_lower for variant in side_variants[side_value])
    else:
        side_ok = side_value in t_lower
    
    # b.2.SUBTYPE
    subtype_value = fields_str["subtype"]
    subtype_variants = {
        "RH+/HER2-": [
            "rh+/her2-", "rh+ her2-", "rh + her2 -",
            "récepteurs hormonaux positifs", "her2 négatif",
        ],
        "RH+/HER2+": [
            "rh+/her2+", "rh+ her2+", "rh + her2 +",
            "récepteurs hormonaux positifs", "her2 positif",
        ],
        "RH-/HER2+": [
            "rh-/her2+", "rh- her2+", "rh - her2 +",
            "récepteurs hormonaux négatifs", "her2 positif", "HER2+",
            "her2 enrichi", "her2 amplifié", "récepteurs hormonaux RH-",
        ],
        "triple négatif": [
            "triple négatif", "triple-négatif", "tn",
            "rh-/her2-", "basal-like",
            "récepteurs hormonaux négatifs", "her2 négatif",
        ],
    }
    
    if subtype_value in subtype_variants:
        subtype_ok = any(variant in t_lower for variant in subtype_variants[subtype_value])
    else:
        subtype_ok = subtype_value.lower() in t_lower
    
    # b.3. STAGE 
    stage_value = fields_str["stage"]
    
    # Extract T, N, M components
    # Example: "cT2N1M0" -> T2, N1, M0
    t_match = re.search(r'c?T(\d+[a-z]*)', stage_value, re.IGNORECASE)
    n_match = re.search(r'N(\d+[a-z]*)', stage_value, re.IGNORECASE)
    m_match = re.search(r'M(\d+[a-z]*)', stage_value, re.IGNORECASE)
    
    stage_components = []
    if t_match:
        stage_components.append(f"t{t_match.group(1)}")
        stage_components.append(f"t {t_match.group(1)}")
    if n_match:
        stage_components.append(f"n{n_match.group(1)}")
        stage_components.append(f"n {n_match.group(1)}")
    if m_match:
        stage_components.append(f"m{m_match.group(1)}")
        stage_components.append(f"m {m_match.group(1)}")
    
    # Checking if stage or its components are mentioned
    stage_exact = stage_value.lower() in t_lower
    stage_components_match = all(any(comp in t_lower for comp in [stage_components[i], stage_components[i+1]]) 
                                  for i in range(0, len(stage_components), 2)) if stage_components else False
    
    stage_ok = stage_exact or stage_components_match
    
    # b.4. CONTEXT
    context_value = fields_str["context"].lower()
    context_variants = {
        "adjuvant": [
            "adjuvant", "adjuvante", "post-opératoire", "post opératoire",
            "après chirurgie", "après intervention",
            "complément", "complémentaire",
        ],
        "néoadjuvant": [
            "néoadjuvant", "néo-adjuvant", "néo adjuvant",
            "pré-opératoire", "pré opératoire", "préopératoire",
            "avant chirurgie", "avant intervention",
            "induction", "première ligne",
        ],
        "récidive locale": [
            "récidive", "récurrence", "rechute",
            "locale", "locorégionale", "locorégional",
            "reprise évolutive", "progression locale",
        ],
        "palliatif": [
            "palliatif", "palliative",
            "métastatique", "stade iv", "stade 4",
            "avancé", "généralisé",
            "soins de confort",
        ],
    }
    
    if context_value in context_variants:
        context_ok = any(variant in t_lower for variant in context_variants[context_value])
    else:
        context_ok = context_value in t_lower
    
 
    # AGGREGATE FLAGS
    
    flags = {
        # Individual flags
        "nom_ok": bool(nom_ok),
        "prenom_ok": bool(prenom_ok),
        "civilite_ok": bool(civilite_ok),
        "birth_date_ok": bool(birth_date_ok),
        "side_ok": bool(side_ok),
        "subtype_ok": bool(subtype_ok),
        "stage_ok": bool(stage_ok),
        "context_ok": bool(context_ok),
    }
    
    # Core clinical fields
    core_clinical_flags = [nom_ok, side_ok, subtype_ok, stage_ok, context_ok]
    core_clinical_rate = sum(core_clinical_flags) / len(core_clinical_flags)
    
    # All fields (comprehensive completeness)
    all_flags = [nom_ok, prenom_ok, civilite_ok, birth_date_ok, side_ok, subtype_ok, stage_ok, context_ok]
    completeness_rate = sum(all_flags) / len(all_flags)
    
    flags["core_clinical_rate"] = round(float(core_clinical_rate), 3)
    flags["completeness_rate"] = round(float(completeness_rate), 3)
    flags["mention_rate"] = flags["core_clinical_rate"]
    
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

# the idea is : 
    # excellent si completeness_rate ≥ 0.70
    #sinon good si core_clinical_rate ≥ 0.80
    # sinon acceptable si core_clinical_rate ≥ 0.60
    # sinon poor


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
            input_ids = tokenizer.apply_chat_template(
                messages,
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
        description="V4.2: Extended relaxed matching for side, subtype, stage, context"
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

        # Use extended relaxed matching
        mention_flags = check_mentions_v4_extended(text, fields_str)
        
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
                "civilite": fields_str["civilite"],
                "nom": fields_str["nom"],
                "prenom": fields_str["prenom"],
                "birth_date": fields_str["birth_date"],
                "side": fields_str["side"],
                "subtype": fields_str["subtype"],
                "stage": fields_str["stage"],
                "context": fields_str["context"],
            },
            **mention_flags,
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