"""
Radiotherapy follow-up report generation (CAG pipeline).

Takes preprocessed patient profiles (METABRIC or SEER), builds a fictitious but age-consistent French identity, maps all clinical fields to French terminology, fills the prompt template, and calls a local LLM to write the end-of-treatment report.

Each output record stores the full generation context alongside the text, so every result is reproducible and auditable after the fact.

In the full campaign, this script is not called directly.
make_jobs.py generates one OAR job script per combination (dataset × prompt_type × sampling_preset × model) with --n 700, and submit_all.sh submits everything to the cluster with automatic evaluation chaining.

For a quick local smoke-test (CPU, small model):
    python scripts/generate_cag.py \\
        --dataset metabric --prompt_type 0shot \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --n 5 --out outputs/smoketest.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from faker import Faker
from transformers import AutoModelForCausalLM, AutoTokenizer

# All paths are relative to the script location so the script works regardless of where it is called from
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data_processed"
PROMPT_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = {
    "metabric": DATA_DIR / "metabric_preprocessed.csv",
    "seer":     DATA_DIR / "seer_preprocessed.csv",
}
PROMPTS = {
    ("metabric", "0shot"): PROMPT_DIR / "user_prompt_metabric_0shot.txt",
    ("metabric", "5shot"): PROMPT_DIR / "user_prompt_metabric_5shot.txt",
    ("seer",     "0shot"): PROMPT_DIR / "user_prompt_seer_0shot.txt",
    ("seer",     "5shot"): PROMPT_DIR / "user_prompt_seer_5shot.txt",
}
SYSTEM_PROMPT_PATH = PROMPT_DIR / "system_prompt.txt"


# Translation table for categorical values that come in English from both datasets and need to appear in French in the prompt
FR = {
    # biomarkers
    "Positive": "positifs",
    "Negative": "négatifs",
    # surgery type
    "MASTECTOMY":        "mastectomie",
    "BREAST CONSERVING": "chirurgie conservatrice",
    # menopausal status
    "Pre":  "pré-ménopausée",
    "Post": "post-ménopausée",
    # laterality
    "Left":  "gauche",
    "Right": "droit",
    # yes/no fields (chemo, hormono)
    "YES": "oui",
    "NO":  "non",
    # PAM50 subtypes
    "LumA":       "Luminal A",
    "LumB":       "Luminal B",
    "Her2":       "HER2-enrichi",
    "Basal":      "Basal-like (Triple Négatif)",
    "claudin-low": "Claudin-low",
    "Normal":     "Normal-like",
    "NC":         "non classifié",
    # histological types
    "Breast Invasive Ductal Carcinoma":          "carcinome canalaire infiltrant",
    "Breast Invasive Lobular Carcinoma":         "carcinome lobulaire infiltrant",
    "Breast Mixed Ductal and Lobular Carcinoma": "carcinome mixte ductal et lobulaire infiltrant",
    "Breast Invasive Mixed Mucinous Carcinoma":  "carcinome mucineux infiltrant",
    "Invasive Breast Carcinoma":                 "carcinome mammaire infiltrant NOS",
    "Breast":                                    "carcinome mammaire NOS",
    "Metaplastic Breast Cancer":                 "carcinome métaplasique",
    "Breast Angiosarcoma":                       "angiosarcome mammaire",
    # histological subtypes
    "Ductal/NST":        "ductal de type non spécifique (NST)",
    "Lobular":           "lobulaire",
    "Mixed":             "mixte",
    "Medullary":         "médullaire",
    "Mucinous":          "mucineux",
    "Tubular/ cribriform": "tubulaire/cribriforme",
    "Other":             "autre",
    "Metaplastic":       "métaplasique",
    # SEER extension
    "Regional": "régionale",
    "Distant":  "distante",
}

# SBR grade and tumour stage are both stored as raw digits ("1", "2", ...) in the CSVs. They can't share the FR table because "1" means something different in each, a collision would silently mistranslate one of them
GRADE_SBR_MAP = {
    "1": "I",
    "2": "II",
    "3": "III",
}
STADE_TUMORAL_MAP = {
    "0": "0",
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
}

# Log untranslated values once per unique value to avoid flooding the console when a field appears in hundreds of rows
_UNTRANSLATED_SEEN: set = set()


def tr(value: Any) -> str:
    """Return the French translation of a categorical value, or the raw value if it is not in the table. Logs a warning the first time an unknown value is seen so the FR table can be updated."""
    if pd.isna(value):
        return "non renseigné"
    s = str(value).strip()
    translated = FR.get(s, s)
    if translated == s and s not in _UNTRANSLATED_SEEN and not s.replace(".", "").replace("-", "").isdigit():
        _UNTRANSLATED_SEEN.add(s)
        print(f"[warn] untranslated value: '{s}' , check the FR table.")
    return translated


def generate_identity(age: float, rng: random.Random, faker_fr: Faker) -> Dict[str, str]:
    """Build a fictitious but internally consistent French identity.

    The birth date is derived from the real patient age so that the age printed in the report matches the source profile exactly, an obvious discrepancy would degrade the clinical realism of the generated text.
    """
    faker_fr.seed_instance(rng.randint(0, 999_999))

    report_year  = rng.randint(2015, 2023)
    report_month = rng.randint(1, 12)
    report_day   = rng.randint(1, 28)   # avoid Feb 29 edge cases
    report_date  = date(report_year, report_month, report_day)

    age_int    = int(round(float(age)))
    birth_year = report_year - age_int
    birth_month = rng.randint(1, 12)
    birth_day   = rng.randint(1, 28)
    birth_date  = date(birth_year, birth_month, birth_day)

    # If the birthday hasn't occurred yet in the report year, the patient was born one year earlier than the naive subtraction gives
    if (birth_month, birth_day) > (report_month, report_day):
        birth_date = date(birth_year - 1, birth_month, birth_day)

    nom    = faker_fr.last_name().upper()
    prenom = faker_fr.first_name_female()

    # Real French CRs often include a birth surname (née ...) for married women
    nom_naissance = ""
    if rng.random() < 0.30:
        nom_naissance = f", née {faker_fr.last_name().upper()}"

    return {
        "civilite":     "Madame",
        "patient_name": f"{nom}{nom_naissance} {prenom}",
        "birth_date":   birth_date.strftime("%d/%m/%Y"),
        "report_date":  report_date.strftime("%d/%m/%Y"),
    }


# ---------------------------------------------------------------------------
# METABRIC context builders

# METABRIC has a significant number of missing fields (the dataset was built for genomic survival analysis, not clinical documentation). The system prompt forbids writing "non renseigné" in the final report text, so we build sentences on the fly and simply skip absent fields rather than inserting them as blanks
# ---------------------------------------------------------------------------

def _histo_fragment(ctx: Dict[str, str]) -> str | None:
    if ctx["type_histologique"] == "non renseigné":
        return None
    if ctx["sous_type_histologique"] != "non renseigné":
        return f"{ctx['type_histologique']} ({ctx['code_oncotree']}, {ctx['sous_type_histologique']})"
    return f"{ctx['type_histologique']} ({ctx['code_oncotree']})"


def build_mentions_metabric(ctx: Dict[str, str]) -> str:
    parts = []
    if ctx["lateralite"] != "non renseigné":
        parts.append(f"sein {ctx['lateralite']}")
    histo = _histo_fragment(ctx)
    if histo:
        parts.append(histo)
    fields = [
        ("grade SBR {grade_sbr}",                      "grade_sbr"),
        ("sous-type moléculaire {sous_type_pam50}",     "sous_type_pam50"),
        ("ER {statut_er}",                              "statut_er"),
        ("PR {statut_pr}",                              "statut_pr"),
        ("HER2 {statut_her2}",                          "statut_her2"),
        ("{taille_tumorale_mm} mm",                     "taille_tumorale_mm"),
        ("stade {stade_tumoral}",                       "stade_tumoral"),
        ("{ganglions_positifs} ganglion(s) positif(s)", "ganglions_positifs"),
        ("indice de Nottingham {indice_nottingham}",    "indice_nottingham"),
        ("{type_chirurgie}",                            "type_chirurgie"),
        ("{etat_menopausique}",                         "etat_menopausique"),
    ]
    for template, key in fields:
        if ctx[key] != "non renseigné":
            parts.append(template.format(**ctx))
    return ", ".join(parts)


def build_context_metabric(row: pd.Series, identity: Dict[str, str]) -> Dict[str, str]:
    """Build the variable dictionary injected into the METABRIC prompt template."""
    ctx: Dict[str, str] = {
        "civilite":             identity["civilite"],
        "patient_name":         identity["patient_name"],
        "birth_date":           identity["birth_date"],
        "age_diagnostic":       str(int(round(float(row["age_diagnostic"])))),
        "lateralite":           tr(row.get("lateralite")),
        "type_histologique":    tr(row.get("type_histologique")),
        "code_oncotree":        str(row["code_oncotree"]).strip()
                                if pd.notna(row.get("code_oncotree")) else "non renseigné",
        "sous_type_histologique": tr(row.get("sous_type_histologique")),
        "grade_sbr":            GRADE_SBR_MAP.get(str(int(float(row["grade_sbr"]))), "non renseigné")
                                if pd.notna(row.get("grade_sbr")) else "non renseigné",
        "sous_type_pam50":      tr(row.get("sous_type_pam50")),
        "statut_er":            tr(row.get("statut_er")),
        "statut_pr":            tr(row.get("statut_pr")),
        "statut_her2":          tr(row.get("statut_her2")),
        "taille_tumorale_mm":   str(int(float(row["taille_tumorale_mm"])))
                                if pd.notna(row.get("taille_tumorale_mm")) else "non renseigné",
        "stade_tumoral":        STADE_TUMORAL_MAP.get(str(int(float(row["stade_tumoral"]))), "non renseigné")
                                if pd.notna(row.get("stade_tumoral")) else "non renseigné",
        "ganglions_positifs":   str(int(float(row["ganglions_positifs"])))
                                if pd.notna(row.get("ganglions_positifs")) else "non renseigné",
        "indice_nottingham":    f"{float(row['indice_nottingham']):.2f}".replace(".", ",")
                                if pd.notna(row.get("indice_nottingham")) else "non renseigné",
        "type_chirurgie":       tr(row.get("type_chirurgie")),
        "etat_menopausique":    tr(row.get("etat_menopausique")),
        "chimiotherapie":       tr(row.get("chimiotherapie")),
        "hormonotherapie":      tr(row.get("hormonotherapie")),
    }
    ctx["mentions_explicites"] = build_mentions_metabric(ctx)
    return ctx


def build_context_seer(row: pd.Series, identity: Dict[str, str]) -> Dict[str, str]:
    """Build the variable dictionary for the SEER prompt template.
    Simpler than METABRIC: SEER has no missing values in the selected columns."""
    return {
        "civilite":           identity["civilite"],
        "patient_name":       identity["patient_name"],
        "birth_date":         identity["birth_date"],
        "age":                str(int(float(row["age"]))),
        "stade_t":            str(row.get("stade_t", "non renseigné")),
        "stade_n":            str(row.get("stade_n", "non renseigné")),
        "stade_global":       str(row.get("stade_global", "non renseigné")),
        "grade":              GRADE_SBR_MAP.get(str(row.get("grade", "")).strip(),
                                              str(row.get("grade", "non renseigné")).strip()),
        "taille_tumorale_mm": str(int(float(row["taille_tumorale_mm"])))
                              if pd.notna(row.get("taille_tumorale_mm")) else "non renseigné",
        "statut_er":          tr(row.get("statut_er")),
        "statut_pr":          tr(row.get("statut_pr")),
        "ganglions_examines": str(int(float(row["ganglions_examines"])))
                              if pd.notna(row.get("ganglions_examines")) else "non renseigné",
        "ganglions_positifs": str(int(float(row["ganglions_positifs"])))
                              if pd.notna(row.get("ganglions_positifs")) else "non renseigné",
    }


def fill_prompt(template: str, ctx: Dict[str, str]) -> str:
    for key, val in ctx.items():
        template = template.replace("{" + key + "}", val)
    return template


def generate(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.5,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 1600,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # add_special_tokens=False: the chat template already inserted the required special tokens (e.g. <s> for Mistral)
    # Letting the tokenizer add them again would create a duplicate at the start of the sequence
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
        )

    # Slice off the prompt echo, only keep the newly generated tokens
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description="CAG — radiotherapy report generation")
    ap.add_argument("--dataset",    required=True, choices=["metabric", "seer"])
    ap.add_argument("--prompt_type", required=True, choices=["0shot", "5shot"])
    ap.add_argument("--model",      required=True, help="HuggingFace model ID")
    ap.add_argument("--out",        required=True, help="output .jsonl file")
    ap.add_argument("--n",     type=int,   default=10,   help="number of reports to generate")
    ap.add_argument("--skip",  type=int,   default=0,
                    help="skip the first N patients after shuffle (used to split long jobs)")
    ap.add_argument("--seed",  type=int,   default=42)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--temperature",        type=float, default=0.5)
    ap.add_argument("--top_p",              type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--max_new_tokens",     type=int,   default=1600)
    args = ap.parse_args()

    rng      = random.Random(args.seed)
    faker_fr = Faker("fr_FR")

    # Fixed-seed shuffle ensures the same patients are selected across runs with the same seed
    # --skip allows splitting one large run into multiple
    # OAR jobs without re-generating already-done patients
    df = pd.read_csv(DATASETS[args.dataset])
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df = df.iloc[args.skip : args.skip + args.n].reset_index(drop=True)
    print(f"Dataset: {args.dataset}, {len(df)} patients (skip={args.skip})")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    user_template = PROMPTS[(args.dataset, args.prompt_type)].read_text(encoding="utf-8")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype_map[args.dtype], device_map="auto"
    )
    model.eval()
    print(f"Model on {next(model.parameters()).device}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i, (_, row) in enumerate(df.iterrows()):
            t0 = time.time()

            age_col  = "age_diagnostic" if args.dataset == "metabric" else "age"
            identity = generate_identity(row[age_col], rng, faker_fr)
            ctx      = build_context_metabric(row, identity) if args.dataset == "metabric" \
                       else build_context_seer(row, identity)
            ctx["report_date"] = identity["report_date"]

            cr_text = generate(
                model, tokenizer, system_prompt, fill_prompt(user_template, ctx),
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            )

            ctx_out = {k: v for k, v in ctx.items() if k != "mentions_explicites"}

            record = {
                "idx":                args.skip + i,
                "dataset":            args.dataset,
                "prompt_type":        args.prompt_type,
                "model":              args.model,
                "temperature":        args.temperature,
                "top_p":              args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "context":            ctx_out,
                "cr_text":            cr_text,
                "duration_s":         round(time.time() - t0, 1),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # flush after each record so a crash mid-run loses nothing

            elapsed = round(time.time() - t0, 1)
            print(f"  [{i+1}/{len(df)}] {identity['patient_name']}  {elapsed}s")

    print(f"\nDone. {out_path}  ({len(df)} reports)")


if __name__ == "__main__":
    main()
