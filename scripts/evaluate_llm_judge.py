"""
LLM-as-a-Judge avec modèle local (Qwen/Mistral/Llama) pour évaluer des CR de fin de RT sein.

Sortie: JSON (agrégé + individuel)
Critères (score 1-5):
1) Complétude clinique
2) Exactitude médicale
3) Conformité au conditionnement
4) Cohérence factuelle interne
5) Qualité linguistique & fluence
6) Réalisme global

Score total: /30
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



# PROMPTS


SYSTEM_PROMPT = """Tu es un expert en oncologie et radiothérapie, spécialisé dans l'évaluation de documents médicaux.
Ta tâche est d'évaluer des comptes rendus de fin de traitement de radiothérapie pour cancer du sein.
Tu dois être rigoureux, objectif et fournir des scores justifiés.
IMPORTANT: Réponds UNIQUEMENT avec un JSON valide, sans aucun texte avant ou après le JSON.
"""

USER_PROMPT_TEMPLATE = """Évalue le compte rendu de radiothérapie ci-dessous.

## COMPTE RENDU À ÉVALUER :
---
{cr_text}
---

Le compte rendu devait inclure ces éléments obligatoires :
- Nom du patient : {nom}
- Latéralité : {side}
- Sous-type moléculaire : {subtype}
- Stade TNM : {stage}
- Contexte thérapeutique : {context}

## CRITÈRES D'ÉVALUATION (note chaque critère de 1 à 5) :

# il faudra que je rajoute signification de 1 et 5 

1. Complétude clinique: Le compte rendu inclut-il l’ensemble des informations cliniques attendues (identité, contexte, latéralité, sous-type, stade, etc.) ?
2. Exactitude médicale: Les informations médicales sont-elles correctes, cohérentes entre elles et conformes aux connaissances cliniques actuelles ?
3. Conformité au conditionnement: Le texte respecte-t-il strictement les contraintes imposées (champs conditionnés, formulations attendues, absence d’éléments interdits) ?
4. Cohérence factuelle interne: Le compte rendu est-il exempt de contradictions internes (dates, stades, traitements, enchaînement logique des faits) ?
5. Qualité linguistique et fluence: Le texte est-il grammaticalement correct, fluide, clair, et rédigé dans un français médical approprié ?
6. Réalisme clinique global: Le document est-il crédible comme un véritable compte rendu médical rédigé par un professionnel de santé ?

## RÉPONDS UNIQUEMENT AVEC CE JSON (rien d'autre) :

{{
  "completude_clinique": {{"score": <1-5>, "justification": "<1 phrase>"}},
  "exactitude_medicale": {{"score": <1-5>, "justification": "<1 phrase>"}},
  "conformite_conditionnement": {{"score": <1-5>, "justification": "<1 phrase>"}},
  "coherence_factuelle": {{"score": <1-5>, "justification": "<1 phrase>"}},
  "qualite_linguistique": {{"score": <1-5>, "justification": "<1 phrase>"}},
  "realisme_global": {{"score": <1-5>, "justification": "<1 phrase>"}},
  "score_total": <somme des 6 scores>
}}
"""



# MODEL


def load_model(model_name: str):
    print(f"[...] Chargement du modèle: {model_name}")
    print("      (Cela peut prendre quelques minutes la première fois)")

    # use_fast parfois casse selon versions/tokenizer.json -> si besoin mets use_fast=False
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
        print(f"[OK] GPU détecté: {torch.cuda.get_device_name(0)}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        print("[WARN] Pas de GPU, utilisation du CPU (lent)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model.to("cpu")

    model.eval()
    print("[OK] Modèle chargé")
    return tokenizer, model



# FIELDS


def extract_fields(record: Dict[str, Any]) -> Dict[str, str]:
    """
    Extrait les champs de conditionnement.
    Cas attendus:
      - dict: {"nom":..., "side":..., ...}
      - list: [patient_name(or nom), side, subtype, stage, context]
    """
    fields = record.get("fields", {})

    if isinstance(fields, dict):
        return {
            "nom": fields.get("nom", "N/A"),
            "side": fields.get("side", "N/A"),
            "subtype": fields.get("subtype", "N/A"),
            "stage": fields.get("stage", "N/A"),
            "context": fields.get("context", "N/A"),
        }

    if isinstance(fields, list):
        # hypothèse la plus fréquente: [name, side, subtype, stage, context]
        return {
            "nom": fields[0] if len(fields) > 0 else "N/A",
            "side": fields[1] if len(fields) > 1 else "N/A",
            "subtype": fields[2] if len(fields) > 2 else "N/A",
            "stage": fields[3] if len(fields) > 3 else "N/A",
            "context": fields[4] if len(fields) > 4 else "N/A",
        }

    return {k: "N/A" for k in ["nom", "side", "subtype", "stage", "context"]}



# PARSE


def _extract_json_block(s: str) -> Optional[str]:
    s = s.strip()
    # enlève éventuels fences ```json ... ```
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # essaie de prendre le premier {...}
    m = re.search(r"\{[\s\S]*\}", s)
    return m.group(0) if m else None


def parse_response(response: str) -> Dict[str, Any]:
    response = response.strip()
    json_str = _extract_json_block(response)

    required = [
        "completude_clinique",
        "exactitude_medicale",
        "conformite_conditionnement",
        "coherence_factuelle",
        "qualite_linguistique",
        "realisme_global",
    ]

    if json_str:
        try:
            # évite les retours ligne qui cassent parfois les parseurs
            json_str = re.sub(r"(?<!\\)\n", " ", json_str)
            parsed = json.loads(json_str)

            if all(k in parsed for k in required):
                if "score_total" not in parsed:
                    parsed["score_total"] = sum(
                        int(parsed[k].get("score", 0)) for k in required if isinstance(parsed[k], dict)
                    )
                return parsed
        except Exception:
            pass

    # fallback regex si JSON non parseable
    scores = {}
    for criterion in required:
        m = re.search(rf'"{criterion}"[^}}]*"score"\s*:\s*([1-5])', response)
        if m:
            scores[criterion] = {"score": int(m.group(1)), "justification": "Extrait par regex"}

    if len(scores) == len(required):
        scores["score_total"] = sum(v["score"] for v in scores.values())
        return scores

    return {
        "error": "Échec du parsing",
        "raw_response": response[:600],
        **{k: {"score": 0, "justification": "Erreur"} for k in required},
        "score_total": 0,
    }



# EVAL


@torch.inference_mode()
def evaluate_single(
    tokenizer,
    model,
    cr_text: str,
    fields: Dict[str, str],
    max_new_tokens: int = 700,
) -> Dict[str, Any]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        cr_text=(cr_text or "")[:3000],
        nom=fields.get("nom", "N/A"),
        side=fields.get("side", "N/A"),
        subtype=fields.get("subtype", "N/A"),
        stage=fields.get("stage", "N/A"),
        context=fields.get("context", "N/A"),
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        flat = f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]\n"
        input_ids = tokenizer(flat, return_tensors="pt").input_ids.to(model.device)

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
    return parse_response(response)



# AGGREGATION


def aggregate_results(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [e for e in evaluations if e.get("score_total", 0) > 0]
    if not valid:
        return {"error": "Aucune évaluation valide", "num_errors": len(evaluations)}

    criteria = [
        "completude_clinique",
        "exactitude_medicale",
        "conformite_conditionnement",
        "coherence_factuelle",
        "qualite_linguistique",
        "realisme_global",
    ]

    scores = {c: [] for c in criteria}
    scores["score_total"] = []

    for e in valid:
        for c in criteria:
            if c in e and isinstance(e[c], dict):
                scores[c].append(int(e[c].get("score", 0)))
        scores["score_total"].append(int(e.get("score_total", 0)))

    stats = {}
    for key, values in scores.items():
        if values:
            stats[key] = {
                "mean": round(float(np.mean(values)), 2),
                "std": round(float(np.std(values)), 2),
                "min": int(np.min(values)),
                "max": int(np.max(values)),
            }

    if scores["score_total"]:
        stats["score_global_100"] = round(float(np.mean(scores["score_total"])) / 30 * 100, 1)

    return {
        "statistics": stats,
        "num_evaluated": len(valid),
        "num_errors": len(evaluations) - len(valid),
    }



# MAIN


def main():
    parser = argparse.ArgumentParser(description="Évaluation LLM-as-a-Judge avec modèle local")
    parser.add_argument("--input", required=True, help="Fichier JSONL d'entrée")
    parser.add_argument("--output", required=True, help="Fichier JSON de sortie")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Modèle HuggingFace")
    parser.add_argument("--max_samples", type=int, default=None, help="Nombre max de CR à évaluer")
    parser.add_argument("--text_field", default="cr_text", help="Champ texte (défaut: cr_text)")
    parser.add_argument("--max_new_tokens", type=int, default=700, help="Budget tokens de la réponse juge")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)

    input_path = Path(args.input)
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.max_samples is not None:
        records = records[: args.max_samples]

    print(f"\n[OK] {len(records)} CR à évaluer\n")

    evaluations: List[Dict[str, Any]] = []

    for i, record in enumerate(records):
        cr_text = record.get(args.text_field, "")
        fields = extract_fields(record)
        seed = record.get("seed", i)

        print(f"[{i+1:2d}/{len(records)}] Seed {seed} | {fields['nom'][:12]:12s} | ", end="", flush=True)

        t0 = time.time()
        result = evaluate_single(tokenizer, model, cr_text, fields, max_new_tokens=args.max_new_tokens)
        dt = time.time() - t0

        result["seed"] = seed
        result["eval_time"] = round(dt, 1)
        evaluations.append(result)

        score = result.get("score_total", 0)
        status = "✓" if "error" not in result else "✗"
        print(f"Score: {score:>2}/30 | {dt:.1f}s {status}")

    aggregated = aggregate_results(evaluations)

    results = {
        "metadata": {
            "input_file": str(input_path),
            "model": args.model,
            "num_samples": len(records),
        },
        "aggregated": aggregated,
        "individual": evaluations,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Résultats sauvegardés: {output_path}\n")
    print(json.dumps(aggregated, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
