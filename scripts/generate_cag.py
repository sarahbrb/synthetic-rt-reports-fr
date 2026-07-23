"""
script de génération des comptes rendus de radiothérapie (projet CAG).

l'idée : on prend des patients réels (metabric ou seer, déjà filtrés et
nettoyés par les scripts précédents), on leur fabrique une fausse identité,
on traduit leur profil clinique en français, et on demande à un LLM de
rédiger le compte rendu de fin de traitement à partir de ce profil.

usage (avec l'env conda synthgen activé) :

    python scripts/generate_cag.py \
        --dataset metabric \
        --prompt_type 0shot \
        --model Qwen/Qwen2.5-7B-Instruct \
        --n 10 \
        --out outputs/metabric_0shot_qwen7b.jsonl

    python scripts/generate_cag.py \
        --dataset seer \
        --prompt_type 5shot \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --n 10 \
        --out outputs/seer_5shot_mistral7b.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from faker import Faker
from transformers import AutoModelForCausalLM, AutoTokenizer

# chemins du projet, tout est relatif à l'emplacement du script pour que
# ça marche peu importe d'où on lance la commande
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROMPT_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = {
    "metabric": DATA_DIR / "metabric_step3_filtered.csv",
    "seer": DATA_DIR / "seer_step2_filtered.csv",
}
PROMPTS = {
    ("metabric", "0shot"): PROMPT_DIR / "user_prompt_metabric_0shot.txt",
    ("metabric", "5shot"): PROMPT_DIR / "user_prompt_metabric_5shot.txt",
    ("seer", "0shot"): PROMPT_DIR / "user_prompt_seer_0shot.txt",
    ("seer", "5shot"): PROMPT_DIR / "user_prompt_seer_5shot.txt",
}
SYSTEM_PROMPT_PATH = PROMPT_DIR / "system_prompt.txt"


# dictionnaire de traduction anglais -> français pour les valeurs
# catégorielles des deux datasets (qui sont en anglais à la base)
FR = {
    # biomarqueurs
    "Positive": "positifs",
    "Negative": "négatifs",
    # chirurgie
    "MASTECTOMY": "mastectomie",
    "BREAST CONSERVING": "chirurgie conservatrice",
    # ménopause
    "Pre": "pré-ménopausée",
    "Post": "post-ménopausée",
    # latéralité
    "Left": "gauche",
    "Right": "droit",
    # chimio / hormono, champs oui/non bruts
    "YES": "oui",
    "NO": "non",
    # sous-types pam50
    "LumA": "Luminal A",
    "LumB": "Luminal B",
    "Her2": "HER2-enrichi",
    "Basal": "Basal-like (Triple Négatif)",
    "claudin-low": "Claudin-low",
    "Normal": "Normal-like",
    "NC": "non classifié",
    # types histologiques
    "Breast Invasive Ductal Carcinoma": "carcinome canalaire infiltrant",
    "Breast Invasive Lobular Carcinoma": "carcinome lobulaire infiltrant",
    "Breast Mixed Ductal and Lobular Carcinoma": "carcinome mixte ductal et lobulaire infiltrant",
    "Breast Invasive Mixed Mucinous Carcinoma": "carcinome mucineux infiltrant",
    "Invasive Breast Carcinoma": "carcinome mammaire infiltrant NOS",
    "Breast": "carcinome mammaire NOS",
    "Metaplastic Breast Cancer": "carcinome métaplasique",
    "Breast Angiosarcoma": "angiosarcome mammaire",
    # sous-types histologiques
    "Ductal/NST": "ductal de type non spécifique (NST)",
    "Lobular": "lobulaire",
    "Mixed": "mixte",
    "Medullary": "médullaire",
    "Mucinous": "mucineux",
    "Tubular/ cribriform": "tubulaire/cribriforme",
    "Other": "autre",
    "Metaplastic": "métaplasique",
    # extension seer
    "Regional": "régionale",
    "Distant": "distante",
}

# le grade sbr et le stade tumoral sont tous les deux des chiffres bruts
# dans les csv ("1", "2", "3"...), donc impossible de les mettre dans le
# même dico que FR : le "1" du grade et le "1" du stade ne veulent pas dire
# la même chose une fois traduits, ça collisionnerait.
GRADE_SBR_MAP = {
    "1": "I (bien différencié)",
    "2": "II (modérément différencié)",
    "3": "III (peu différencié)",
}
STADE_TUMORAL_MAP = {
    "0": "0",
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
}

# pour repérer une valeur catégorielle qui nous échappe (ex: nouvelle
# catégorie apparue après une mise à jour du dataset). on log une seule
# fois par valeur pour ne pas spammer la console à chaque ligne.
_UNTRANSLATED_SEEN: set = set()


def tr(value: Any) -> str:
    """traduit une valeur en français si on la connait, sinon on la laisse telle quelle."""
    if pd.isna(value):
        return "non renseigné"
    s = str(value).strip()
    translated = FR.get(s, s)
    if translated == s and s not in _UNTRANSLATED_SEEN and not s.replace(".", "").replace("-", "").isdigit():
        _UNTRANSLATED_SEEN.add(s)
        print(f"[WARN] valeur non traduite (FR) : '{s}' -- vérifier le dictionnaire FR.")
    return translated


def generate_identity(age: float, rng: random.Random, faker_fr: Faker) -> Dict[str, str]:
    """
    fabrique une identité bidon mais cohérente avec l'âge réel du patient.
    il faut que la date de naissance calculée redonne bien le même âge que
    celui du dossier source, sinon ça se voit tout de suite à la lecture.
    """
    faker_fr.seed_instance(rng.randint(0, 999_999))

    # date du courrier : on tire au hasard quelque part entre 2015 et 2023
    report_year = rng.randint(2015, 2023)
    report_month = rng.randint(1, 12)
    report_day = rng.randint(1, 28)  # 28 pour éviter de se prendre la tête avec février
    report_date = date(report_year, report_month, report_day)

    # date de naissance déduite de l'âge au diagnostic
    age_int = int(round(float(age)))
    birth_year = report_year - age_int
    birth_month = rng.randint(1, 12)
    birth_day = rng.randint(1, 28)
    birth_date = date(birth_year, birth_month, birth_day)

    # si l'anniversaire n'est pas encore passé à la date du courrier,
    # ça veut dire qu'elle est née un an plus tôt que ce qu'on vient de calculer
    if (birth_month, birth_day) > (report_month, report_day):
        birth_date = date(birth_year - 1, birth_month, birth_day)

    prenom = faker_fr.first_name_female()
    nom = faker_fr.last_name().upper()

    # pas mal de patientes mariées gardent leur nom de naissance sur les
    # vrais CR, donc on simule ça avec une petite proba
    nom_naissance = ""
    if rng.random() < 0.30:
        nom_naissance = f", née {faker_fr.last_name().upper()}"

    return {
        "civilite": "Madame",
        "patient_name": f"{nom}{nom_naissance} {prenom}",
        "birth_date": birth_date.strftime("%d/%m/%Y"),
        "report_date": report_date.strftime("%d/%m/%Y"),
    }


# tout ce qui suit ne concerne que metabric. contrairement à seer, metabric
# a pas mal de champs qui peuvent être manquants, et le system prompt
# interdit formellement d'écrire un truc comme "stade non renseigné" dans
# le texte final. donc on construit les phrases à la volée en zappant
# simplement les champs absents plutôt que de les insérer tels quels.

def _histo_fragment(ctx: Dict[str, str]) -> str | None:
    """renvoie le type histo (+ sous-type si on l'a), ou rien si on n'a aucune info histo."""
    if ctx["type_histologique"] == "non renseigné":
        return None
    if ctx["sous_type_histologique"] != "non renseigné":
        return f"{ctx['type_histologique']} ({ctx['code_oncotree']}, {ctx['sous_type_histologique']})"
    return f"{ctx['type_histologique']} ({ctx['code_oncotree']})"


def build_diagnostic_text_metabric(ctx: Dict[str, str]) -> str:
    """la phrase de diagnostic, en ne gardant que ce qui est effectivement renseigné."""
    parts = []
    histo = _histo_fragment(ctx)
    if histo:
        parts.append(histo)

    # (gabarit, clé) : le gabarit n'est ajouté que si la clé correspondante est renseignée
    simple = [
        ("sein {lateralite}", "lateralite"),
        ("grade SBR {grade_sbr}", "grade_sbr"),
        ("sous-type moléculaire {sous_type_pam50}", "sous_type_pam50"),
        ("récepteurs ER {statut_er}", "statut_er"),
        ("PR {statut_pr}", "statut_pr"),
        ("HER2 {statut_her2}", "statut_her2"),
        ("taille tumorale {taille_tumorale_mm} mm", "taille_tumorale_mm"),
        ("stade {stade_tumoral}", "stade_tumoral"),
        ("{ganglions_positifs} ganglion(s) positif(s)", "ganglions_positifs"),
        ("indice pronostique de Nottingham {indice_nottingham}", "indice_nottingham"),
    ]
    for template, key in simple:
        if ctx[key] != "non renseigné":
            parts.append(template.format(**ctx))
    return ", ".join(parts)


def build_mentions_metabric(ctx: Dict[str, str]) -> str:
    """la liste de tout ce que le prompt va explicitement demander de citer dans le texte."""
    parts = []
    if ctx["lateralite"] != "non renseigné":
        parts.append(f"sein {ctx['lateralite']}")
    histo = _histo_fragment(ctx)
    if histo:
        parts.append(histo)

    simple = [
        ("grade SBR {grade_sbr}", "grade_sbr"),
        ("sous-type moléculaire {sous_type_pam50}", "sous_type_pam50"),
        ("ER {statut_er}", "statut_er"),
        ("PR {statut_pr}", "statut_pr"),
        ("HER2 {statut_her2}", "statut_her2"),
        ("{taille_tumorale_mm} mm", "taille_tumorale_mm"),
        ("stade {stade_tumoral}", "stade_tumoral"),
        ("{ganglions_positifs} ganglion(s) positif(s)", "ganglions_positifs"),
        ("indice de Nottingham {indice_nottingham}", "indice_nottingham"),
        ("{type_chirurgie}", "type_chirurgie"),
        ("{etat_menopausique}", "etat_menopausique"),
    ]
    for template, key in simple:
        if ctx[key] != "non renseigné":
            parts.append(template.format(**ctx))
    return ", ".join(parts)


def build_contexte_metabric(ctx: Dict[str, str]) -> str:
    """la phrase de contexte thérapeutique. la chirurgie est zappée si son type est inconnu."""
    parts = []
    if ctx["type_chirurgie"] != "non renseigné":
        parts.append(f"chirurgie ({ctx['type_chirurgie']})")
    parts.append(f"état ménopausique ({ctx['etat_menopausique']})")
    parts.append("traitements systémiques adjuvants selon le profil patient")
    parts.append("décision de réunion de concertation pluridisciplinaire")
    return ", ".join(parts)


def build_context_metabric(row: pd.Series, identity: Dict[str, str]) -> Dict[str, str]:
    """construit le dictionnaire de variables qui sera injecté dans le prompt metabric."""
    ctx = {
        "civilite": identity["civilite"],
        "patient_name": identity["patient_name"],
        "birth_date": identity["birth_date"],
        "age_diagnostic": str(int(round(float(row["age_diagnostic"])))),
        "lateralite": tr(row.get("lateralite")),
        "type_histologique": tr(row.get("type_histologique")),
        "code_oncotree": str(row["code_oncotree"]).strip()
        if pd.notna(row.get("code_oncotree")) else "non renseigné",
        "sous_type_histologique": tr(row.get("sous_type_histologique")),
        "grade_sbr": GRADE_SBR_MAP.get(str(int(float(row["grade_sbr"]))), "non renseigné")
        if pd.notna(row.get("grade_sbr")) else "non renseigné",
        "sous_type_pam50": tr(row.get("sous_type_pam50")),
        "statut_er": tr(row.get("statut_er")),
        "statut_pr": tr(row.get("statut_pr")),
        "statut_her2": tr(row.get("statut_her2")),
        "taille_tumorale_mm": str(int(float(row["taille_tumorale_mm"])))
        if pd.notna(row.get("taille_tumorale_mm")) else "non renseigné",
        "stade_tumoral": STADE_TUMORAL_MAP.get(str(int(float(row["stade_tumoral"]))), "non renseigné")
        if pd.notna(row.get("stade_tumoral")) else "non renseigné",
        "ganglions_positifs": str(int(float(row["ganglions_positifs"])))
        if pd.notna(row.get("ganglions_positifs")) else "non renseigné",
        "indice_nottingham": f"{float(row['indice_nottingham']):.2f}".replace(".", ",")
        if pd.notna(row.get("indice_nottingham")) else "non renseigné",
        "type_chirurgie": tr(row.get("type_chirurgie")),
        "etat_menopausique": tr(row.get("etat_menopausique")),
        "chimiotherapie": tr(row.get("chimiotherapie")),
        "hormonotherapie": tr(row.get("hormonotherapie")),
    }
    # les phrases composées ne peuvent être construites qu'une fois le reste du contexte prêt
    ctx["diagnostic_text"] = build_diagnostic_text_metabric(ctx)
    ctx["contexte_text"] = build_contexte_metabric(ctx)
    ctx["mentions_explicites"] = build_mentions_metabric(ctx)
    return ctx


def build_context_seer(row: pd.Series, identity: Dict[str, str]) -> Dict[str, str]:
    """pareil que pour metabric, mais plus simple : seer n'a aucune valeur manquante."""
    grade_raw = str(row.get("grade", "")).strip()
    return {
        "civilite": identity["civilite"],
        "patient_name": identity["patient_name"],
        "birth_date": identity["birth_date"],
        "age": str(int(float(row["age"]))),
        "stade_t": str(row.get("stade_t", "non renseigné")),
        "stade_n": str(row.get("stade_n", "non renseigné")),
        "stade_global": str(row.get("stade_global", "non renseigné")),
        "grade": tr(grade_raw),
        "taille_tumorale_mm": str(int(float(row["taille_tumorale_mm"])))
        if pd.notna(row.get("taille_tumorale_mm")) else "non renseigné",
        "statut_er": tr(row.get("statut_er")),
        "statut_pr": tr(row.get("statut_pr")),
        "ganglions_examines": str(int(float(row["ganglions_examines"])))
        if pd.notna(row.get("ganglions_examines")) else "non renseigné",
        "ganglions_positifs": str(int(float(row["ganglions_positifs"])))
        if pd.notna(row.get("ganglions_positifs")) else "non renseigné",
    }


def load_prompt_template(dataset: str, prompt_type: str) -> str:
    path = PROMPTS[(dataset, prompt_type)]
    return path.read_text(encoding="utf-8")


def fill_prompt(template: str, ctx: Dict[str, str]) -> str:
    # remplacement basique, pas besoin de jinja pour quelques placeholders
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
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # add_special_tokens=False : le chat template a déjà inséré les tokens
    # spéciaux nécessaires (par ex <s> pour mistral). si on laisse le
    # tokenizer faire son travail normalement, il va en remettre un par
    # dessus, et on se retrouve avec un doublon en début de séquence.
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

    # on ne garde que les tokens générés en plus, pas le prompt renvoyé en écho
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description="CAG -- génération de CR de radiothérapie")
    ap.add_argument("--dataset", required=True, choices=["metabric", "seer"])
    ap.add_argument("--prompt_type", required=True, choices=["0shot", "5shot"])
    ap.add_argument("--model", required=True, help="nom huggingface du modèle")
    ap.add_argument("--out", required=True, help="fichier jsonl de sortie")
    ap.add_argument("--n", type=int, default=10, help="nombre de CR à générer")
    ap.add_argument("--skip", type=int, default=0, help="sauter les N premiers patients après le shuffle (pour splitter un run en plusieurs jobs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--max_new_tokens", type=int, default=1600)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    faker_fr = Faker("fr_FR")

    # on charge le dataset déjà filtré, on mélange avec un seed fixe (pour
    # que relancer avec le même seed donne les mêmes patients), puis on
    # garde seulement les n premiers de ce mélange
    df = pd.read_csv(DATASETS[args.dataset])
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df = df.iloc[args.skip : args.skip + args.n].reset_index(drop=True)
    print(f"[DATA]  {len(df)} patients -- dataset {args.dataset} (skip={args.skip})")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    user_template = load_prompt_template(args.dataset, args.prompt_type)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    print(f"[MODEL] chargement de {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")
    model.eval()
    print(f"[MODEL] chargé sur {next(model.parameters()).device}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i, (_, row) in enumerate(df.iterrows()):
            t0 = time.time()

            age_col = "age_diagnostic" if args.dataset == "metabric" else "age"
            identity = generate_identity(row[age_col], rng, faker_fr)

            if args.dataset == "metabric":
                ctx = build_context_metabric(row, identity)
            else:
                ctx = build_context_seer(row, identity)

            user_prompt = fill_prompt(user_template, ctx)

            cr_text = generate(
                model, tokenizer, system_prompt, user_prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            )

            # on garde une trace complète (identité + contexte clinique utilisé) pour
            # pouvoir retracer exactement ce qui a produit chaque texte généré
            record = {
                "idx": args.skip + i,
                "dataset": args.dataset,
                "prompt_type": args.prompt_type,
                "model": args.model,
                # nécessaire pour pouvoir comparer plusieurs configs de sampling au
                # moment de l'évaluation -- sans ça, des CR générés avec des réglages
                # différents pour le même modèle/dataset/prompt_type se retrouveraient
                # mélangés dans le même groupe lors de l'agrégation
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "identity": identity,
                "context": ctx,
                "cr_text": cr_text,
                "duration_s": round(time.time() - t0, 1),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # au cas où ça plante en cours de route, on ne perd pas ce qui est déjà généré

            print(f"  [{i+1}/{len(df)}] {identity['patient_name']} -- {round(time.time()-t0,1)}s")

    print(f"\n[DONE]  {out_path}  ({len(df)} CR générés)")


if __name__ == "__main__":
    main()
