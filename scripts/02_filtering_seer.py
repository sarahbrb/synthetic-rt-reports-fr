"""
SEER — Preprocessing Step 2 : Filtrage des cas incohérents
============================================================
Step 1 : Keep only relevant columns + rename to French (already done in step 1)
Step 2 : Remove patients violating clinical coherence rules
         Sources : ONCO AURA 2023 + SENORIF 2021-2022 (AJCC 2017 v8)

Règles EXCLURE :
  A1       : T1 + taille > 20mm (violation TNM)
  A2       : T2 + taille ≤ 20mm ou > 50mm (violation TNM)
  A3       : T3 + taille ≤ 50mm (violation TNM — 7 patients observés)
  B1       : N1 + ganglions positifs hors [1-3] (violation TNM)
  B2       : N2 + ganglions positifs hors [4-9] (violation TNM)
  B5       : ganglions positifs > ganglions examinés (impossible)
  C_MASTER : incohérence T×N → stade global (table AJCC 2017 v8)
  K5       : stade IIA/IIB avec N2 ou N3 (impossible AJCC)
  K4       : stade IIIC sans N3 (impossible AJCC)
  D1       : A Stage Distant + stade IIA-IIIC (contradiction M0/M1)
  E1       : Grade 'anaplastic/Grade IV' (invalide en SBR mammaire)
  E2       : Grade 1 + ER- + PR- (Triple Négatif Grade 1 — biologiquement impossible)

Usage :
    conda activate synthgen
    python scripts/02_filtering_seer.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR  = Path(__file__).parent.parent / "data"
INPUT_RAW = DATA_DIR / "SEER_Breast_Cancer.csv"
INPUT     = DATA_DIR / "seer_step1.csv"
OUTPUT    = DATA_DIR / "seer_step2_filtered.csv"
REPORT    = DATA_DIR / "seer_step2_exclusion_report.csv"

# ─────────────────────────────────────────────────────────────────────────────
# TABLE AJCC 2017 v8 — combinaisons T×N valides (N+ uniquement, M0)
# ─────────────────────────────────────────────────────────────────────────────
VALID_STAGE = {
    ("T1","N1"): "IIA",
    ("T2","N1"): "IIB",
    ("T1","N2"): "IIIA",
    ("T2","N2"): "IIIA",
    ("T3","N1"): "IIIA",
    ("T3","N2"): "IIIA",
    ("T4","N1"): "IIIB",
    ("T4","N2"): "IIIB",
}

RULE_LABELS = {
    "A1":       "T1 + taille > 20mm (violation TNM : T1 = ≤ 20mm)",
    "A2":       "T2 + taille ≤ 20mm ou > 50mm (violation TNM : T2 = 20-50mm)",
    "A3":       "T3 + taille ≤ 50mm (violation TNM : T3 = > 50mm strict)",
    "B1":       "N1 + ganglions positifs hors [1-3] (violation TNM : N1a = 1-3 ganglions)",
    "B2":       "N2 + ganglions positifs hors [4-9] (violation TNM : N2a = 4-9 ganglions)",
    "B5":       "Ganglions positifs > Ganglions examinés (anatomiquement impossible)",
    "C_MASTER": "Incohérence T×N → Stade global (table AJCC 2017 v8)",
    "K5":       "Stade IIA/IIB avec N2 ou N3 (IIA/IIB = N0 ou N1 uniquement)",
    "K4":       "Stade IIIC sans N3 (IIIC = Tout T + N3 par définition)",
    "D1":       "A Stage Distant + Stade IIA-IIIC (Distant = M1 = Stade IV, contradiction M0/M1)",
    "E1":       "Grade 'anaplastic/Grade IV' (n'existe pas dans la classification SBR mammaire)",
    "E2":       "Grade 1 + ER- + PR- (Triple Négatif Grade 1 biologiquement impossible/très rare)",
}

RULE_SOURCE = {
    "A1":       "AJCC 2017 v8, PDF SENORIF p.12",
    "A2":       "AJCC 2017 v8, PDF SENORIF p.12",
    "A3":       "AJCC 2017 v8, PDF SENORIF p.12",
    "B1":       "AJCC 2017 v8, PDF SENORIF p.12",
    "B2":       "AJCC 2017 v8, PDF SENORIF p.12",
    "B5":       "Standard anatomopathologique, PDF SENORIF p.82-83",
    "C_MASTER": "Table AJCC 2017 v8, PDF SENORIF p.13",
    "K5":       "Table AJCC 2017 v8, PDF SENORIF p.13",
    "K4":       "Table AJCC 2017 v8, PDF SENORIF p.13",
    "D1":       "PDF SENORIF p.13 ; PDF ONCO AURA p.47",
    "E1":       "PDF SENORIF p.41 ; PDF ONCO AURA p.27-28",
    "E2":       "PDF SENORIF p.59 (TN = SBR III quasi-exclusivement)",
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Sélection des colonnes pertinentes + renommage
# ─────────────────────────────────────────────────────────────────────────────
COLS_KEEP = {
    "Age":                    "age",
    "T Stage":                "stade_t",
    "N Stage":                "stade_n",
    "6th Stage":              "stade_global",
    "Grade":                  "grade",
    "A Stage":                "extension",
    "Tumor Size":             "taille_tumorale_mm",
    "Estrogen Status":        "statut_er",
    "Progesterone Status":    "statut_pr",
    "Regional Node Examined": "ganglions_examines",
    "Reginol Node Positive":  "ganglions_positifs",
}

df_raw_seer = pd.read_csv(INPUT_RAW)
df_raw_seer.columns = df_raw_seer.columns.str.strip()
df_raw_seer = df_raw_seer.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df = df_raw_seer[list(COLS_KEEP.keys())].rename(columns=COLS_KEEP)
print(f"[STEP 1] Colonnes sélectionnées : {len(df.columns)} (sur {len(df_raw_seer.columns)})")
print(f"         Note : 'differentiate' supprimée (100% redondante avec 'grade')")

# Sauvegarder step 1
df.to_csv(INPUT, index=False)
print(f"         Sauvegardé : {INPUT.name}")

N_init = len(df)
print(f"\n[LOAD]  {N_init} patients  x  {len(df.columns)} colonnes")
print(f"        0 NA (SEER est complet)")

# Colonnes numériques
taille = pd.to_numeric(df["taille_tumorale_mm"], errors="coerce")
gpos   = pd.to_numeric(df["ganglions_positifs"],  errors="coerce")
gexam  = pd.to_numeric(df["ganglions_examines"],  errors="coerce")
grade  = df["grade"].astype(str).str.strip()

# ─────────────────────────────────────────────────────────────────────────────
# CALCUL DES VIOLATIONS AU NIVEAU PATIENT
# ─────────────────────────────────────────────────────────────────────────────
def check_stage(row):
    t_v  = str(row["stade_t"])
    n_v  = str(row["stade_n"])
    sg_v = str(row["stade_global"])
    if n_v == "N3":
        return sg_v != "IIIC"
    if (t_v, n_v) in VALID_STAGE:
        return sg_v != VALID_STAGE[(t_v, n_v)]
    return False

violations = {
    # Groupe A — T Stage × Taille
    "A1": (df["stade_t"] == "T1") & (taille > 20),
    "A2": (df["stade_t"] == "T2") & ((taille <= 20) | (taille > 50)),
    "A3": (df["stade_t"] == "T3") & (taille <= 50),
    # Groupe B — N Stage × Ganglions
    "B1": (df["stade_n"] == "N1") & ((gpos < 1) | (gpos > 3)),
    "B2": (df["stade_n"] == "N2") & ((gpos < 4) | (gpos > 9)),
    "B5": gpos > gexam,
    # Groupe C — Stade global AJCC
    "C_MASTER": df.apply(check_stage, axis=1),
    "K5": df["stade_global"].isin(["IIA","IIB"]) & df["stade_n"].isin(["N2","N3"]),
    "K4": (df["stade_global"] == "IIIC") & ~(df["stade_n"] == "N3"),
    # Groupe D — Extension
    "D1": (df["extension"] == "Distant") & df["stade_global"].isin(["IIA","IIB","IIIA","IIIB","IIIC"]),
    # Groupe E — Grade
    "E1": grade.str.lower().str.contains("anaplastic|iv", na=False),
    "E2": (grade == "1") & (df["statut_er"] == "Negative") & (df["statut_pr"] == "Negative"),
}

# ─────────────────────────────────────────────────────────────────────────────
# UNION DE TOUTES LES VIOLATIONS
# ─────────────────────────────────────────────────────────────────────────────
excl_mask = pd.Series(False, index=df.index)
for rule_id, mask in violations.items():
    excl_mask |= mask.fillna(False)

# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[STEP 2] FILTRAGE DES CAS INCOHÉRENTS")
print("=" * 70)
print(f"\nRègles EXCLURE ({len(violations)} règles — AJCC 2017 v8 + Référentiels) :")

report_rows = []
for rule_id, mask in violations.items():
    n_viol = int(mask.fillna(False).sum())
    pct    = round(100 * n_viol / N_init, 1)
    print(f"  {rule_id:<12} {RULE_LABELS[rule_id]:<60} → {n_viol:4d} ({pct}%)")
    report_rows.append({
        "règle":              rule_id,
        "description":        RULE_LABELS[rule_id],
        "source":             RULE_SOURCE[rule_id],
        "n_patients_violant": n_viol,
        "pct_sur_total":      pct,
    })

n_excl = int(excl_mask.sum())
print(f"\n  → Total patients avec au moins 1 violation : {n_excl} ({100*n_excl/N_init:.1f}%)")
print(f"     (dédoublonnés — un patient peut violer plusieurs règles)")

# ─────────────────────────────────────────────────────────────────────────────
# FILTRAGE
# ─────────────────────────────────────────────────────────────────────────────
df_clean = df[~excl_mask].copy()
df_excl  = df[excl_mask].copy()
N_final  = len(df_clean)
N_removed= len(df_excl)

print(f"\n[FILTRAGE]")
print(f"  Avant : {N_init} patients")
print(f"  Exclus: {N_removed} patients ({100*N_removed/N_init:.1f}%)")
print(f"  Après : {N_final} patients ({100*N_final/N_init:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# AJOUT DES RAISONS D'EXCLUSION DANS LE RAPPORT
# ─────────────────────────────────────────────────────────────────────────────
df_excl = df_excl.copy()
for rule_id, mask in violations.items():
    df_excl[f"excl_{rule_id}"] = mask[excl_mask].fillna(False).values

excl_rule_cols = [f"excl_{rid}" for rid in violations if f"excl_{rid}" in df_excl.columns]
df_excl["règles_violées"] = df_excl[excl_rule_cols].apply(
    lambda row: " | ".join([rid for rid in violations if row.get(f"excl_{rid}", False)]),
    axis=1
)

# ─────────────────────────────────────────────────────────────────────────────
# CONTRÔLE DES DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[CONTRÔLE] Distributions avant/après filtrage :")
check_cols = ["stade_t", "stade_n", "stade_global", "grade", "extension",
              "statut_er", "statut_pr"]
for col in check_cols:
    before = df[col].value_counts(dropna=False).to_dict()
    after  = df_clean[col].value_counts(dropna=False).to_dict()
    print(f"\n  {col}:")
    for val in sorted(set(list(before.keys()) + list(after.keys())), key=str):
        b = before.get(val, 0)
        a = after.get(val, 0)
        diff = b - a
        flag = f"  ← -{diff}" if diff > 0 else ""
        print(f"    {str(val):<30} {b:5d} → {a:5d}{flag}")

# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────
df_clean.to_csv(OUTPUT, index=False)
print(f"\n[SAVE]  Dataset filtré    → {OUTPUT.name}  ({N_final} patients)")

df_excl_report = df_excl[["stade_t","stade_n","stade_global","grade","extension",
                            "statut_er","statut_pr","taille_tumorale_mm",
                            "ganglions_positifs","ganglions_examines","règles_violées"]].copy()
df_excl_report.to_csv(REPORT, index=False)
print(f"        Rapport exclusion  → {REPORT.name}  ({N_removed} patients exclus)")

print(f"\n[RÉSUMÉ FINAL]")
print(f"  Dataset initial (seer_step1)         : {N_init:5d} patients")
print(f"  Patients exclus (12 règles AJCC+Réf) : {N_removed:5d} patients ({100*N_removed/N_init:.1f}%)")
print(f"  Dataset final exploitable            : {N_final:5d} patients ({100*N_final/N_init:.1f}%)")
print(f"\n  Fichier sauvegardé : {OUTPUT}")
