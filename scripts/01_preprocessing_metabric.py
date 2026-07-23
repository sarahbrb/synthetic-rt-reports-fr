"""
METABRIC — Preprocessing Step 1 & 2
=====================================
Step 1 : Keep only clinically relevant columns
Step 2 : Keep only cases where Radio Therapy = YES

Usage:
    conda activate synthgen
    python scripts/01_preprocessing_metabric.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).parent.parent / "data"
INPUT     = DATA_DIR / "brca_metabric_clinical_data.tsv"
OUTPUT    = DATA_DIR / "metabric_step1_2.csv"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Load
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT, sep="\t", low_memory=False)
df.replace("NA", np.nan, inplace=True)
print(f"[LOAD]  {len(df)} patients  x  {len(df.columns)} colonnes")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Sélection des colonnes pertinentes
# ─────────────────────────────────────────────────────────────────────────────
COLS_KEEP = {
    "Age at Diagnosis"               : "age_diagnostic",
    "Type of Breast Surgery"         : "type_chirurgie",
    "Cancer Type Detailed"           : "type_histologique",
    "Tumor Other Histologic Subtype" : "sous_type_histologique",
    "Neoplasm Histologic Grade"      : "grade_sbr",
    "Pam50 + Claudin-low subtype"    : "sous_type_pam50",
    "ER Status"                      : "statut_er",
    "PR Status"                      : "statut_pr",
    "HER2 Status"                    : "statut_her2",
    "Hormone Therapy"                : "hormonotherapie",
    "Chemotherapy"                   : "chimiotherapie",
    "Radio Therapy"                  : "radiotherapie",
    "Inferred Menopausal State"      : "etat_menopausique",
    "Primary Tumor Laterality"       : "lateralite",
    "Lymph nodes examined positive"  : "ganglions_positifs",
    "Nottingham prognostic index"    : "indice_nottingham",
    "Tumor Size"                     : "taille_tumorale_mm",
    "Tumor Stage"                    : "stade_tumoral",
    "Oncotree Code"                  : "code_oncotree",
}

df = df[list(COLS_KEEP.keys())].rename(columns=COLS_KEEP)
print(f"[STEP 1] Sélection colonnes : {len(df.columns)} colonnes conservées")
for old, new in COLS_KEEP.items():
    print(f"         {old}  →  {new}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Garder uniquement les cas avec radiothérapie = YES
# ─────────────────────────────────────────────────────────────────────────────
n_before = len(df)
df = df[df["radiotherapie"] == "YES"].copy()
n_after  = len(df)
print(f"\n[STEP 2] Filtre radiotherapie = YES")
print(f"         Avant : {n_before} patients")
print(f"         Après : {n_after} patients")
print(f"         Retirés : {n_before - n_after} patients (Radio Therapy = NO ou NA)")

# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT, index=False)
print(f"\n[SAVE]  {OUTPUT}")
print(f"        {len(df)} patients  x  {len(df.columns)} colonnes")

# ─────────────────────────────────────────────────────────────────────────────
# APERÇU
# ─────────────────────────────────────────────────────────────────────────────
print("\n[APERÇU] Premières lignes :")
print(df.head(3).to_string())

print("\n[NA RESTANTS]")
na_summary = df.isna().sum()
na_summary = na_summary[na_summary > 0].sort_values(ascending=False)
for col, n in na_summary.items():
    print(f"  {col}: {n} NA ({100*n/len(df):.1f}%)")
if len(na_summary) == 0:
    print("  Aucun NA ✅")
