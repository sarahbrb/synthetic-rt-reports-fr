"""
METABRIC preprocessing pipeline for the CAG campaign.

Produces the patient profiles used by generate_cag.py from the raw cBioPortal clinical file.
Three steps applied in sequence:
  1. Select and rename clinically relevant columns
  2. Restrict to radiotherapy-treated patients (Radio Therapy == YES)
  3. Exclude clinically incoherent profiles

The coherence rules in step 3 were defined with a practising radiation oncologist against SENORIF 2021-2022 and ONCO AURA 2023 guidelines.
Each rule is traceable to its row in:
    Analyse datasets/analyse_incoherences (metabric).xlsx

Expected result (validated against the xlsx summary sheet):
    1173 patients after RT filter  →  702 retained after exclusions (59.8%)

Usage:
    python scripts/preprocess_metabric.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW      = DATA_DIR / "brca_metabric_clinical_data.tsv"
OUT      = DATA_DIR / "metabric_preprocessed.csv"
EXCL_LOG = DATA_DIR / "metabric_exclusion_report.csv"


# ---------------------------------------------------------------------------
# Step 1 : Column selection
# ---------------------------------------------------------------------------

df = pd.read_csv(RAW, sep="\t", low_memory=False)
df.replace("NA", np.nan, inplace=True)
print(f"Loaded: {len(df)} patients, {len(df.columns)} columns")

# Only the columns needed for generation; renamed to match the French clinical terminology used in the prompt templates.
cols = {
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
df = df[list(cols.keys())].rename(columns=cols)


# ---------------------------------------------------------------------------
# Step 2 : Radiotherapy filter
# ---------------------------------------------------------------------------

n_raw = len(df)
df = df[df["radiotherapie"] == "YES"].copy()
print(f"After RT filter: {len(df)} patients (removed {n_raw - len(df)})")

# Constant after the filter, no longer informative
df.drop(columns=["radiotherapie"], inplace=True)


# ---------------------------------------------------------------------------
# Step 3 : Coherence rule filtering
# ---------------------------------------------------------------------------

# Source: Analyse datasets/analyse_incoherences (metabric).xlsx
# Severity levels: Absolute (biologically impossible) / Strong (clinically inconsistent, high rate) / Atypical (rare but documented).

pam   = df["sous_type_pam50"].astype(str)
er    = df["statut_er"].astype(str)
pr    = df["statut_pr"].astype(str)
her2  = df["statut_her2"].astype(str)
horm  = df["hormonotherapie"].astype(str)
chim  = df["chimiotherapie"].astype(str)
chir  = df["type_chirurgie"].astype(str)
grade = pd.to_numeric(df["grade_sbr"], errors="coerce")
stade = pd.to_numeric(df["stade_tumoral"], errors="coerce")
taille= pd.to_numeric(df["taille_tumorale_mm"], errors="coerce")
gangl = pd.to_numeric(df["ganglions_positifs"], errors="coerce")
npi   = pd.to_numeric(df["indice_nottingham"], errors="coerce")
type_h = df["type_histologique"].astype(str)
sub_h  = df["sous_type_histologique"].fillna("").astype(str)

rh_neg = (er == "Negative") & (pr == "Negative")   # no hormonal target
tn     = rh_neg & (her2 == "Negative")              # triple negative

violations = {}


# Group R : Biomarkers / Hormone therapy
# --------------------------------------

# R5/R12 [Absolute] : hormone therapy with no hormonal target (ER- AND PR-)
violations["R5/R12"] = rh_neg & (horm == "YES")

# R7 [Absolute] : triple-negative profile (ER-, PR-, HER2-) with hormone therapy
violations["R7"]  = tn & (horm == "YES")

# R8 [Absolute] : Basal-like with ER- and hormone therapy (functionally TN)
violations["R8"]  = (pam == "Basal") & (er == "Negative") & (horm == "YES")

# R11 [Absolute] : Claudin-low with ER- and hormone therapy
violations["R11"] = (pam == "claudin-low") & (er == "Negative") & (horm == "YES")

# R14 [Absolute] : HER2+ with RH- (ER- AND PR-) and hormone therapy
violations["R14"] = (her2 == "Positive") & rh_neg & (horm == "YES")

# R1 [Strong] : LumA with ER-: contradicts LumA definition (ER+PR+ by definition)
violations["R1"]  = (pam == "LumA") & (er == "Negative")

# R2 [Strong] : LumB with ER-: same issue (LumB also implies ER+ by definition)
violations["R2"]  = (pam == "LumB") & (er == "Negative")

# R4 [Strong] : HER2-enriched (PAM50) with HER2- by IHC
violations["R4"]  = (pam == "Her2") & (her2 == "Negative")

# R6 [Strong] : ER+ without hormone therapy
violations["R6"]  = (er == "Positive") & (horm == "NO")


# Group C : Chemotherapy
# ----------------------

# C1 [Absolute] : DCIS (stage 0) with chemotherapy
#   0 patients in METABRIC, expert confirmed rule.
violations["C1"]  = (stade == 0) & (chim == "YES")

# C3 [Strong] : HER2-enriched (PAM50) without chemotherapy, tumor > 5mm
violations["C3"]  = (pam == "Her2") & (chim == "NO") & ((taille > 5) | taille.isna())


# Group G : Histological grade
# ----------------------------

# G1 [Absolute] : Basal-like with Grade 1: biologically impossible
violations["G1"]  = (pam == "Basal") & (grade == 1)

# G3 [Strong] : LumA with Grade 3
violations["G3"]  = (pam == "LumA") & (grade == 3)

# G5 [Atypical] : Mucinous carcinoma with Grade 3
violations["G5"]  = type_h.str.contains("Mucinous", na=False) & (grade == 3)


# Group SG : Stage / Lymph nodes
# --------------------------------

# SG1 [Absolute] : Stage 1 with any positive lymph node
violations["SG1"] = (stade == 1) & (gangl > 0)

# SG4 [Absolute] : Stage < 3 with ≥ 10 positive nodes
violations["SG4"] = (stade < 3) & (gangl >= 10)


# Group ST : Stage / Lymph nodes (additional)
# --------------------------------------------

# ST1 [Strong] : Stage 1 with more than 3 positive nodes
violations["ST1"] = (stade == 1) & (gangl > 3)


# Group NPI : Nottingham Prognostic Index
# ----------------------------------------

# NPI2 [Strong] : NPI ≥ 6.0 with Grade 1: mathematically impossible
#   0 patients in METABRIC
violations["NPI2"] = (npi >= 6.0) & (grade == 1)


# Group S : Surgery
# -----------------

# S2 [Strong] : Stage 4 (metastatic) with breast-conserving surgery
violations["S2"]  = (stade == 4) & (chir == "BREAST CONSERVING")


# Group H : Histology
# --------------------

# H1 [Atypical] : mismatch between the primary histological type and the
histo_map = {
    "Breast Invasive Ductal Carcinoma":          "Ductal/NST",
    "Breast Invasive Lobular Carcinoma":         "Lobular",
    "Breast Mixed Ductal and Lobular Carcinoma": "Mixed",
    "Breast Invasive Mixed Mucinous Carcinoma":  "Mucinous",
}
h1 = pd.Series(False, index=df.index)
for type_val, expected_sub in histo_map.items():
    h1 |= (type_h == type_val) & (sub_h != "") & (sub_h != "nan") & (sub_h != expected_sub)
violations["H1"]  = h1


# ---------------------------------------------------------------------------
# Apply exclusions
# ---------------------------------------------------------------------------

excl_mask = pd.Series(False, index=df.index)
for mask in violations.values():
    excl_mask |= mask.fillna(False)

n_start = len(df)
print(f"\nCoherence rules ({len(violations)} applied, action = EXCLURE):")
for rule_id, mask in violations.items():
    n = int(mask.fillna(False).sum())
    print(f"  {rule_id:<8} {n:4d} patients  ({100*n/n_start:.1f}%)")

n_excl = int(excl_mask.sum())
df_clean = df[~excl_mask].copy()
print(f"\nRemoved : {n_excl} patients ({100*n_excl/n_start:.1f}%)")
print(f"Retained: {len(df_clean)} patients ({100*len(df_clean)/n_start:.1f}%)")
# Expected: 702 retained


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

df_clean.to_csv(OUT, index=False)
print(f"\nSaved: {OUT.name}  ({len(df_clean)} patients, {len(df_clean.columns)} columns)")

# Exclusion log (useful for auditing rules without re-running preprocessing)
df_excl = df[excl_mask].copy()
for rule_id, mask in violations.items():
    df_excl[f"excl_{rule_id}"] = mask[excl_mask].fillna(False).values

rule_cols = [c for c in df_excl.columns if c.startswith("excl_")]
df_excl["rules_violated"] = df_excl[rule_cols].apply(
    lambda row: " | ".join(c.replace("excl_", "") for c in rule_cols if row[c]),
    axis=1,
)
df_excl[["sous_type_pam50", "statut_er", "statut_pr", "statut_her2",
          "hormonotherapie", "chimiotherapie", "type_chirurgie",
          "grade_sbr", "stade_tumoral", "ganglions_positifs",
          "indice_nottingham", "rules_violated"]].to_csv(EXCL_LOG, index=False)
print(f"Exclusion log: {EXCL_LOG.name}  ({n_excl} patients)")
