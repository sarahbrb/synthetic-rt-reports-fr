"""
SEER preprocessing pipeline for the CAG campaign.

Produces the patient profiles used by generate_cag.py from the raw SEER export. Two steps applied in sequence:
  1. Select and rename clinically relevant columns
  2. Exclude clinically incoherent profiles

The coherence rules in step 2 were defined with a practising radiation oncologist against SENORIF 2021-2022 and ONCO AURA 2023 guidelines.
Each rule is traceable to its row in:
    Analyse datasets/analyse_incoherences (SEER).xlsx

Expected result (validated against the xlsx summary sheet):
    4024 patients  -->  3832 retained after exclusions (95.2%)

Usage:
    python scripts/preprocess_seer.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW      = DATA_DIR / "SEER_Breast_Cancer.csv"
OUT      = DATA_DIR / "seer_preprocessed.csv"
EXCL_LOG = DATA_DIR / "seer_exclusion_report.csv"



# Step 1 : Column selection


df = pd.read_csv(RAW)
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
print(f"Loaded: {len(df)} patients, {len(df.columns)} columns")

# 'differentiate' is 100% redundant with 'grade' : dropped
cols = {
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
df = df[list(cols.keys())].rename(columns=cols)
print(f"Selected {len(df.columns)} columns")

taille = pd.to_numeric(df["taille_tumorale_mm"], errors="coerce")
gpos   = pd.to_numeric(df["ganglions_positifs"],  errors="coerce")
gexam  = pd.to_numeric(df["ganglions_examines"],  errors="coerce")
grade  = df["grade"].astype(str).str.strip()



# Step 2 : Coherence rule filtering

# Source: Analyse datasets/analyse_incoherences (SEER).xlsx
# Severity levels: Absolute (biologically/clinically impossible) / Strong (clinically inconsistent) / Atypical (rare but possible)


violations = {}


# Group A : T Stage × Tumor size (SENORIF p.12)
# -------------------------------------------------------------

# A1 [Absolute] : T1 with tumor size > 20mm (T1 = ≤ 20mm by definition)
#   0 patients
violations["A1"] = (df["stade_t"] == "T1") & (taille > 20)

# A2 [Absolute] : T2 with size ≤ 20mm or > 50mm (T2 = 20-50mm strict)
#   0 patients
violations["A2"] = (df["stade_t"] == "T2") & ((taille <= 20) | (taille > 50))

# A3 [Absolute] : T3 with size ≤ 50mm (T3 = strictly > 50mm; 50mm is T2 upper bound)
violations["A3"] = (df["stade_t"] == "T3") & (taille <= 50)


# Group B : N Stage × Lymph node count (SENORIF p.12-13)
# -----------------------------------------------------------------------

# B1 [Absolute] : N1 with positive nodes outside [1-3] (N1a axillary = 1-3 nodes)
violations["B1"] = (df["stade_n"] == "N1") & ((gpos < 1) | (gpos > 3))

# B2 [Absolute] : N2 with positive nodes outside [4-9] (N2a axillary = 4-9 nodes)
violations["B2"] = (df["stade_n"] == "N2") & ((gpos < 4) | (gpos > 9))

# B5 [Absolute] : more positive nodes than examined nodes (anatomically impossible)
#   0 patients
violations["B5"] = gpos > gexam


# Group C : AJCC global stage coherence (SENORIF p.13)
# ------------------------------------------------------

# AJCC 2017 v8 valid T×N → stage combinations (N+, M0 only):
#   T1+N1 → IIA  |  T2+N1 → IIB  |  T1/T2/T3+N2 → IIIA
#   T3+N1 → IIIA |  T4+N1/N2 → IIIB  |  any T+N3 → IIIC

VALID_TN_STAGE = {
    ("T1", "N1"): "IIA",
    ("T2", "N1"): "IIB",
    ("T1", "N2"): "IIIA",
    ("T2", "N2"): "IIIA",
    ("T3", "N1"): "IIIA",
    ("T3", "N2"): "IIIA",
    ("T4", "N1"): "IIIB",
    ("T4", "N2"): "IIIB",
}

def stage_mismatch(row):
    t, n, sg = str(row["stade_t"]), str(row["stade_n"]), str(row["stade_global"])
    if n == "N3":
        return sg != "IIIC"
    if (t, n) in VALID_TN_STAGE:
        return sg != VALID_TN_STAGE[(t, n)]
    return False

# C1 [Absolute] : T×N combination inconsistent with the recorded global stage
#   0 patients
violations["C1"] = df.apply(stage_mismatch, axis=1)

# C2 [Absolute] : Stage IIA or IIB recorded with N2 or N3
#   0 patients
violations["C2"] = df["stade_global"].isin(["IIA", "IIB"]) & df["stade_n"].isin(["N2", "N3"])

# C3 [Absolute] : Stage IIIC recorded without N3
#   0 patients
violations["C3"] = (df["stade_global"] == "IIIC") & (df["stade_n"] != "N3")


# Group D : Distant extension vs AJCC stage (SENORIF p.13, ONCO AURA p.47)
# ---------------------------------------------------------------------------

# D1 [Absolute] : A Stage = Distant with a locoregional AJCC stage (IIA - IIIC)
violations["D1"] = (
    (df["extension"] == "Distant") &
    df["stade_global"].isin(["IIA", "IIB", "IIIA", "IIIB", "IIIC"])
)


# Group E : SBR histological grade (SENORIF p.41, ONCO AURA p.27-28)
# --------------------------------------------------------------------

# E1 [Absolute] : Grade recorded as 'anaplastic' or 'Grade IV'
violations["E1"] = grade.str.lower().str.contains("anaplastic|iv", na=False)

# E2 [Atypical] : Grade 1 with triple-negative profile (ER- AND PR-)
violations["E2"] = (grade == "1") & (df["statut_er"] == "Negative") & (df["statut_pr"] == "Negative")



# Apply exclusions


excl_mask = pd.Series(False, index=df.index)
for mask in violations.values():
    excl_mask |= mask.fillna(False)

n_start = len(df)
print(f"\nCoherence rules ({len(violations)} applied, action = EXCLURE):")
for rule_id, mask in violations.items():
    n = int(mask.fillna(False).sum())
    print(f"  {rule_id:<12} {n:4d} patients  ({100*n/n_start:.1f}%)")

n_excl = int(excl_mask.sum())
df_clean = df[~excl_mask].copy()
print(f"\nRemoved : {n_excl} patients ({100*n_excl/n_start:.1f}%)")
print(f"Retained: {len(df_clean)} patients ({100*len(df_clean)/n_start:.1f}%)")
# Expected: 3832 retained



# Save

df_clean.to_csv(OUT, index=False)
print(f"\nSaved: {OUT.name}  ({len(df_clean)} patients, {len(df_clean.columns)} columns)")

# Exclusion log :  to audit rules without re-running preprocessing
df_excl = df[excl_mask].copy()
for rule_id, mask in violations.items():
    df_excl[f"excl_{rule_id}"] = mask[excl_mask].fillna(False).values

rule_cols = [c for c in df_excl.columns if c.startswith("excl_")]
df_excl["rules_violated"] = df_excl[rule_cols].apply(
    lambda row: " | ".join(c.replace("excl_", "") for c in rule_cols if row[c]),
    axis=1,
)
df_excl[["stade_t", "stade_n", "stade_global", "grade", "extension",
          "statut_er", "statut_pr", "taille_tumorale_mm",
          "ganglions_positifs", "ganglions_examines",
          "rules_violated"]].to_csv(EXCL_LOG, index=False)
print(f"Exclusion log: {EXCL_LOG.name}  ({n_excl} patients)")
