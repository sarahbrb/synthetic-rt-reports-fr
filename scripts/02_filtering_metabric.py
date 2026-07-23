"""
METABRIC — Preprocessing Step 3 : Filtrage des cas incohérents
===============================================================
Supprime les patients violant les règles validées par l'experte médicale.

Règles EXCLURE (20 règles, validées experte) :
  R5/R12 : ER- ET PR- + Hormono OUI
  R7     : Triple Négatif + Hormono OUI
  R8     : Basal + ER- + Hormono OUI
  R11    : Claudin-low + ER- + Hormono OUI
  R14    : HER2+ + RH- + Hormono OUI
  C1     : Stade 0 (DCIS) + Chimio OUI
  G1     : Basal + Grade 1
  SG1    : Stade 1 + ganglions > 0 (AJCC Stage I = N0)
  SG4    : Stade < 3 + ganglions ≥ 10 (N3 → IIIC obligatoire)
  NPI2   : NPI ≥ 6.0 + Grade 1 (contradiction mathématique)
  R1     : LumA + ER- (par définition LumA = RE+RP+)
  R2     : LumB + ER- (idem)
  R4     : HER2-enrichi PAM50 + HER2- IHC
  R6     : ER+ + Hormono NON (% trop élevé — recommandation experte)
  S2     : Stade 4 + Chirurgie conservatrice
  G3     : LumA + Grade 3
  ST1    : Stade 1 + ganglions > 3
  H1     : IDC + sous-type ≠ Ductal/NST (discordance histologique)
  G5     : Carcinome mucineux + Grade 3
  C3     : HER2-enrichi + Chimio NON + taille > 5mm (ou taille NR)

Usage :
    conda activate synthgen
    python scripts/02_filtering_metabric.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT    = DATA_DIR / "metabric_step1_2.csv"
OUTPUT   = DATA_DIR / "metabric_step3_filtered.csv"
REPORT   = DATA_DIR / "metabric_step3_exclusion_report.csv"

# ─────────────────────────────────────────────────────────────────────────────
# RÈGLES VALIDÉES PAR L'EXPERTE
# ─────────────────────────────────────────────────────────────────────────────
EXCL_RULES = {
    "R5/R12","R7","R8","R11","R14",
    "C1","G1","SG1","SG4","NPI2",
    "R1","R2","R4","R6","S2","G3","ST1","H1","G5","C3",
}

RULE_LABELS = {
    "R5/R12": "ER- ET PR- + Hormonothérapie OUI",
    "R7":     "Triple Négatif (ER-/PR-/HER2-) + Hormonothérapie OUI",
    "R8":     "Basal-like + ER- + Hormonothérapie OUI",
    "R11":    "Claudin-low + ER- + Hormonothérapie OUI",
    "R14":    "HER2+ + RH- (ER- ET PR-) + Hormonothérapie OUI",
    "C1":     "Stade 0 (DCIS) + Chimiothérapie OUI",
    "G1":     "Basal-like + Grade 1 (biologiquement impossible)",
    "SG1":    "Stade 1 + ganglions positifs > 0 (AJCC Stage I = N0)",
    "SG4":    "Stade < 3 + ganglions ≥ 10 (N3 = Stade IIIC obligatoire)",
    "NPI2":   "NPI ≥ 6.0 + Grade 1 (contradiction mathématique NPI)",
    "R1":     "LumA + ER- (LumA défini par RE+RP+ par définition)",
    "R2":     "LumB + ER- (LumB implique ER+ par définition)",
    "R4":     "HER2-enrichi PAM50 + HER2- IHC (discordance validée à exclure)",
    "R6":     "ER+ + Hormonothérapie NON (15.7% — recommandation experte d'exclure)",
    "S2":     "Stade 4 (métastatique) + Chirurgie conservatrice",
    "G3":     "LumA + Grade 3 (LumA = bas grade, Grade 3 = reclassement LumB)",
    "ST1":    "Stade 1 + ganglions positifs > 3 (incompatible avec AJCC Stage I)",
    "H1":     "IDC + sous-type histologique ≠ Ductal/NST (discordance saisie)",
    "G5":     "Carcinome mucineux + Grade 3 (mucineux = bas grade)",
    "C3":     "HER2-enrichi + Chimio NON + taille > 5mm (ou taille NR)",
}

EXPERT_AVIS = {
    "R5/R12": "Oui",
    "R7":     "Oui",
    "R8":     "Oui",
    "R11":    "Oui",
    "R14":    "Oui",
    "C1":     "DCIS n'envahit pas les ganglions, pas de chimiothérapie dans ce contexte",
    "G1":     "Le plus souvent grade III, peut être grade II — jamais vu de grade I",
    "SG1":    "Oui",
    "SG4":    "Oui",
    "NPI2":   "Oui",
    "R1":     "Par définition, le Luminal A est RE et RP positif",
    "R2":     "Identique — LumB implique ER+",
    "R4":     "Oui — discordance ARNm vs protéine fréquente (~47%)",
    "R6":     "Peut avoir une justification mais pourcentage très élevé (15.7%) — recommande de les enlever",
    "S2":     "Oui",
    "G3":     "Oui",
    "ST1":    "Si envahissement ganglionnaire, stade plus élevé — d'accord",
    "H1":     "Oui",
    "G5":     "Oui",
    "C3":     "J'ai checké le dataset, toutes tailles tumorales sont >5mm — exclure aussi si taille NR",
}

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT)
N_init = len(df)
print(f"[LOAD]  {N_init} patients  x  {len(df.columns)} colonnes")
print(f"        Source : {INPUT.name}")

# ─────────────────────────────────────────────────────────────────────────────
# CALCUL DES VIOLATIONS AU NIVEAU PATIENT
# ─────────────────────────────────────────────────────────────────────────────
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
sub_h = df["sous_type_histologique"].fillna("").astype(str)
type_h= df["type_histologique"].astype(str)

rh_neg = (er == "Negative") & (pr == "Negative")
tn     = rh_neg & (her2 == "Negative")

HISTO_MAP = {
    "Breast Invasive Ductal Carcinoma":           "Ductal/NST",
    "Breast Invasive Lobular Carcinoma":          "Lobular",
    "Breast Mixed Ductal and Lobular Carcinoma":  "Mixed",
    "Breast Invasive Mixed Mucinous Carcinoma":   "Mucinous",
}

violations = {
    "R5/R12": rh_neg & (horm == "YES"),
    "R7":     tn & (horm == "YES"),
    "R8":     (pam == "Basal") & (er == "Negative") & (horm == "YES"),
    "R11":    (pam == "claudin-low") & (er == "Negative") & (horm == "YES"),
    "R14":    (her2 == "Positive") & rh_neg & (horm == "YES"),
    "C1":     (stade == 0) & (chim == "YES"),
    "G1":     (pam == "Basal") & (grade == 1),
    "SG1":    (stade == 1) & (gangl > 0),
    "SG4":    (stade < 3) & (gangl >= 10),
    "NPI2":   (npi >= 6.0) & (grade == 1),
    "R1":     (pam == "LumA") & (er == "Negative"),
    "R2":     (pam == "LumB") & (er == "Negative"),
    "R4":     (pam == "Her2") & (her2 == "Negative"),
    "R6":     (er == "Positive") & (horm == "NO"),
    "S2":     (stade == 4) & (chir == "BREAST CONSERVING"),
    "G3":     (pam == "LumA") & (grade == 3),
    "ST1":    (stade == 1) & (gangl > 3),
    "G5":     type_h.str.contains("Mucinous", na=False) & (grade == 3),
    # C3 : HER2-enrichi + Chimio NON + taille > 5mm OU taille NR
    "C3":     (pam == "Her2") & (chim == "NO") & ((taille > 5) | taille.isna()),
}

# Cohérence histologique H1
h1_mask = pd.Series(False, index=df.index)
for type_val, expected_sub in HISTO_MAP.items():
    h1_mask |= (type_h == type_val) & (sub_h != "") & (sub_h != "nan") & (sub_h != expected_sub)
violations["H1"] = h1_mask

# ─────────────────────────────────────────────────────────────────────────────
# UNION DE TOUTES LES VIOLATIONS EXCLURE
# ─────────────────────────────────────────────────────────────────────────────
excl_mask = pd.Series(False, index=df.index)
for rule_id, mask in violations.items():
    if rule_id in EXCL_RULES:
        excl_mask |= mask.fillna(False)

# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT D'EXCLUSION DÉTAILLÉ
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[STEP 3] FILTRAGE DES CAS INCOHÉRENTS")
print("=" * 65)
print(f"\nRègles EXCLURE validées par l'experte ({len(EXCL_RULES)} règles) :")

report_rows = []
for rule_id in sorted(EXCL_RULES):
    mask = violations.get(rule_id, pd.Series(False, index=df.index)).fillna(False)
    n_viol = int(mask.sum())
    print(f"  {rule_id:<10} {RULE_LABELS[rule_id]:<55} → {n_viol:4d} patients")
    report_rows.append({
        "règle": rule_id,
        "description": RULE_LABELS[rule_id],
        "avis_experte": EXPERT_AVIS.get(rule_id, ""),
        "n_patients_violant": n_viol,
        "pct_sur_total": round(100 * n_viol / N_init, 2),
    })

n_excl = int(excl_mask.sum())
print(f"\n  → Total patients avec au moins 1 violation : {n_excl} ({100*n_excl/N_init:.1f}%)")
print(f"     (dédoublonnés — un patient peut violer plusieurs règles)")

# ─────────────────────────────────────────────────────────────────────────────
# FILTRAGE
# ─────────────────────────────────────────────────────────────────────────────
df_clean   = df[~excl_mask].copy()
df_excl    = df[excl_mask].copy()
N_final    = len(df_clean)
N_removed  = len(df_excl)

# "radiotherapie" est constante (YES) depuis le filtre de l'étape 1+2 : sans valeur informative.
df_clean   = df_clean.drop(columns=["radiotherapie"])

print(f"\n[FILTRAGE]")
print(f"  Avant : {N_init} patients")
print(f"  Exclus: {N_removed} patients ({100*N_removed/N_init:.1f}%)")
print(f"  Après : {N_final} patients ({100*N_final/N_init:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# AJOUT DES COLONNES DE RAISON D'EXCLUSION (pour traçabilité)
# ─────────────────────────────────────────────────────────────────────────────
df_excl = df_excl.copy()
for rule_id, mask in violations.items():
    if rule_id in EXCL_RULES:
        df_excl[f"excl_{rule_id}"] = mask[excl_mask].fillna(False).values

# Colonne résumé des règles violées
def get_rules_violated(row):
    rules = [rid for rid in EXCL_RULES if row.get(f"excl_{rid}", False)]
    return " | ".join(sorted(rules))

excl_rule_cols = [f"excl_{rid}" for rid in EXCL_RULES if f"excl_{rid}" in df_excl.columns]
df_excl["règles_violées"] = df_excl[excl_rule_cols].apply(
    lambda row: " | ".join([
        rid for rid in EXCL_RULES
        if row.get(f"excl_{rid}", False)
    ]), axis=1
)

# ─────────────────────────────────────────────────────────────────────────────
# VÉRIFICATION DES DISTRIBUTIONS AVANT/APRÈS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[CONTRÔLE] Distributions avant/après filtrage :")
check_cols = ["sous_type_pam50", "statut_er", "statut_pr", "statut_her2",
              "hormonotherapie", "chimiotherapie", "type_chirurgie", "etat_menopausique"]
for col in check_cols:
    before = df[col].value_counts(dropna=False).to_dict()
    after  = df_clean[col].value_counts(dropna=False).to_dict()
    print(f"\n  {col}:")
    for val in sorted(set(list(before.keys()) + list(after.keys())), key=str):
        b = before.get(val, 0)
        a = after.get(val, 0)
        diff = b - a
        flag = f"  ← -{diff}" if diff > 0 else ""
        print(f"    {str(val):<30} {b:4d} → {a:4d}{flag}")

# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────
df_clean.to_csv(OUTPUT, index=False)
print(f"\n[SAVE]  Dataset filtré   → {OUTPUT.name}  ({N_final} patients)")

# Rapport CSV des patients exclus avec raison
df_excl_report = df_excl[["sous_type_pam50","statut_er","statut_pr","statut_her2",
                            "hormonotherapie","chimiotherapie","type_chirurgie",
                            "grade_sbr","stade_tumoral","taille_tumorale_mm",
                            "ganglions_positifs","indice_nottingham","règles_violées"]].copy()
df_excl_report.to_csv(REPORT, index=False)
print(f"        Rapport exclusion → {REPORT.name}  ({N_removed} patients exclus)")

# Rapport règle par règle
report_df = pd.DataFrame(report_rows)
print(f"\n[RÉSUMÉ FINAL]")
print(f"  Dataset initial (après étapes 1+2)  : {N_init:5d} patients")
print(f"  Patients exclus (règles experte)      : {N_removed:5d} patients ({100*N_removed/N_init:.1f}%)")
print(f"  Dataset final exploitable             : {N_final:5d} patients ({100*N_final/N_init:.1f}%)")
print(f"\n  Fichier sauvegardé : {OUTPUT}")
