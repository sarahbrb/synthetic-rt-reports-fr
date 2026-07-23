"""
Préprocessing METABRIC pour la campagne CAG.

Prend le fichier brut téléchargé depuis cBioPortal et produit le fichier
final utilisé par generate_cag.py. Trois étapes enchaînées :
  1. Sélection et renommage des colonnes cliniques
  2. Restriction aux patients ayant reçu une radiothérapie
  3. Exclusion des profils cliniquement incohérents

Les règles d'exclusion (étape 3) ont été définies avec une radio-oncologue
praticienne à partir des référentiels SENORIF 2021-2022 et ONCO AURA 2023,
puis validées cas par cas par retour d'expert.

Usage :
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
# 1. Chargement et sélection des colonnes
# ---------------------------------------------------------------------------

df = pd.read_csv(RAW, sep="\t", low_memory=False)
df.replace("NA", np.nan, inplace=True)

print(f"Chargé : {len(df)} patients, {len(df.columns)} colonnes")

# Seules ces colonnes sont utiles pour la génération ; on renomme en français
# pour garder la cohérence avec les templates de prompts.
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
# 2. Restriction aux cas traités par radiothérapie
# ---------------------------------------------------------------------------

n_before = len(df)
df = df[df["radiotherapie"] == "YES"].copy()
print(f"Après filtre radiothérapie : {len(df)} patients (retirés : {n_before - len(df)})")

# La colonne ne porte plus d'information différenciante, on la supprime.
df.drop(columns=["radiotherapie"], inplace=True)


# ---------------------------------------------------------------------------
# 3. Filtrage des profils cliniquement incohérents
# ---------------------------------------------------------------------------
#
# Chaque règle correspond à une combinaison de caractéristiques biologiquement
# ou cliniquement impossible dans ce contexte. Validée par l'experte médicale.
#
# Abrév. : RH = récepteurs hormonaux (ER/PR), TN = triple négatif,
#           NPI = Nottingham Prognostic Index.

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

rh_neg = (er == "Negative") & (pr == "Negative")
tn     = rh_neg & (her2 == "Negative")

violations = {
    # Hormonothérapie sans cible hormonale — biologiquement sans sens.
    "R5":  rh_neg & (horm == "YES"),
    "R7":  tn & (horm == "YES"),
    "R8":  (pam == "Basal") & (er == "Negative") & (horm == "YES"),
    "R11": (pam == "claudin-low") & (er == "Negative") & (horm == "YES"),
    "R14": (her2 == "Positive") & rh_neg & (horm == "YES"),

    # ER+ sans hormonothérapie — rare cliniquement ; à 15.7% dans METABRIC,
    # l'experte recommande d'exclure (probable erreur de saisie ou protocole
    # non renseigné).
    "R6":  (er == "Positive") & (horm == "NO"),

    # Sous-type PAM50 vs statut IHC : contradictions par définition.
    "R1":  (pam == "LumA") & (er == "Negative"),   # LumA = RE+RP+ par déf.
    "R2":  (pam == "LumB") & (er == "Negative"),
    "R4":  (pam == "Her2") & (her2 == "Negative"),  # discordance ARNm/protéine

    # Stade vs ganglions — la classification AJCC est incompatible avec ces
    # combinaisons (Stade I = N0 obligatoirement).
    "SG1": (stade == 1) & (gangl > 0),
    "SG4": (stade < 3) & (gangl >= 10),  # N3 impose un stade ≥ IIIC
    "ST1": (stade == 1) & (gangl > 3),

    # Chimiothérapie au stade 0 (DCIS) : le carcinome in situ n'envahit pas
    # les ganglions, la chimio n'est pas indiquée.
    "C1":  (stade == 0) & (chim == "YES"),

    # HER2-enrichi non traité par chimio — recommandé systématiquement dès
    # que la tumeur dépasse 5 mm. L'experte confirme d'exclure si taille NR
    # aussi (toutes les tailles METABRIC sont > 5mm).
    "C3":  (pam == "Her2") & (chim == "NO") & ((taille > 5) | taille.isna()),

    # Stade 4 métastatique + chirurgie conservatrice : incompatible.
    "S2":  (stade == 4) & (chir == "BREAST CONSERVING"),

    # Contradictions grade / sous-type.
    "G1":  (pam == "Basal") & (grade == 1),       # Basal = quasi exclusivement G3
    "G3":  (pam == "LumA") & (grade == 3),         # LumA = bas grade, G3 → reclassement LumB
    "G5":  type_h.str.contains("Mucinous", na=False) & (grade == 3),  # mucineux = bas grade

    # NPI ≥ 6 implique mathématiquement grade ≥ 2 (NPI = 0.2×taille + G + N).
    "NPI2": (npi >= 6.0) & (grade == 1),

    # Discordance entre type histologique principal et sous-type renseigné.
    # Ex. : type = IDC mais sous-type = Lobular → erreur de saisie.
}

# Règle H1 traitée séparément car elle compare deux colonnes texte.
histo_map = {
    "Breast Invasive Ductal Carcinoma":          "Ductal/NST",
    "Breast Invasive Lobular Carcinoma":         "Lobular",
    "Breast Mixed Ductal and Lobular Carcinoma": "Mixed",
    "Breast Invasive Mixed Mucinous Carcinoma":  "Mucinous",
}
h1 = pd.Series(False, index=df.index)
for type_val, expected_sub in histo_map.items():
    h1 |= (type_h == type_val) & (sub_h != "") & (sub_h != "nan") & (sub_h != expected_sub)
violations["H1"] = h1

# Un patient est exclu s'il viole au moins une règle.
excl_mask = pd.Series(False, index=df.index)
for mask in violations.values():
    excl_mask |= mask.fillna(False)

# Résumé règle par règle avant d'appliquer l'exclusion globale.
print(f"\nRègles d'exclusion ({len(violations)} règles) :")
n_total = len(df)
for rule_id, mask in sorted(violations.items()):
    n = int(mask.fillna(False).sum())
    print(f"  {rule_id:<6} {n:4d} patients  ({100*n/n_total:.1f}%)")

df_clean = df[~excl_mask].copy()
n_excl   = int(excl_mask.sum())
print(f"\nExclus  : {n_excl} patients ({100*n_excl/n_total:.1f}%)")
print(f"Retenus : {len(df_clean)} patients ({100*len(df_clean)/n_total:.1f}%)")


# ---------------------------------------------------------------------------
# 4. Sauvegarde
# ---------------------------------------------------------------------------

df_clean.to_csv(OUT, index=False)
print(f"\nDataset final → {OUT.name}  ({len(df_clean)} patients, {len(df_clean.columns)} colonnes)")

# Log des patients exclus avec les règles violées — utile pour valider ou
# retravailler les règles sans relancer tout le préprocessing.
df_excl = df[excl_mask].copy()
for rule_id, mask in violations.items():
    df_excl[f"excl_{rule_id}"] = mask[excl_mask].fillna(False).values

rule_cols = [c for c in df_excl.columns if c.startswith("excl_")]
df_excl["règles_violées"] = df_excl[rule_cols].apply(
    lambda row: " | ".join(c.replace("excl_", "") for c in rule_cols if row[c]),
    axis=1,
)
df_excl[["sous_type_pam50", "statut_er", "statut_pr", "statut_her2",
          "hormonotherapie", "chimiotherapie", "type_chirurgie",
          "grade_sbr", "stade_tumoral", "ganglions_positifs",
          "indice_nottingham", "règles_violées"]].to_csv(EXCL_LOG, index=False)
print(f"Log exclusion  → {EXCL_LOG.name}  ({n_excl} patients)")
