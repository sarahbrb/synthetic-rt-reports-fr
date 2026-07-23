"""
configuration de la campagne multi-modèles (génération + évaluation).
centralise les décisions prises ensemble : quel modèle sur quel noeud,
quelles configs de sampling comparer.

ne contient aucune logique d'exécution -- juste des données, importées par
run_one_model.sh (via un petit appel python) et launch_campaign.py.
"""

# ──────────────────────────────────────────────────────────────────────────
# modèles : nom court -> identifiant huggingface
# ──────────────────────────────────────────────────────────────────────────
MODELS = {
    # entrée de test uniquement -- ne fait PAS partie de la campagne des 9 modèles.
    # modèle minuscule (0.5B), utilisé pour valider toute la chaîne (run_one_model.py,
    # sous-process generate_cag.py/evaluate_cag.py, fusion des .jsonl) sans avoir besoin
    # d'un vrai GPU compatible -- tourne sur CPU en quelques minutes.
    "_smoketest": "Qwen/Qwen2.5-0.5B-Instruct",

    "qwen7b":     "Qwen/Qwen2.5-7B-Instruct",
    "qwen32b":    "Qwen/Qwen2.5-32B-Instruct",
    "mistral7b":  "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral12b": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    "mixtral":    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama70b":   "meta-llama/Llama-3.3-70B-Instruct",
    "apertus8b":  "swiss-ai/Apertus-8B-Instruct-2509",
    "apertus70b": "swiss-ai/Apertus-70B-Instruct-2509",
}

# affectation modèle -> (noeud, nb de gpu, walltime PAR COMBINAISON)
# 1 job OAR = 1 combinaison (dataset × prompt_type × sampling_preset)
#
# lig-gpu1 à 5 exclus : Pascal/sm_61, incompatible avec sm_70+ requis.
#
# walltimes estimés à partir de la mesure réelle qwen7b (~2.4 min/CR sur
# 2x RTX2080Ti avec device_map="auto"). extrapolés par taille de modèle.
# llama70b sur H100 NVLink supposé ~3x plus rapide qu'apertus70b.
MODEL_NODE = {
    # modèle      : ([pool de nœuds], n_gpu, walltime/combo)
    # OAR choisit n'importe quel nœud libre dans le pool → parallélisme max.
    #
    # 7B/8B → gpu6 ou gpu7 (même classe de VRAM, ~22 GB par paire)
    "qwen7b":     (["lig-gpu6.imag.fr", "lig-gpu7.imag.fr"],  2, "30:00:00"),
    "mistral7b":  (["lig-gpu6.imag.fr", "lig-gpu7.imag.fr"],  2, "30:00:00"),
    "apertus8b":  (["lig-gpu6.imag.fr", "lig-gpu7.imag.fr"],  2, "30:00:00"),
    # 12B → gpu7 ou gpu8
    "mistral12b": (["lig-gpu7.imag.fr", "lig-gpu8.imag.fr"],  2, "45:00:00"),
    # 24B/32B → gpu8 uniquement (VRAM supérieure requise)
    "mistral24b": (["lig-gpu8.imag.fr"],                       2, "90:00:00"),
    "qwen32b":    (["lig-gpu8.imag.fr"],                       2, "90:00:00"),
    # 70B : nœuds spécialisés uniquement
    "mixtral":    (["lig-gpu9.imag.fr"],                       4, "90:00:00"),   # exclu
    "apertus70b": (["lig-gpu9.imag.fr"],                       4, "96:00:00"),
    "llama70b":   (["lig-gpu10.imag.fr"],                      2, "48:00:00"),   # H100 NVLink
}

# ──────────────────────────────────────────────────────────────────────────
# configs de sampling à comparer (3 retenues : conservateur/standard/exploratoire)
# nom -> (temperature, top_p, repetition_penalty)
# "standard" reprend les défauts déjà utilisés dans generate_cag.py
# ──────────────────────────────────────────────────────────────────────────
SAMPLING_PRESETS = {
    "conservateur": (0.3, 0.85, 1.10),
    "standard":     (0.5, 0.90, 1.10),
    "exploratoire": (0.8, 0.95, 1.15),
}

DATASETS = ["metabric", "seer"]
PROMPT_TYPES = ["0shot", "5shot"]
N_PER_RUN = 700

# apertus70b sur gpu9 (4 GPU sans NVLink) : ~12 min/CR → 700 CR = 140h >> 100h (limite OAR).
# Solution : 2 jobs par combinaison avec --skip. Part1=400 CR (~80h), Part2=300 CR (~60h).
N_SPLIT = {
    "apertus70b": [
        {"n": 400, "skip": 0,   "walltime": "84:00:00", "suffix": "part1"},
        {"n": 300, "skip": 400, "walltime": "64:00:00", "suffix": "part2"},
    ]
}
