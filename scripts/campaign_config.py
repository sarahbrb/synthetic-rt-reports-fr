"""
Multi-model campaign configuration (generation + evaluation).
Central place for all decisions: which model runs on which node,
which sampling configs to compare.

No executable logic — just data, imported by make_jobs.py to generate OAR scripts.
"""

# ──────────────────────────────────────────────────────────────────────────
# Models: short name -> HuggingFace model ID
# ──────────────────────────────────────────────────────────────────────────
MODELS = {
    "qwen7b":     "Qwen/Qwen2.5-7B-Instruct",
    "qwen32b":    "Qwen/Qwen2.5-32B-Instruct",
    "mistral7b":  "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral12b": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    "apertus8b":  "swiss-ai/Apertus-8B-Instruct-2509",
    "apertus70b": "swiss-ai/Apertus-70B-2509",
}

# Model-to-node mapping: short_name -> (node_pool, n_gpu, walltime_per_combo, dtype)
# One OAR job = one combination (dataset × prompt_type × sampling_preset).
#
# lig-gpu1 to gpu5 excluded: Pascal/sm_61, incompatible with PyTorch sm_70+ requirement.
#
# Walltimes extrapolated from measured qwen7b throughput (~2.4 min/CR on
# 2× RTX 2080 Ti with device_map="auto").
MODEL_NODE = {
    # dtype: fp16 for Turing (gpu6/7, sm_75 — bf16 emulated in fp32, 2× slower)
    #        bf16 for Ampere/Hopper (gpu8/9/10 — native bf16 tensor cores)
    #        gpu9 (RTX 8000) is Turing but used for 70B where numerical stability
    #        outweighs the speed cost.

    # 7B/8B → gpu6, gpu7, or gpu10 (H100 overkill but reduces waiting time)
    "qwen7b":     (["lig-gpu6.imag.fr", "lig-gpu7.imag.fr", "lig-gpu10.imag.fr"],  2, "30:00:00", "fp16"),
    "mistral7b":  (["lig-gpu6.imag.fr", "lig-gpu7.imag.fr", "lig-gpu10.imag.fr"],  2, "30:00:00", "fp16"),
    "apertus8b":  (["lig-gpu6.imag.fr", "lig-gpu7.imag.fr", "lig-gpu10.imag.fr"],  2, "30:00:00", "fp16"),
    # 12B → gpu7, gpu8, or gpu10
    "mistral12b": (["lig-gpu7.imag.fr", "lig-gpu8.imag.fr", "lig-gpu10.imag.fr"],  2, "45:00:00", "fp16"),
    # 24B/32B → gpu8 or gpu10 (both have ≥96 GB over 2 GPUs)
    "mistral24b": (["lig-gpu8.imag.fr", "lig-gpu10.imag.fr"],                       2, "90:00:00", "bf16"),
    "qwen32b":    (["lig-gpu8.imag.fr", "lig-gpu10.imag.fr"],                       2, "90:00:00", "bf16"),
    # 70B → gpu9 only (4×48 GB = 192 GB required; gpu10 only has 96 GB total)
    "apertus70b": (["lig-gpu9.imag.fr"],                                             4, "96:00:00", "bf16"),
}

# ──────────────────────────────────────────────────────────────────────────
# Sampling configurations to compare (3 presets: conservative/standard/exploratory)
# name -> (temperature, top_p, repetition_penalty)
# "standard" matches the defaults already used in generate_cag.py
# ──────────────────────────────────────────────────────────────────────────
SAMPLING_PRESETS = {
    "conservateur": (0.3, 0.85, 1.10),
    "standard":     (0.5, 0.90, 1.10),
    "exploratoire": (0.8, 0.95, 1.15),
}

DATASETS = ["metabric", "seer"]
PROMPT_TYPES = ["0shot", "5shot"]
N_PER_RUN = 700

# apertus70b on gpu9 (4 GPUs, no NVLink): ~12 min/CR → 700 CR = 140h >> 100h (OAR cap).
# Fix: 2 jobs per combination using --skip. Part1=400 CR (~80h), Part2=300 CR (~60h).
N_SPLIT = {
    "apertus70b": [
        {"n": 400, "skip": 0,   "walltime": "84:00:00", "suffix": "part1"},
        {"n": 300, "skip": 400, "walltime": "64:00:00", "suffix": "part2"},
    ]
}
