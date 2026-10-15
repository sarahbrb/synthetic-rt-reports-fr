"""
Campaign configuration — apertus70b, medllama8b, openbiollm8b, openbiollm70b.
No executable logic. Imported by make_jobs_jeanzay.py and evaluate_cag.py.
"""

# ── Generation models ──────────────────────────────────────────────────────────

MODELS = {
    "apertus70b":    "swiss-ai/Apertus-70B-Instruct-2509",
    "medllama8b":    "Henrychur/MMed-Llama-3-8B-EnIns",
    "openbiollm8b":  "aaditya/Llama3-OpenBioLLM-8B",
    "openbiollm70b": "aaditya/Llama3-OpenBioLLM-70B",
}

# ── Jean Zay SLURM config ─────────────────────────────────────────────────────
# Seul bbo@v100 est actif. GPU disponibles confirmés par scontrol :
#   gpu_p13 → V100 16GB, 4/nœud (4×16=64 GB)  → OK pour 8B, pas pour 70B
#   gpu_p2  → V100 32GB, 8/nœud (8×32=256 GB) → OK pour 70B fp16
#   gpu_p5  → A100 80GB, bbo@a100 (accès refusé)
#   gpu_p6  → H100 80GB, bbo@h100 (accès refusé)
#
# V100 ne supporte PAS bf16 → dtype=fp16 obligatoire pour tous les modèles.
# 8B fp16  = 16 GB → 2 × V100 16GB = 32 GB (avec marge pour le KV cache)
# 70B fp16 = 140 GB → 8 × V100 32GB = 256 GB

ACCOUNT        = "bbo@v100"
QOS_SHORT      = "qos_gpu-t3"   # max 20h — modèles 8B
QOS_LONG       = "qos_gpu-t4"   # max 100h — modèles 70B
QOS_EVAL       = "qos_gpu-t3"

PARTITION_8B   = "gpu_p13"      # V100 16GB, 4 GPU/nœud
PARTITION_70B  = "gpu_p2"       # V100 32GB, 8 GPU/nœud → 256 GB
PARTITION_EVAL = "gpu_p13"

N_GPU_8B   = 2    # 2 × 16GB = 32GB (8B fp16 ≈ 16GB + KV cache)
N_GPU_70B  = 8    # 8 × 32GB = 256GB (70B fp16 ≈ 140GB)
N_GPU_EVAL = 2

# CPUs : gpu_p13 → 40 CPU / 4 GPU = 10 CPU/GPU ; gpu_p2 → 40 CPU / 8 GPU = 5 CPU/GPU
N_CPU_8B   = 20   # 2 GPU × 10 CPU/GPU
N_CPU_70B  = 40   # 8 GPU × 5 CPU/GPU
N_CPU_EVAL = 20

MODEL_JEANZAY = {
    # fp16 obligatoire : V100 ne supporte pas bf16
    "medllama8b":    {"partition": PARTITION_8B,  "n_gpu": N_GPU_8B,  "n_cpu": N_CPU_8B,  "qos": QOS_SHORT, "walltime": "20:00:00", "dtype": "fp16"},
    "openbiollm8b":  {"partition": PARTITION_8B,  "n_gpu": N_GPU_8B,  "n_cpu": N_CPU_8B,  "qos": QOS_SHORT, "walltime": "20:00:00", "dtype": "fp16"},
    "apertus70b":    {"partition": PARTITION_70B, "n_gpu": N_GPU_70B, "n_cpu": N_CPU_70B, "qos": QOS_LONG,  "walltime": "90:00:00", "dtype": "fp16"},
    "openbiollm70b": {"partition": PARTITION_70B, "n_gpu": N_GPU_70B, "n_cpu": N_CPU_70B, "qos": QOS_LONG,  "walltime": "90:00:00", "dtype": "fp16"},
}

# ── Sampling presets ───────────────────────────────────────────────────────────
# (temperature, top_p, repetition_penalty)

SAMPLING_PRESETS = {
    "conservateur": (0.3, 0.85, 1.10),
    "standard":     (0.5, 0.90, 1.10),
    "exploratoire": (0.8, 0.95, 1.15),
}

# ── Dataset / prompt setup ─────────────────────────────────────────────────────

DATASETS     = ["metabric", "seer"]
PROMPT_TYPES = ["0shot", "5shot"]
N_PER_RUN    = 700

# ── BERTScore models (evaluation) ─────────────────────────────────────────────
# key → HuggingFace model ID.
# evaluate_cag.py computes scores for all three and stores results in separate columns.

BERTSCORE_MODELS = {
    "camembert_base": "camembert-base",
    "camembert_bio":  "almanach/camembert-bio-base",
    "drbert":         "Dr-BERT/DrBERT-7GB",
}
