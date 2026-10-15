"""
Multi-model campaign configuration (generation + evaluation).
No executable logic, just data, imported by make_jobs.py to generate OAR scripts
"""


# Models: short name -> HuggingFace model ID

MODELS = {
    "qwen7b":     "Qwen/Qwen2.5-7B-Instruct",
    "qwen32b":    "Qwen/Qwen2.5-32B-Instruct",
    "mistral7b":  "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral12b": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    "apertus8b":  "swiss-ai/Apertus-8B-Instruct-2509",
    # Medical instruct models — verify HF model IDs and accept licenses before submitting
    "medgemma27b": "google/medgemma-27b-it",   # requires HF license acceptance
}

# Model-to-node mapping: short_name -> (node_pool, n_gpu, walltime_per_combo, dtype)
# One OAR job = one combination (dataset × prompt_type × sampling_preset).


MODEL_NODE = {
    
    # 7B/8B: ~14-16 GB fp16. 1 GPU on A6000 (48 GB) --> 32 GB headroom, no OOM risk
    "qwen7b":     (["lig-gpu8.imag.fr", "lig-gpu10.imag.fr"],  1, "30:00:00", "fp16"),
    "mistral7b":  (["lig-gpu8.imag.fr", "lig-gpu10.imag.fr"],  1, "30:00:00", "fp16"),
    # apertus8b: fp32 required 
    # fp32 weights ≈ 32 GB, still comfortably fits in A6000 (48 GB)
    "apertus8b":  (["lig-gpu8.imag.fr", "lig-gpu10.imag.fr"],  1, "30:00:00", "fp32"),
    # 12B: ~24 GB fp16. 1 GPU on A6000 (48 GB) --> 24 GB headroom, safe for 5-shot
    "mistral12b": (["lig-gpu8.imag.fr", "lig-gpu10.imag.fr"],  1, "45:00:00", "fp16"),
    # 24B/32B: 48-64 GB bf16. 2 GPUs on gpu8 (96 GB) or gpu10 --> ≥32 GB headroom
    "mistral24b": (["lig-gpu8.imag.fr", "lig-gpu10.imag.fr"],  2, "90:00:00", "bf16"),
    "qwen32b":    (["lig-gpu8.imag.fr", "lig-gpu10.imag.fr"],  2, "90:00:00", "bf16"),
    # MedGemma-27B: ~54 GB bf16. 2 GPUs on gpu8/gpu10 (96 GB total)
    "medgemma27b": (["lig-gpu8.imag.fr", "lig-gpu10.imag.fr"],  2, "90:00:00", "bf16"),
}


# Sampling configurations to compare (3 presets: conservateur, standard, exploratoire) -> (temperature, top_p, repetition_penalty)
# "standard" matches the defaults already used in generate_cag.py

SAMPLING_PRESETS = {
    "conservateur": (0.3, 0.85, 1.10),
    "standard": (0.5, 0.90, 1.10),
    "exploratoire": (0.8, 0.95, 1.15),
}

DATASETS = ["metabric", "seer"]
PROMPT_TYPES = ["0shot", "5shot"]
N_PER_RUN = 700

N_SPLIT = {}
