"""
Generate SLURM batch scripts for Jean Zay.
Models: apertus70b, medllama8b, openbiollm8b, openbiollm70b.

Jean Zay setup:
  - Account     : bbo@v100  (seul compte actif)
  - Partitions  : gpu_p13 (V100 16GB, 4 GPU/nœud) pour les 8B
                  gpu_p2  (V100 32GB, 8 GPU/nœud) pour les 70B
  - Module      : pytorch-gpu/py3/2.7.0 (Python 3.12)
  - Extra pkgs  : $SCRATCH/pyuser/lib/python3.12/site-packages
  - HF cache    : $SCRATCH/hf_cache
  - Outputs     : $SCRATCH/CAG_outputs/

Generation: 4 modèles × 12 combinaisons (2 datasets × 2 prompts × 3 presets) = 48 jobs.
  8B  models → 2 V100 16GB (gpu_p13), qos_gpu-t3 (max 20h)
  70B models → 8 V100 32GB (gpu_p2),  qos_gpu-t4 (max 100h)

Evaluation: 1 job par modèle, démarre après la réussite de tous ses jobs de génération (afterok).

To reproduce:
  cd /lustre/fshomisc/home/rech/genxai01/uuf58kz/Data_Generation_3/CAG
  python scripts/make_jobs_jeanzay.py
  bash $SCRATCH/CAG_slurm/submit_jeanzay.sh
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from campaign_config import (
    MODELS, MODEL_JEANZAY, SAMPLING_PRESETS, DATASETS, PROMPT_TYPES, N_PER_RUN,
    ACCOUNT, QOS_EVAL, PARTITION_EVAL, N_GPU_EVAL, N_CPU_EVAL,
)

PROJECT_DIR = Path(__file__).parent.parent.resolve()
SCRATCH     = Path(os.environ["SCRATCH"])
JOBS_DIR    = SCRATCH / "CAG_slurm" / "jobscripts"
LOGS_DIR    = SCRATCH / "CAG_slurm" / "logs"
OUT_DIR     = SCRATCH / "CAG_outputs"

for d in (JOBS_DIR, LOGS_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

EVAL_WALLTIME = "10:00:00"   # BERTScore × 3 modèles

MODULE_LOAD = "pytorch-gpu/py3/2.7.0"
ENV_BLOCK = f"""\
module purge
module load {MODULE_LOAD}

export PYTHONPATH=$SCRATCH/pyuser:$PYTHONPATH
export HF_HOME=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1"""


def write_gen_job(model_key: str, tag: str, dataset: str, prompt_type: str,
                  temp: float, top_p: float, rep_pen: float) -> Path:
    cfg      = MODEL_JEANZAY[model_key]
    out_file = OUT_DIR / f"{tag}.jsonl"
    script   = JOBS_DIR / f"{tag}.sh"

    script.write_text(f"""\
#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={cfg['partition']}
#SBATCH --qos={cfg['qos']}
#SBATCH --gres=gpu:{cfg['n_gpu']}
#SBATCH --cpus-per-task={cfg['n_cpu']}
#SBATCH --time={cfg['walltime']}
#SBATCH --output={LOGS_DIR}/{tag}.%j.out
#SBATCH --error={LOGS_DIR}/{tag}.%j.err

{ENV_BLOCK}

echo "=== $(date) -- START {tag} on $(hostname) ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES"

cd {PROJECT_DIR}

python scripts/generate_cag.py \\
    --dataset       {dataset} \\
    --prompt_type   {prompt_type} \\
    --model         {MODELS[model_key]} \\
    --n             {N_PER_RUN} \\
    --skip          0 \\
    --dtype         {cfg['dtype']} \\
    --temperature   {temp} \\
    --top_p         {top_p} \\
    --repetition_penalty {rep_pen} \\
    --out           {out_file}

code=$?
echo "=== $(date) -- END {tag} (exit $code) ==="
exit $code
""")
    script.chmod(0o755)
    return script


def write_eval_job(model_key: str) -> Path:
    tag      = f"eval_{model_key}"
    script   = JOBS_DIR / f"{tag}.sh"
    out_glob = OUT_DIR / f"{model_key}_*.jsonl"

    script.write_text(f"""\
#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION_EVAL}
#SBATCH --qos={QOS_EVAL}
#SBATCH --gres=gpu:{N_GPU_EVAL}
#SBATCH --cpus-per-task={N_CPU_EVAL}
#SBATCH --time={EVAL_WALLTIME}
#SBATCH --output={LOGS_DIR}/{tag}.%j.out
#SBATCH --error={LOGS_DIR}/{tag}.%j.err

{ENV_BLOCK}

echo "=== $(date) -- START {tag} on $(hostname) ==="

cd {PROJECT_DIR}

python scripts/evaluate_cag.py \\
    --input         {out_glob} \\
    --out_per_cr    {OUT_DIR}/{model_key}_eval_per_cr.csv \\
    --out_summary   {OUT_DIR}/{model_key}_eval_summary.csv

code=$?
echo "=== $(date) -- END {tag} (exit $code) ==="
exit $code
""")
    script.chmod(0o755)
    return script


# ── Build submit_jeanzay.sh ────────────────────────────────────────────────────
submit_lines = [
    "#!/bin/bash",
    "# Submit all generation + evaluation jobs to Jean Zay.",
    "# Each model's eval job starts only after ALL its generation jobs succeed.",
    "# Usage: bash $SCRATCH/CAG_slurm/submit_jeanzay.sh",
    "",
    "set -e",
    "",
]

n_gen = 0
n_eval = 0

for model_key in MODELS:
    submit_lines += [
        f"# {'─' * 60}",
        f"# {model_key}  ({MODELS[model_key]})",
        f"# {'─' * 60}",
        f'echo "=== Submitting {model_key} ==="',
        f"DEPS_{model_key.upper()}=''",
        "",
    ]

    for dataset in DATASETS:
        for prompt_type in PROMPT_TYPES:
            for preset_name, (temp, top_p, rep_pen) in SAMPLING_PRESETS.items():
                tag    = f"{model_key}_{dataset}_{prompt_type}_{preset_name}"
                script = write_gen_job(model_key, tag, dataset, prompt_type, temp, top_p, rep_pen)
                submit_lines += [
                    f"echo '  Submitting {tag} ...'",
                    f"JID=$(sbatch --parsable {script})",
                    f'echo "    → job $JID"',
                    f"DEPS_{model_key.upper()}=\"${{DEPS_{model_key.upper()}}}:$JID\"",
                    "",
                ]
                n_gen += 1

    eval_script = write_eval_job(model_key)
    submit_lines += [
        f"echo '  Submitting eval_{model_key} (après tous les jobs de génération) ...'",
        f"JID=$(sbatch --parsable --dependency=afterok$DEPS_{model_key.upper()} {eval_script})",
        f'echo "    → eval job $JID"',
        "",
    ]
    n_eval += 1

submit_lines += [
    'echo ""',
    f'echo "All {n_gen} generation + {n_eval} evaluation jobs submitted."',
    f'echo "Monitor with: squeue -u $USER"',
    f'echo "Outputs: {OUT_DIR}/"',
]

submit_path = SCRATCH / "CAG_slurm" / "submit_jeanzay.sh"
submit_path.write_text("\n".join(submit_lines) + "\n")
submit_path.chmod(0o755)

print(f"[OK] {n_gen} generation scripts + {n_eval} evaluation scripts → {JOBS_DIR}")
print(f"[OK] Submission script → {submit_path}")
print()
print("Next steps:")
print(f"  1. Télécharger les modèles (voir $SCRATCH/CAG_slurm/download_*.sh)")
print(f"  2. bash {submit_path}")
