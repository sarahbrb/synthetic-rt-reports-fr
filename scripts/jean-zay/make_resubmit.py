"""
Génère les scripts de relance pour les jobs incomplets/échoués.

Logique :
  - 700 lignes → complet, on ne relance pas
  - 0 lignes ou fichier absent → relance depuis 0 (mode "w")
  - 0 < N < 700 → continuation depuis N (--skip N --n 700-N, mode "a" grâce au fix)

Usage (depuis Jean Zay, après avoir annulé les anciens jobs) :
  python scripts/make_resubmit.py
  bash $SCRATCH/CAG_slurm/resubmit_jeanzay.sh
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
JOBS_DIR.mkdir(parents=True, exist_ok=True)

MODULE_LOAD = "pytorch-gpu/py3/2.7.0"
ENV_BLOCK = f"""\
module purge
module load {MODULE_LOAD}

export PYTHONPATH=$SCRATCH/synth-data-generation/pyuser:$PYTHONPATH
export HF_HOME=$SCRATCH/synth-data-generation/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/synth-data-generation/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1"""

EVAL_WALLTIME = "10:00:00"


def count_lines(path: Path) -> int:
    if not path.exists():
        return -1  # missing
    with path.open() as f:
        return sum(1 for _ in f)


def write_gen_script(tag: str, model_key: str, dataset: str, prompt_type: str,
                     temp: float, top_p: float, rep_pen: float,
                     skip: int, n: int) -> Path:
    cfg      = MODEL_JEANZAY[model_key]
    out_file = OUT_DIR / f"{model_key}_{dataset}_{prompt_type}_{list(SAMPLING_PRESETS.keys())[[v for v in SAMPLING_PRESETS.values()].index((temp, top_p, rep_pen))]}.jsonl"
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
    --n             {n} \\
    --skip          {skip} \\
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


def write_eval_script(model_key: str) -> Path:
    tag    = f"eval_{model_key}"
    script = JOBS_DIR / f"{tag}.sh"
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


# ── Analyse de l'état des outputs ─────────────────────────────────────────────

preset_names = list(SAMPLING_PRESETS.keys())
preset_vals  = list(SAMPLING_PRESETS.values())

submit_lines = [
    "#!/bin/bash",
    "# Relance intelligente : continuation pour partiels, fresh pour vides/manquants.",
    "# Chaque ligne ci-dessous correspond à un job utile.",
    "# Lancer APRÈS : scancel -u $USER",
    "",
    "set -e",
    "",
]

total_jobs = 0
complete   = 0
skipped    = 0

for model_key in MODELS:
    submit_lines += [
        f"# {'─'*60}",
        f"# {model_key}",
        f"# {'─'*60}",
        f'echo "=== {model_key} ==="',
        f"DEPS_{model_key.upper()}=''",
        "",
    ]

    for dataset in DATASETS:
        for prompt_type in PROMPT_TYPES:
            for preset_name, (temp, top_p, rep_pen) in SAMPLING_PRESETS.items():
                base_tag = f"{model_key}_{dataset}_{prompt_type}_{preset_name}"
                out_file = OUT_DIR / f"{base_tag}.jsonl"
                n_done   = count_lines(out_file)

                if n_done >= N_PER_RUN:
                    complete += 1
                    submit_lines.append(f"# SKIP {base_tag} (already {n_done}/{N_PER_RUN} lines)")
                    submit_lines.append("")
                    continue

                if n_done <= 0:
                    # fresh run
                    skip = 0
                    n    = N_PER_RUN
                    tag  = base_tag
                elif 0 < n_done < N_PER_RUN:
                    # continuation
                    skip = n_done
                    n    = N_PER_RUN - n_done
                    tag  = f"{base_tag}_cont{n_done}"

                script = write_gen_script(tag, model_key, dataset, prompt_type,
                                          temp, top_p, rep_pen, skip, n)
                submit_lines += [
                    f"echo '  Submitting {tag} (skip={skip}, n={n}) ...'",
                    f"JID=$(sbatch --parsable {script})",
                    f'echo "    → job $JID"',
                    f"DEPS_{model_key.upper()}=\"${{DEPS_{model_key.upper()}}}:$JID\"",
                    "",
                ]
                total_jobs += 1

    eval_script = write_eval_script(model_key)
    submit_lines += [
        f"if [ -n \"$DEPS_{model_key.upper()}\" ]; then",
        f"  echo '  Submitting eval_{model_key} ...'",
        f"  JID=$(sbatch --parsable --dependency=afterok$DEPS_{model_key.upper()} {eval_script})",
        f'  echo "    → eval job $JID"',
        f"fi",
        "",
    ]

submit_lines += [
    'echo ""',
    f'echo "Submitted {total_jobs} generation jobs ({complete} already complete, skipped)."',
    f'echo "Monitor: squeue -u $USER"',
    f'echo "Outputs: {OUT_DIR}/"',
]

submit_path = SCRATCH / "CAG_slurm" / "resubmit_jeanzay.sh"
submit_path.write_text("\n".join(submit_lines) + "\n")
submit_path.chmod(0o755)

print(f"[OK] Scripts → {JOBS_DIR}")
print(f"[OK] Submit script → {submit_path}")
print()
print(f"Résumé:")
print(f"  {complete} fichiers déjà complets (700 lignes) → ignorés")
print(f"  {total_jobs} jobs à soumettre")
print()
print("Commandes :")
print(f"  scancel -u $USER")
print(f"  bash {submit_path}")
