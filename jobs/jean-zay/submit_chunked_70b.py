#!/usr/bin/env python3
"""
Découpe les jobs 70B en chunks de 175 patients (qos_gpu-t3, 20h max)
et les soumet en chaîne via --dependency=afterok.

Usage (depuis le nœud de login) :
    python3 $SCRATCH/CAG_slurm/submit_chunked_70b.py [--dry-run]
    python3 $SCRATCH/CAG_slurm/submit_chunked_70b.py --cancel-old  # annule les jobs pendants avant
"""

import argparse
import os
import subprocess
from pathlib import Path

SCRATCH   = os.environ["SCRATCH"]
OUT_DIR   = Path(f"{SCRATCH}/CAG_outputs")
LOG_DIR   = Path(f"{SCRATCH}/CAG_slurm/logs")
PROJ_DIR  = Path("/lustre/fshomisc/home/rech/genxai01/uuf58kz/Data_Generation_3/CAG")

CHUNK_SIZE = 175   # ~9.7h à 200s/patient — bonne marge sous les 20h de t3
N_TOTAL    = 700
TIME_LIMIT = "18:00:00"  # conservateur vs 20h max du t3

MODELS = {
    "apertus70b":    "swiss-ai/Apertus-70B-Instruct-2509",
    "openbiollm70b": "aaditya/Llama3-OpenBioLLM-70B",
}

DATASETS      = ["metabric", "seer"]
PROMPT_TYPES  = ["0shot", "5shot"]
TEMPERATURES  = {
    "conservateur": (0.3,  0.85, 1.1),
    "standard":     (0.5,  0.92, 1.1),
    "exploratoire": (0.8,  0.95, 1.05),
}

SLURM_HEADER = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=bbo@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=40
#SBATCH --time={time_limit}
#SBATCH --output={log_dir}/{job_name}.%j.out
#SBATCH --error={log_dir}/{job_name}.%j.err

module purge
module load pytorch-gpu/py3/2.7.0

export PYTHONPATH=$SCRATCH/pyuser:$PYTHONPATH
export HF_HOME=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== $(date) -- START {job_name} on $(hostname) ==="

cd {proj_dir}

python scripts/generate_cag.py \\
    --dataset       {dataset} \\
    --prompt_type   {prompt_type} \\
    --model         {model_id} \\
    --n             {n} \\
    --skip          {skip} \\
    --dtype         fp16 \\
    --temperature   {temperature} \\
    --top_p         {top_p} \\
    --repetition_penalty {rep_penalty} \\
    --out           {out_file}

code=$?
echo "=== $(date) -- END {job_name} (exit $code) ==="
exit $code
"""


def n_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def sbatch(script, dependency, dry_run):
    """Submit a SLURM script string, return the job ID (or '' in dry-run)."""
    import tempfile
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd += [f"--dependency=afterok:{dependency}"]
    if dry_run:
        print(f"  [dry-run] would submit: {cmd}")
        return "DRY"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp:
        tmp.write(script)
        tmp_path = tmp.name
    try:
        result = subprocess.run(cmd + [tmp_path], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed:\n{result.stderr}")
        job_id = result.stdout.strip()
    finally:
        os.unlink(tmp_path)
    return job_id


def cancel_pending_70b():
    """Annule tous les jobs pendants dont le nom contient apertus70b ou openbiollm70b."""
    result = subprocess.run(
        ["squeue", "-u", os.environ["USER"], "-h",
         "-o", "%i %j %t", "--state=PD"],
        capture_output=True, text=True
    )
    to_cancel = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            job_id, name = parts[0], parts[1]
            if "apertus70" in name or "openbiollm70" in name or "eval_ape" in name or "eval_ope" in name:
                to_cancel.append(job_id)
    if to_cancel:
        print(f"Annulation de {len(to_cancel)} jobs : {' '.join(to_cancel)}")
        subprocess.run(["scancel"] + to_cancel)
    else:
        print("Aucun job 70B pendant à annuler.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",    action="store_true")
    ap.add_argument("--cancel-old", action="store_true",
                    help="Annuler les jobs 70B pendants avant de soumettre")
    ap.add_argument("--model", choices=list(MODELS.keys()),
                    help="Soumettre uniquement ce modèle (défaut: tous)")
    ap.add_argument("--prompt-type", choices=PROMPT_TYPES,
                    help="Soumettre uniquement ce type de prompt (défaut: tous)")
    args = ap.parse_args()

    if args.cancel_old:
        cancel_pending_70b()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    models_to_run = {args.model: MODELS[args.model]} if args.model else MODELS
    prompt_types_to_run = [args.prompt_type] if args.prompt_type else PROMPT_TYPES
    all_final_ids = {k: [] for k in models_to_run}

    for short_name, model_id in models_to_run.items():
        for dataset in DATASETS:
            for prompt_type in prompt_types_to_run:
                for preset, (temp, top_p, rep_pen) in TEMPERATURES.items():
                    base_name = f"{short_name}_{dataset}_{prompt_type}_{preset}"
                    out_file  = OUT_DIR / f"{base_name}.jsonl"

                    # Vérifie si déjà terminé
                    done = n_lines(out_file)
                    if done >= N_TOTAL:
                        print(f"  SKIP {base_name} ({done} lignes déjà présentes)")
                        continue

                    # Calcule depuis quel patient continuer
                    start_skip = done  # reprend là où ça s'est arrêté

                    # Découpe en chunks
                    chunks = []
                    pos = start_skip
                    while pos < N_TOTAL:
                        n = min(CHUNK_SIZE, N_TOTAL - pos)
                        chunks.append((pos, n))
                        pos += n

                    print(f"\n  {base_name}: {len(chunks)} chunk(s) "
                          f"(patients {start_skip}→{N_TOTAL-1})")

                    dep = None
                    for c_idx, (skip, n) in enumerate(chunks):
                        job_name = f"{base_name}_c{c_idx}"
                        script = SLURM_HEADER.format(
                            job_name     = job_name,
                            time_limit   = TIME_LIMIT,
                            log_dir      = LOG_DIR,
                            proj_dir     = PROJ_DIR,
                            dataset      = dataset,
                            prompt_type  = prompt_type,
                            model_id     = model_id,
                            n            = n,
                            skip         = skip,
                            temperature  = temp,
                            top_p        = top_p,
                            rep_penalty  = rep_pen,
                            out_file     = out_file,
                        )
                        job_id = sbatch(script, dep, args.dry_run)
                        print(f"    chunk {c_idx} (skip={skip}, n={n}) → job {job_id}")
                        dep = job_id

                    if dep:
                        all_final_ids[short_name].append(dep)

    # Résumé des dépendances pour les jobs d'évaluation
    print("\n=== JOBS D'ÉVALUATION (à soumettre après génération) ===")
    for short_name, ids in all_final_ids.items():
        if ids:
            dep_str = ":".join(ids)
            print(f"  {short_name}: --dependency=afterok:{dep_str}")
            print(f"  (script: $SCRATCH/CAG_slurm/jobscripts/eval_{short_name}.sh)")


if __name__ == "__main__":
    main()
