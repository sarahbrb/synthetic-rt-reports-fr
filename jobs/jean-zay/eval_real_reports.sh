#!/bin/bash
#SBATCH --job-name=eval_real_reports
#SBATCH --account=bbo@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/logs/eval_real_reports.%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/logs/eval_real_reports.%j.err

module purge
module load pytorch-gpu/py3/2.7.0

# Pas de pyuser : textstat remplacé par implémentation native dans le script
export HF_HOME=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== $(date) -- START eval_real_reports on $(hostname) ==="

cd /lustre/fshomisc/home/rech/genxai01/uuf58kz/synthetic-rt-reports-fr

python scripts/evaluate_real_reports.py \
    --input_dir real_data/ \
    --output    real_data/results_real.csv \
    --lm_model  asi/gpt-fr-cased-base \
    --device    cuda

code=$?
echo "=== $(date) -- END eval_real_reports (exit $code) ==="
