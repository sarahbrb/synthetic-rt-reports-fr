#!/bin/bash
# Télécharge les 4 modèles CAG dans $SCRATCH/synth-data-generation/hf_cache.
# Soumettre avec : bash $SCRATCH/CAG_slurm/download_models.sh
# Chaque modèle est un job SLURM indépendant (qos_gpu-dev, 2h max).
#
# Modèles :
#   apertus70b    swiss-ai/Apertus-70B-Instruct-2509       ~140 GB
#   medllama8b    Henrychur/MMed-Llama-3-8B-EnIns          ~16 GB
#   openbiollm8b  aaditya/Llama3-OpenBioLLM-8B             ~16 GB
#   openbiollm70b aaditya/Llama3-OpenBioLLM-70B            ~140 GB

set -e

LOGS_DIR="$SCRATCH/CAG_slurm/logs"
mkdir -p "$LOGS_DIR"

declare -A MODELS
MODELS["apertus70b"]="swiss-ai/Apertus-70B-Instruct-2509"
MODELS["medllama8b"]="Henrychur/MMed-Llama-3-8B-EnIns"
MODELS["openbiollm8b"]="aaditya/Llama3-OpenBioLLM-8B"
MODELS["openbiollm70b"]="aaditya/Llama3-OpenBioLLM-70B"

for MODEL_KEY in apertus70b medllama8b openbiollm8b openbiollm70b; do
    MODEL_ID="${MODELS[$MODEL_KEY]}"
    echo "Submitting download job for $MODEL_KEY ($MODEL_ID) ..."

    JID=$(sbatch --parsable \
        --job-name="dl_${MODEL_KEY}" \
        --account=bbo@v100 \
        --partition=gpu_p13 \
        --qos=qos_gpu-dev \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --time=02:00:00 \
        --output="$LOGS_DIR/download_${MODEL_KEY}.%j.out" \
        --error="$LOGS_DIR/download_${MODEL_KEY}.%j.err" \
        --wrap="
module purge
module load pytorch-gpu/py3/2.7.0
export PYTHONPATH=\$SCRATCH/synth-data-generation/pyuser/lib/python3.12/site-packages:\$PYTHONPATH
export HF_HOME=\$SCRATCH/synth-data-generation/hf_cache
export HUGGINGFACE_HUB_CACHE=\$SCRATCH/synth-data-generation/hf_cache
export TRANSFORMERS_CACHE=\$SCRATCH/synth-data-generation/hf_cache

echo \"=== \$(date) -- Téléchargement de $MODEL_ID ===\"

python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
import os
model_id  = '$MODEL_ID'
cache_dir = os.environ['HF_HOME']
print(f'Downloading {model_id} to {cache_dir} ...')
path = snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    ignore_patterns=['*.pt', 'original/*'],
)
print(f'Done: {path}')
PYEOF

echo \"=== \$(date) -- Terminé (exit \$?) ===\"
")
    echo "  → job $JID  (logs: $LOGS_DIR/download_${MODEL_KEY}.${JID}.out)"
done

echo ""
echo "Surveiller avec : squeue -u \$USER"
