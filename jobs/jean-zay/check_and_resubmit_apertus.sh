#!/bin/bash
# Vérifier que transformers 4.57.6 supporte apertus, puis resoumettre les 12 jobs.
# Lancer depuis le nœud de login APRÈS download_eval_models.sh

set -e

export HF_HOME=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$SCRATCH/pyuser:$PYTHONPATH

echo "=== Test chargement config apertus ==="
python3 -c "
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = os.environ['SCRATCH'] + '/hf_cache'
from transformers import AutoConfig
c = AutoConfig.from_pretrained('swiss-ai/Apertus-70B-Instruct-2509', trust_remote_code=True)
print('OK — model_type:', c.model_type)
"

echo ""
echo "=== Soumission des 12 jobs apertus70b ==="

JOBS_DIR=$SCRATCH/CAG_slurm/jobscripts
LOGS_DIR=$SCRATCH/CAG_slurm/logs
DEPS=""

for script in $JOBS_DIR/apertus70b_*.sh; do
    [ -f "$script" ] || continue
    name=$(basename "$script" .sh)
    echo "  Submitting $name ..."
    JID=$(sbatch --parsable "$script")
    echo "    → job $JID"
    DEPS="${DEPS}:$JID"
done

echo ""
echo "=== Soumission eval_apertus70b (après tous les jobs de génération) ==="
JID=$(sbatch --parsable --dependency=afterok${DEPS} $JOBS_DIR/eval_apertus70b.sh)
echo "  → eval job $JID"

echo ""
echo "Terminé. Surveiller avec : squeue -u \$USER"
