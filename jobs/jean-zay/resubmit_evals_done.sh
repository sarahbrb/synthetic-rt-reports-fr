#!/bin/bash
# Resoumettre les evals pour les modèles dont la génération est terminée.
# medllama8b : 12/12 fichiers complets
# openbiollm8b : 12/12 fichiers complets
set -e
JOBS_DIR=$SCRATCH/CAG_slurm/jobscripts

echo "Submitting eval_medllama8b ..."
JID=$(sbatch --parsable $JOBS_DIR/eval_medllama8b.sh)
echo "  → job $JID"

echo "Submitting eval_openbiollm8b ..."
JID=$(sbatch --parsable $JOBS_DIR/eval_openbiollm8b.sh)
echo "  → job $JID"

echo "Terminé. squeue -u \$USER"
