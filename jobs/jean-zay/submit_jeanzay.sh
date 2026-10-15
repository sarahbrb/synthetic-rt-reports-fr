#!/bin/bash
# Submit all generation + evaluation jobs to Jean Zay.
# Each model's eval job starts only after ALL its generation jobs succeed.
# Usage: bash $SCRATCH/CAG_slurm/submit_jeanzay.sh

set -e

# ────────────────────────────────────────────────────────────
# apertus70b  (swiss-ai/Apertus-70B-Instruct-2509)
# ────────────────────────────────────────────────────────────
echo "=== Submitting apertus70b ==="
DEPS_APERTUS70B=''

echo '  Submitting apertus70b_metabric_0shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_metabric_0shot_conservateur.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_metabric_0shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_metabric_0shot_standard.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_metabric_0shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_metabric_0shot_exploratoire.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_metabric_5shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_metabric_5shot_conservateur.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_metabric_5shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_metabric_5shot_standard.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_metabric_5shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_metabric_5shot_exploratoire.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_seer_0shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_seer_0shot_conservateur.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_seer_0shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_seer_0shot_standard.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_seer_0shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_seer_0shot_exploratoire.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_seer_5shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_seer_5shot_conservateur.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_seer_5shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_seer_5shot_standard.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_seer_5shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_seer_5shot_exploratoire.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting eval_apertus70b (après tous les jobs de génération) ...'
JID=$(sbatch --parsable --dependency=afterok$DEPS_APERTUS70B /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/eval_apertus70b.sh)
echo "    → eval job $JID"

# ────────────────────────────────────────────────────────────
# medllama8b  (Henrychur/MMed-Llama-3-8B-EnIns)
# ────────────────────────────────────────────────────────────
echo "=== Submitting medllama8b ==="
DEPS_MEDLLAMA8B=''

echo '  Submitting medllama8b_metabric_0shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/medllama8b_metabric_0shot_conservateur.sh)
echo "    → job $JID"
DEPS_MEDLLAMA8B="${DEPS_MEDLLAMA8B}:$JID"

echo '  Submitting medllama8b_metabric_0shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/medllama8b_metabric_0shot_standard.sh)
echo "    → job $JID"
DEPS_MEDLLAMA8B="${DEPS_MEDLLAMA8B}:$JID"

echo '  Submitting medllama8b_metabric_0shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/medllama8b_metabric_0shot_exploratoire.sh)
echo "    → job $JID"
DEPS_MEDLLAMA8B="${DEPS_MEDLLAMA8B}:$JID"

echo '  Submitting medllama8b_metabric_5shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/medllama8b_metabric_5shot_conservateur.sh)
echo "    → job $JID"
DEPS_MEDLLAMA8B="${DEPS_MEDLLAMA8B}:$JID"

echo '  Submitting medllama8b_metabric_5shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/medllama8b_metabric_5shot_standard.sh)
echo "    → job $JID"
DEPS_MEDLLAMA8B="${DEPS_MEDLLAMA8B}:$JID"

echo '  Submitting medllama8b_metabric_5shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/medllama8b_metabric_5shot_exploratoire.sh)
echo "    → job $JID"
DEPS_MEDLLAMA8B="${DEPS_MEDLLAMA8B}:$JID"

echo '  Submitting medllama8b_seer_0shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/medllama8b_seer_0shot_conservateur.sh)
echo "    → job $JID"
DEPS_MEDLLAMA8B="${DEPS_MEDLLAMA8B}:$JID"

echo '  Submitting medllama8b_seer_0shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/medllama8b_seer_0shot_standard.sh)
echo "    → job $JID"
DEPS_MEDLLAMA8B="${DEPS_MEDLLAMA8B}:$JID"

echo '  Submitting medllama8b_seer_0shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/medllama8b_seer_0shot_exploratoire.sh)
echo "    → job $JID"
DEPS_MEDLLAMA8B="${DEPS_MEDLLAMA8B}:$JID"

echo '  Submitting medllama8b_seer_5shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/medllama8b_seer_5shot_conservateur.sh)
echo "    → job $JID"
DEPS_MEDLLAMA8B="${DEPS_MEDLLAMA8B}:$JID"

echo '  Submitting medllama8b_seer_5shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/medllama8b_seer_5shot_standard.sh)
echo "    → job $JID"
DEPS_MEDLLAMA8B="${DEPS_MEDLLAMA8B}:$JID"

echo '  Submitting medllama8b_seer_5shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/medllama8b_seer_5shot_exploratoire.sh)
echo "    → job $JID"
DEPS_MEDLLAMA8B="${DEPS_MEDLLAMA8B}:$JID"

echo '  Submitting eval_medllama8b (après tous les jobs de génération) ...'
JID=$(sbatch --parsable --dependency=afterok$DEPS_MEDLLAMA8B /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/eval_medllama8b.sh)
echo "    → eval job $JID"

# ────────────────────────────────────────────────────────────
# openbiollm8b  (aaditya/Llama3-OpenBioLLM-8B)
# ────────────────────────────────────────────────────────────
echo "=== Submitting openbiollm8b ==="
DEPS_OPENBIOLLM8B=''

echo '  Submitting openbiollm8b_metabric_0shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm8b_metabric_0shot_conservateur.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM8B="${DEPS_OPENBIOLLM8B}:$JID"

echo '  Submitting openbiollm8b_metabric_0shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm8b_metabric_0shot_standard.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM8B="${DEPS_OPENBIOLLM8B}:$JID"

echo '  Submitting openbiollm8b_metabric_0shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm8b_metabric_0shot_exploratoire.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM8B="${DEPS_OPENBIOLLM8B}:$JID"

echo '  Submitting openbiollm8b_metabric_5shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm8b_metabric_5shot_conservateur.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM8B="${DEPS_OPENBIOLLM8B}:$JID"

echo '  Submitting openbiollm8b_metabric_5shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm8b_metabric_5shot_standard.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM8B="${DEPS_OPENBIOLLM8B}:$JID"

echo '  Submitting openbiollm8b_metabric_5shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm8b_metabric_5shot_exploratoire.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM8B="${DEPS_OPENBIOLLM8B}:$JID"

echo '  Submitting openbiollm8b_seer_0shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm8b_seer_0shot_conservateur.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM8B="${DEPS_OPENBIOLLM8B}:$JID"

echo '  Submitting openbiollm8b_seer_0shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm8b_seer_0shot_standard.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM8B="${DEPS_OPENBIOLLM8B}:$JID"

echo '  Submitting openbiollm8b_seer_0shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm8b_seer_0shot_exploratoire.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM8B="${DEPS_OPENBIOLLM8B}:$JID"

echo '  Submitting openbiollm8b_seer_5shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm8b_seer_5shot_conservateur.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM8B="${DEPS_OPENBIOLLM8B}:$JID"

echo '  Submitting openbiollm8b_seer_5shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm8b_seer_5shot_standard.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM8B="${DEPS_OPENBIOLLM8B}:$JID"

echo '  Submitting openbiollm8b_seer_5shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm8b_seer_5shot_exploratoire.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM8B="${DEPS_OPENBIOLLM8B}:$JID"

echo '  Submitting eval_openbiollm8b (après tous les jobs de génération) ...'
JID=$(sbatch --parsable --dependency=afterok$DEPS_OPENBIOLLM8B /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/eval_openbiollm8b.sh)
echo "    → eval job $JID"

# ────────────────────────────────────────────────────────────
# openbiollm70b  (aaditya/Llama3-OpenBioLLM-70B)
# ────────────────────────────────────────────────────────────
echo "=== Submitting openbiollm70b ==="
DEPS_OPENBIOLLM70B=''

echo '  Submitting openbiollm70b_metabric_0shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_metabric_0shot_conservateur.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_metabric_0shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_metabric_0shot_standard.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_metabric_0shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_metabric_0shot_exploratoire.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_metabric_5shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_metabric_5shot_conservateur.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_metabric_5shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_metabric_5shot_standard.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_metabric_5shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_metabric_5shot_exploratoire.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_seer_0shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_seer_0shot_conservateur.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_seer_0shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_seer_0shot_standard.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_seer_0shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_seer_0shot_exploratoire.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_seer_5shot_conservateur ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_seer_5shot_conservateur.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_seer_5shot_standard ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_seer_5shot_standard.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_seer_5shot_exploratoire ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_seer_5shot_exploratoire.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting eval_openbiollm70b (après tous les jobs de génération) ...'
JID=$(sbatch --parsable --dependency=afterok$DEPS_OPENBIOLLM70B /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/eval_openbiollm70b.sh)
echo "    → eval job $JID"

echo ""
echo "All 48 generation + 4 evaluation jobs submitted."
echo "Monitor with: squeue -u $USER"
echo "Outputs: /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_outputs/"
