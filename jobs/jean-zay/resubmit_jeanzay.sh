#!/bin/bash
# Relance intelligente : continuation pour partiels, fresh pour vides/manquants.
# Chaque ligne ci-dessous correspond à un job utile.
# Lancer APRÈS : scancel -u $USER

set -e

# ────────────────────────────────────────────────────────────
# apertus70b
# ────────────────────────────────────────────────────────────
echo "=== apertus70b ==="
DEPS_APERTUS70B=''

echo '  Submitting apertus70b_metabric_0shot_conservateur (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_metabric_0shot_conservateur.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_metabric_0shot_standard (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_metabric_0shot_standard.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_metabric_0shot_exploratoire (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_metabric_0shot_exploratoire.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_metabric_5shot_conservateur (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_metabric_5shot_conservateur.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_metabric_5shot_standard (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_metabric_5shot_standard.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_metabric_5shot_exploratoire (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_metabric_5shot_exploratoire.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_seer_0shot_conservateur (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_seer_0shot_conservateur.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_seer_0shot_standard (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_seer_0shot_standard.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_seer_0shot_exploratoire (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_seer_0shot_exploratoire.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_seer_5shot_conservateur (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_seer_5shot_conservateur.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_seer_5shot_standard (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_seer_5shot_standard.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

echo '  Submitting apertus70b_seer_5shot_exploratoire (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/apertus70b_seer_5shot_exploratoire.sh)
echo "    → job $JID"
DEPS_APERTUS70B="${DEPS_APERTUS70B}:$JID"

if [ -n "$DEPS_APERTUS70B" ]; then
  echo '  Submitting eval_apertus70b ...'
  JID=$(sbatch --parsable --dependency=afterok$DEPS_APERTUS70B /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/eval_apertus70b.sh)
  echo "    → eval job $JID"
fi

# ────────────────────────────────────────────────────────────
# medllama8b
# ────────────────────────────────────────────────────────────
echo "=== medllama8b ==="
DEPS_MEDLLAMA8B=''

# SKIP medllama8b_metabric_0shot_conservateur (already 700/700 lines)

# SKIP medllama8b_metabric_0shot_standard (already 700/700 lines)

# SKIP medllama8b_metabric_0shot_exploratoire (already 700/700 lines)

# SKIP medllama8b_metabric_5shot_conservateur (already 700/700 lines)

# SKIP medllama8b_metabric_5shot_standard (already 700/700 lines)

# SKIP medllama8b_metabric_5shot_exploratoire (already 700/700 lines)

# SKIP medllama8b_seer_0shot_conservateur (already 700/700 lines)

# SKIP medllama8b_seer_0shot_standard (already 700/700 lines)

# SKIP medllama8b_seer_0shot_exploratoire (already 700/700 lines)

# SKIP medllama8b_seer_5shot_conservateur (already 700/700 lines)

# SKIP medllama8b_seer_5shot_standard (already 700/700 lines)

# SKIP medllama8b_seer_5shot_exploratoire (already 700/700 lines)

if [ -n "$DEPS_MEDLLAMA8B" ]; then
  echo '  Submitting eval_medllama8b ...'
  JID=$(sbatch --parsable --dependency=afterok$DEPS_MEDLLAMA8B /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/eval_medllama8b.sh)
  echo "    → eval job $JID"
fi

# ────────────────────────────────────────────────────────────
# openbiollm8b
# ────────────────────────────────────────────────────────────
echo "=== openbiollm8b ==="
DEPS_OPENBIOLLM8B=''

# SKIP openbiollm8b_metabric_0shot_conservateur (already 700/700 lines)

# SKIP openbiollm8b_metabric_0shot_standard (already 700/700 lines)

# SKIP openbiollm8b_metabric_0shot_exploratoire (already 700/700 lines)

# SKIP openbiollm8b_metabric_5shot_conservateur (already 700/700 lines)

# SKIP openbiollm8b_metabric_5shot_standard (already 700/700 lines)

# SKIP openbiollm8b_metabric_5shot_exploratoire (already 700/700 lines)

# SKIP openbiollm8b_seer_0shot_conservateur (already 700/700 lines)

# SKIP openbiollm8b_seer_0shot_standard (already 700/700 lines)

# SKIP openbiollm8b_seer_0shot_exploratoire (already 700/700 lines)

# SKIP openbiollm8b_seer_5shot_conservateur (already 700/700 lines)

# SKIP openbiollm8b_seer_5shot_standard (already 700/700 lines)

# SKIP openbiollm8b_seer_5shot_exploratoire (already 700/700 lines)

if [ -n "$DEPS_OPENBIOLLM8B" ]; then
  echo '  Submitting eval_openbiollm8b ...'
  JID=$(sbatch --parsable --dependency=afterok$DEPS_OPENBIOLLM8B /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/eval_openbiollm8b.sh)
  echo "    → eval job $JID"
fi

# ────────────────────────────────────────────────────────────
# openbiollm70b
# ────────────────────────────────────────────────────────────
echo "=== openbiollm70b ==="
DEPS_OPENBIOLLM70B=''

# SKIP openbiollm70b_metabric_0shot_conservateur (already 700/700 lines)

# SKIP openbiollm70b_metabric_0shot_standard (already 700/700 lines)

# SKIP openbiollm70b_metabric_0shot_exploratoire (already 700/700 lines)

echo '  Submitting openbiollm70b_metabric_5shot_conservateur (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_metabric_5shot_conservateur.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_metabric_5shot_standard (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_metabric_5shot_standard.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_metabric_5shot_exploratoire (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_metabric_5shot_exploratoire.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_seer_0shot_conservateur (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_seer_0shot_conservateur.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_seer_0shot_standard (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_seer_0shot_standard.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_seer_0shot_exploratoire (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_seer_0shot_exploratoire.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_seer_5shot_conservateur (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_seer_5shot_conservateur.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_seer_5shot_standard (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_seer_5shot_standard.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

echo '  Submitting openbiollm70b_seer_5shot_exploratoire (skip=0, n=700) ...'
JID=$(sbatch --parsable /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/openbiollm70b_seer_5shot_exploratoire.sh)
echo "    → job $JID"
DEPS_OPENBIOLLM70B="${DEPS_OPENBIOLLM70B}:$JID"

if [ -n "$DEPS_OPENBIOLLM70B" ]; then
  echo '  Submitting eval_openbiollm70b ...'
  JID=$(sbatch --parsable --dependency=afterok$DEPS_OPENBIOLLM70B /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_slurm/jobscripts/eval_openbiollm70b.sh)
  echo "    → eval job $JID"
fi

echo ""
echo "Submitted 21 generation jobs (27 already complete, skipped)."
echo "Monitor: squeue -u $USER"
echo "Outputs: /lustre/fsn1/projects/rech/bbo/uuf58kz/CAG_outputs/"
