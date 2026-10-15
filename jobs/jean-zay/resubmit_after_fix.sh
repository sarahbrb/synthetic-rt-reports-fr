#!/bin/bash
# Resoumission de tous les jobs qui ont échoué après le fix des dépendances.
# Lancer APRÈS download_eval_models.sh (qui corrige tokenizers + huggingface_hub).
#
# Ce script relance :
#   - openbiollm70b  : 9 combinaisons manquantes (seer + 5shot)
#   - apertus70b     : 12 combinaisons (0 fichiers générés)
#   - eval_medllama8b    : génération complète, modèles d'eval maintenant en cache
#   - eval_openbiollm8b  : idem
#   - eval_openbiollm70b : dépend des 9 nouveaux jobs
#   - eval_apertus70b    : dépend des 12 nouveaux jobs apertus

set -e

JOBS_DIR=$SCRATCH/CAG_slurm/jobscripts

echo "=== Annulation des vieux jobs eval bloqués ==="
for jid in $(squeue -u $USER -h -o "%i %j" | grep "eval_" | awk '{print $1}'); do
    echo "  Cancelling job $jid"
    scancel $jid
done

echo ""
echo "=== openbiollm70b — 9 combinaisons manquantes ==="
DEPS_70B=""
for combo in seer_0shot_conservateur seer_0shot_standard seer_0shot_exploratoire \
             metabric_5shot_conservateur metabric_5shot_standard metabric_5shot_exploratoire \
             seer_5shot_conservateur seer_5shot_standard seer_5shot_exploratoire; do
    script="$JOBS_DIR/openbiollm70b_${combo}.sh"
    if [ -f "$script" ]; then
        echo "  Submitting openbiollm70b_${combo} ..."
        JID=$(sbatch --parsable "$script")
        echo "    → job $JID"
        DEPS_70B="${DEPS_70B}:$JID"
    else
        echo "  AVERTISSEMENT: script introuvable : $script"
    fi
done

echo ""
echo "=== apertus70b — 12 combinaisons ==="
DEPS_APERTUS=""
for script in $JOBS_DIR/apertus70b_*.sh; do
    [ -f "$script" ] || continue
    name=$(basename "$script" .sh)
    echo "  Submitting $name ..."
    JID=$(sbatch --parsable "$script")
    echo "    → job $JID"
    DEPS_APERTUS="${DEPS_APERTUS}:$JID"
done

echo ""
echo "=== Evals medllama8b et openbiollm8b (génération déjà complète) ==="
JID=$(sbatch --parsable $JOBS_DIR/eval_medllama8b.sh)
echo "  eval_medllama8b → job $JID"

JID=$(sbatch --parsable $JOBS_DIR/eval_openbiollm8b.sh)
echo "  eval_openbiollm8b → job $JID"

echo ""
echo "=== eval_openbiollm70b (après les 9 nouveaux jobs) ==="
if [ -n "$DEPS_70B" ]; then
    JID=$(sbatch --parsable --dependency=afterok${DEPS_70B} $JOBS_DIR/eval_openbiollm70b.sh)
    echo "  eval_openbiollm70b → job $JID"
fi

echo ""
echo "=== eval_apertus70b (après les 12 jobs apertus) ==="
if [ -n "$DEPS_APERTUS" ]; then
    JID=$(sbatch --parsable --dependency=afterok${DEPS_APERTUS} $JOBS_DIR/eval_apertus70b.sh)
    echo "  eval_apertus70b → job $JID"
fi

echo ""
echo "=== Terminé. Surveiller avec : squeue -u \$USER ==="
