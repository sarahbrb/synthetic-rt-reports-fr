#!/bin/bash
# Corrige les dépendances de transformers 4.57.6 et télécharge les modèles d'évaluation.
# Lancer depuis le nœud de login dans un screen :
#   screen -S dl_eval
#   bash $SCRATCH/CAG_slurm/download_eval_models.sh

set -e

PYTHON=/lustre/fshomisc/sup/hpe/pub/miniforge/24.11.3/envs/pytorch-gpu-2.7.0+py3.12.10/bin/python3
PIP="$PYTHON -m pip"

export HF_HOME=$SCRATCH/synth-data-generation/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/synth-data-generation/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

# Utiliser le token HF stocké localement (créé lors de huggingface-cli login)
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [ -f "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
    export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
    echo "Token HF trouvé et chargé."
else
    echo "ATTENTION : token HF non trouvé dans $HF_TOKEN_FILE"
    echo "Lance 'huggingface-cli login' si des modèles échouent."
fi

echo ""
echo "=== Étape 1 : correction des dépendances de transformers 4.57.6 ==="
# transformers 4.57.6 exige huggingface-hub>=0.34.0,<1.0 ; la version 1.x rompt l'import
rm -rf $SCRATCH/synth-data-generation/pyuser/huggingface_hub $SCRATCH/synth-data-generation/pyuser/huggingface_hub-*.dist-info 2>/dev/null || true
$PIP install --target=$SCRATCH/synth-data-generation/pyuser "huggingface_hub>=0.34.0,<1.0" --index-url https://pypi.org/simple
$PIP install --target=$SCRATCH/synth-data-generation/pyuser --no-deps "tokenizers==0.22.0"
rm -rf $SCRATCH/synth-data-generation/pyuser/tokenizers-0.21*.dist-info 2>/dev/null || true
echo "OK"

echo ""
echo "=== Étape 2 : téléchargement des modèles d'évaluation ==="
PYTHONPATH=$SCRATCH/synth-data-generation/pyuser $PYTHON - <<'PYEOF'
import os, sys

os.environ['HF_HOME'] = os.environ['SCRATCH'] + '/hf_cache'
cache = os.environ['HF_HOME']

from huggingface_hub import snapshot_download

models = [
    "asi/gpt-fr-cased-base",
    "camembert-base",
    "almanach/camembert-bio-base",
    "Dr-BERT/DrBERT-7GB",
]

failed = []
for m in models:
    print(f"\n>>> {m}")
    try:
        path = snapshot_download(
            repo_id=m,
            cache_dir=cache,
            ignore_patterns=["*.pt", "original/*", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"    OK → {path}")
    except Exception as e:
        print(f"    ERREUR : {e}")
        failed.append(m)

print()
if failed:
    print(f"ATTENTION : {len(failed)} modèle(s) non téléchargé(s) : {failed}")
    sys.exit(1)
else:
    print("Tous les modèles téléchargés avec succès.")
PYEOF

echo ""
echo "=== Terminé ==="
echo "Lance maintenant : bash $SCRATCH/CAG_slurm/resubmit_after_fix.sh"
