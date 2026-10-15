#!/bin/bash
# Submit BERTScore recomputation jobs — only for models with missing encoder columns.
#
# Reads jobs/bertscore_config.sh to know which encoders are expected.
# Checks each model's summary CSV; submits a job only if at least one
# encoder column is absent.
#
# Usage (from aker.imag.fr, project root):
#   bash jobs/submit_bertscore.sh
#
# To add a new encoder: edit jobs/bertscore_config.sh, then re-run this script.

set -euo pipefail
cd "$(dirname "$0")/.."          # run from CAG/ project root
source jobs/bertscore_config.sh

JOBS=logs/jobscripts
LOGS=logs/oar
MODELS=(qwen7b qwen32b mistral7b mistral12b mistral24b apertus8b medgemma27b)

# Build list of expected column names (col_mean) from config
NEEDED_COLS=()
for entry in "${BERTSCORE_ENCODERS[@]}"; do
    col="${entry%%|*}"
    NEEDED_COLS+=("${col}_mean")
done

submitted=0
skipped=0

for short in "${MODELS[@]}"; do
    summary="outputs/${short}_eval_summary.csv"

    if [ ! -f "$summary" ]; then
        echo "[skip] $short — $summary not found (generation/eval not complete)"
        (( skipped++ )) || true
        continue
    fi

    # Check if any expected column is missing from the summary header
    header=$(head -1 "$summary")
    missing_any=false
    for col in "${NEEDED_COLS[@]}"; do
        if ! echo "$header" | tr ',' '\n' | grep -qx "$col"; then
            missing_any=true
            echo "  → $short: missing $col"
        fi
    done

    if $missing_any; then
        echo "Submitting bertscore_${short}..."
        oarsub \
            -l /gpu=1,walltime=06:00:00 \
            -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" \
            -q default \
            -O "$LOGS/bertscore_${short}.stdout" \
            -E "$LOGS/bertscore_${short}.stderr" \
            "$JOBS/bertscore_${short}.sh"
        (( submitted++ )) || true
    else
        echo "[ok]   $short — all encoder columns present"
        (( skipped++ )) || true
    fi
done

echo ""
echo "Done: $submitted job(s) submitted, $skipped model(s) skipped."
