#!/bin/bash
# OAR submission: generation + automatic evaluation per model
# Each model's evaluation job starts only after all its generation jobs
# have completed successfully (OAR dependencies via -a <job_id>)

# Helper: submit a job and return its OAR job ID
submit() {
    local jid
    jid=$("$@" 2>&1 | grep OAR_JOB_ID | cut -d= -f2)
    if [ -z "$jid" ]; then
        echo "[ERROR] oarsub failed: $*" >&2
        exit 1
    fi
    echo "$jid"
}

# ────────────────────────────────────────────────────────────
# qwen7b  (Qwen/Qwen2.5-7B-Instruct)
# lig-gpu8.imag.fr / lig-gpu10.imag.fr  1 GPU  N=700
# ────────────────────────────────────────────────────────────
echo "=== Submitting qwen7b ==="
DEPS_QWEN7B=""

JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_metabric_0shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_metabric_0shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen7b_metabric_0shot_conservateur.sh)
echo "  qwen7b_metabric_0shot_conservateur -> JID=$JID"
DEPS_QWEN7B="$DEPS_QWEN7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_metabric_0shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_metabric_0shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen7b_metabric_0shot_standard.sh)
echo "  qwen7b_metabric_0shot_standard -> JID=$JID"
DEPS_QWEN7B="$DEPS_QWEN7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_metabric_0shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_metabric_0shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen7b_metabric_0shot_exploratoire.sh)
echo "  qwen7b_metabric_0shot_exploratoire -> JID=$JID"
DEPS_QWEN7B="$DEPS_QWEN7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_metabric_5shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_metabric_5shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen7b_metabric_5shot_conservateur.sh)
echo "  qwen7b_metabric_5shot_conservateur -> JID=$JID"
DEPS_QWEN7B="$DEPS_QWEN7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_metabric_5shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_metabric_5shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen7b_metabric_5shot_standard.sh)
echo "  qwen7b_metabric_5shot_standard -> JID=$JID"
DEPS_QWEN7B="$DEPS_QWEN7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_metabric_5shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_metabric_5shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen7b_metabric_5shot_exploratoire.sh)
echo "  qwen7b_metabric_5shot_exploratoire -> JID=$JID"
DEPS_QWEN7B="$DEPS_QWEN7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_seer_0shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_seer_0shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen7b_seer_0shot_conservateur.sh)
echo "  qwen7b_seer_0shot_conservateur -> JID=$JID"
DEPS_QWEN7B="$DEPS_QWEN7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_seer_0shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_seer_0shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen7b_seer_0shot_standard.sh)
echo "  qwen7b_seer_0shot_standard -> JID=$JID"
DEPS_QWEN7B="$DEPS_QWEN7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_seer_0shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_seer_0shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen7b_seer_0shot_exploratoire.sh)
echo "  qwen7b_seer_0shot_exploratoire -> JID=$JID"
DEPS_QWEN7B="$DEPS_QWEN7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_seer_5shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_seer_5shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen7b_seer_5shot_conservateur.sh)
echo "  qwen7b_seer_5shot_conservateur -> JID=$JID"
DEPS_QWEN7B="$DEPS_QWEN7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_seer_5shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_seer_5shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen7b_seer_5shot_standard.sh)
echo "  qwen7b_seer_5shot_standard -> JID=$JID"
DEPS_QWEN7B="$DEPS_QWEN7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_seer_5shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen7b_seer_5shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen7b_seer_5shot_exploratoire.sh)
echo "  qwen7b_seer_5shot_exploratoire -> JID=$JID"
DEPS_QWEN7B="$DEPS_QWEN7B -a $JID"

echo "  -> submitting evaluation qwen7b (after all generation jobs)"
JID=$(submit oarsub $DEPS_QWEN7B -l /gpu=2,walltime=08:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/eval_qwen7b.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/eval_qwen7b.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/eval_qwen7b.sh)
echo "  eval_qwen7b -> JID=$JID"

# ────────────────────────────────────────────────────────────
# qwen32b  (Qwen/Qwen2.5-32B-Instruct)
# lig-gpu8.imag.fr / lig-gpu10.imag.fr  2 GPU  N=700
# ────────────────────────────────────────────────────────────
echo "=== Submitting qwen32b ==="
DEPS_QWEN32B=""

JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_metabric_0shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_metabric_0shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen32b_metabric_0shot_conservateur.sh)
echo "  qwen32b_metabric_0shot_conservateur -> JID=$JID"
DEPS_QWEN32B="$DEPS_QWEN32B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_metabric_0shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_metabric_0shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen32b_metabric_0shot_standard.sh)
echo "  qwen32b_metabric_0shot_standard -> JID=$JID"
DEPS_QWEN32B="$DEPS_QWEN32B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_metabric_0shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_metabric_0shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen32b_metabric_0shot_exploratoire.sh)
echo "  qwen32b_metabric_0shot_exploratoire -> JID=$JID"
DEPS_QWEN32B="$DEPS_QWEN32B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_metabric_5shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_metabric_5shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen32b_metabric_5shot_conservateur.sh)
echo "  qwen32b_metabric_5shot_conservateur -> JID=$JID"
DEPS_QWEN32B="$DEPS_QWEN32B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_metabric_5shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_metabric_5shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen32b_metabric_5shot_standard.sh)
echo "  qwen32b_metabric_5shot_standard -> JID=$JID"
DEPS_QWEN32B="$DEPS_QWEN32B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_metabric_5shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_metabric_5shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen32b_metabric_5shot_exploratoire.sh)
echo "  qwen32b_metabric_5shot_exploratoire -> JID=$JID"
DEPS_QWEN32B="$DEPS_QWEN32B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_seer_0shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_seer_0shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen32b_seer_0shot_conservateur.sh)
echo "  qwen32b_seer_0shot_conservateur -> JID=$JID"
DEPS_QWEN32B="$DEPS_QWEN32B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_seer_0shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_seer_0shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen32b_seer_0shot_standard.sh)
echo "  qwen32b_seer_0shot_standard -> JID=$JID"
DEPS_QWEN32B="$DEPS_QWEN32B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_seer_0shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_seer_0shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen32b_seer_0shot_exploratoire.sh)
echo "  qwen32b_seer_0shot_exploratoire -> JID=$JID"
DEPS_QWEN32B="$DEPS_QWEN32B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_seer_5shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_seer_5shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen32b_seer_5shot_conservateur.sh)
echo "  qwen32b_seer_5shot_conservateur -> JID=$JID"
DEPS_QWEN32B="$DEPS_QWEN32B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_seer_5shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_seer_5shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen32b_seer_5shot_standard.sh)
echo "  qwen32b_seer_5shot_standard -> JID=$JID"
DEPS_QWEN32B="$DEPS_QWEN32B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_seer_5shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/qwen32b_seer_5shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/qwen32b_seer_5shot_exploratoire.sh)
echo "  qwen32b_seer_5shot_exploratoire -> JID=$JID"
DEPS_QWEN32B="$DEPS_QWEN32B -a $JID"

echo "  -> submitting evaluation qwen32b (after all generation jobs)"
JID=$(submit oarsub $DEPS_QWEN32B -l /gpu=2,walltime=08:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/eval_qwen32b.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/eval_qwen32b.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/eval_qwen32b.sh)
echo "  eval_qwen32b -> JID=$JID"

# ────────────────────────────────────────────────────────────
# mistral7b  (mistralai/Mistral-7B-Instruct-v0.3)
# lig-gpu8.imag.fr / lig-gpu10.imag.fr  1 GPU  N=700
# ────────────────────────────────────────────────────────────
echo "=== Submitting mistral7b ==="
DEPS_MISTRAL7B=""

JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_metabric_0shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_metabric_0shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral7b_metabric_0shot_conservateur.sh)
echo "  mistral7b_metabric_0shot_conservateur -> JID=$JID"
DEPS_MISTRAL7B="$DEPS_MISTRAL7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_metabric_0shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_metabric_0shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral7b_metabric_0shot_standard.sh)
echo "  mistral7b_metabric_0shot_standard -> JID=$JID"
DEPS_MISTRAL7B="$DEPS_MISTRAL7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_metabric_0shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_metabric_0shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral7b_metabric_0shot_exploratoire.sh)
echo "  mistral7b_metabric_0shot_exploratoire -> JID=$JID"
DEPS_MISTRAL7B="$DEPS_MISTRAL7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_metabric_5shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_metabric_5shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral7b_metabric_5shot_conservateur.sh)
echo "  mistral7b_metabric_5shot_conservateur -> JID=$JID"
DEPS_MISTRAL7B="$DEPS_MISTRAL7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_metabric_5shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_metabric_5shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral7b_metabric_5shot_standard.sh)
echo "  mistral7b_metabric_5shot_standard -> JID=$JID"
DEPS_MISTRAL7B="$DEPS_MISTRAL7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_metabric_5shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_metabric_5shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral7b_metabric_5shot_exploratoire.sh)
echo "  mistral7b_metabric_5shot_exploratoire -> JID=$JID"
DEPS_MISTRAL7B="$DEPS_MISTRAL7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_seer_0shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_seer_0shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral7b_seer_0shot_conservateur.sh)
echo "  mistral7b_seer_0shot_conservateur -> JID=$JID"
DEPS_MISTRAL7B="$DEPS_MISTRAL7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_seer_0shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_seer_0shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral7b_seer_0shot_standard.sh)
echo "  mistral7b_seer_0shot_standard -> JID=$JID"
DEPS_MISTRAL7B="$DEPS_MISTRAL7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_seer_0shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_seer_0shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral7b_seer_0shot_exploratoire.sh)
echo "  mistral7b_seer_0shot_exploratoire -> JID=$JID"
DEPS_MISTRAL7B="$DEPS_MISTRAL7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_seer_5shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_seer_5shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral7b_seer_5shot_conservateur.sh)
echo "  mistral7b_seer_5shot_conservateur -> JID=$JID"
DEPS_MISTRAL7B="$DEPS_MISTRAL7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_seer_5shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_seer_5shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral7b_seer_5shot_standard.sh)
echo "  mistral7b_seer_5shot_standard -> JID=$JID"
DEPS_MISTRAL7B="$DEPS_MISTRAL7B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_seer_5shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral7b_seer_5shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral7b_seer_5shot_exploratoire.sh)
echo "  mistral7b_seer_5shot_exploratoire -> JID=$JID"
DEPS_MISTRAL7B="$DEPS_MISTRAL7B -a $JID"

echo "  -> submitting evaluation mistral7b (after all generation jobs)"
JID=$(submit oarsub $DEPS_MISTRAL7B -l /gpu=2,walltime=08:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/eval_mistral7b.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/eval_mistral7b.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/eval_mistral7b.sh)
echo "  eval_mistral7b -> JID=$JID"

# ────────────────────────────────────────────────────────────
# mistral12b  (mistralai/Mistral-Nemo-Instruct-2407)
# lig-gpu8.imag.fr / lig-gpu10.imag.fr  1 GPU  N=700
# ────────────────────────────────────────────────────────────
echo "=== Submitting mistral12b ==="
DEPS_MISTRAL12B=""

JID=$(submit oarsub -l /gpu=1,walltime=45:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_metabric_0shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_metabric_0shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral12b_metabric_0shot_conservateur.sh)
echo "  mistral12b_metabric_0shot_conservateur -> JID=$JID"
DEPS_MISTRAL12B="$DEPS_MISTRAL12B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=45:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_metabric_0shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_metabric_0shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral12b_metabric_0shot_standard.sh)
echo "  mistral12b_metabric_0shot_standard -> JID=$JID"
DEPS_MISTRAL12B="$DEPS_MISTRAL12B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=45:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_metabric_0shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_metabric_0shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral12b_metabric_0shot_exploratoire.sh)
echo "  mistral12b_metabric_0shot_exploratoire -> JID=$JID"
DEPS_MISTRAL12B="$DEPS_MISTRAL12B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=45:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_metabric_5shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_metabric_5shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral12b_metabric_5shot_conservateur.sh)
echo "  mistral12b_metabric_5shot_conservateur -> JID=$JID"
DEPS_MISTRAL12B="$DEPS_MISTRAL12B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=45:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_metabric_5shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_metabric_5shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral12b_metabric_5shot_standard.sh)
echo "  mistral12b_metabric_5shot_standard -> JID=$JID"
DEPS_MISTRAL12B="$DEPS_MISTRAL12B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=45:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_metabric_5shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_metabric_5shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral12b_metabric_5shot_exploratoire.sh)
echo "  mistral12b_metabric_5shot_exploratoire -> JID=$JID"
DEPS_MISTRAL12B="$DEPS_MISTRAL12B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=45:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_seer_0shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_seer_0shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral12b_seer_0shot_conservateur.sh)
echo "  mistral12b_seer_0shot_conservateur -> JID=$JID"
DEPS_MISTRAL12B="$DEPS_MISTRAL12B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=45:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_seer_0shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_seer_0shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral12b_seer_0shot_standard.sh)
echo "  mistral12b_seer_0shot_standard -> JID=$JID"
DEPS_MISTRAL12B="$DEPS_MISTRAL12B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=45:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_seer_0shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_seer_0shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral12b_seer_0shot_exploratoire.sh)
echo "  mistral12b_seer_0shot_exploratoire -> JID=$JID"
DEPS_MISTRAL12B="$DEPS_MISTRAL12B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=45:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_seer_5shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_seer_5shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral12b_seer_5shot_conservateur.sh)
echo "  mistral12b_seer_5shot_conservateur -> JID=$JID"
DEPS_MISTRAL12B="$DEPS_MISTRAL12B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=45:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_seer_5shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_seer_5shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral12b_seer_5shot_standard.sh)
echo "  mistral12b_seer_5shot_standard -> JID=$JID"
DEPS_MISTRAL12B="$DEPS_MISTRAL12B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=45:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_seer_5shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral12b_seer_5shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral12b_seer_5shot_exploratoire.sh)
echo "  mistral12b_seer_5shot_exploratoire -> JID=$JID"
DEPS_MISTRAL12B="$DEPS_MISTRAL12B -a $JID"

echo "  -> submitting evaluation mistral12b (after all generation jobs)"
JID=$(submit oarsub $DEPS_MISTRAL12B -l /gpu=2,walltime=08:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/eval_mistral12b.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/eval_mistral12b.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/eval_mistral12b.sh)
echo "  eval_mistral12b -> JID=$JID"

# ────────────────────────────────────────────────────────────
# mistral24b  (mistralai/Mistral-Small-24B-Instruct-2501)
# lig-gpu8.imag.fr / lig-gpu10.imag.fr  2 GPU  N=700
# ────────────────────────────────────────────────────────────
echo "=== Submitting mistral24b ==="
DEPS_MISTRAL24B=""

JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_metabric_0shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_metabric_0shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral24b_metabric_0shot_conservateur.sh)
echo "  mistral24b_metabric_0shot_conservateur -> JID=$JID"
DEPS_MISTRAL24B="$DEPS_MISTRAL24B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_metabric_0shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_metabric_0shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral24b_metabric_0shot_standard.sh)
echo "  mistral24b_metabric_0shot_standard -> JID=$JID"
DEPS_MISTRAL24B="$DEPS_MISTRAL24B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_metabric_0shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_metabric_0shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral24b_metabric_0shot_exploratoire.sh)
echo "  mistral24b_metabric_0shot_exploratoire -> JID=$JID"
DEPS_MISTRAL24B="$DEPS_MISTRAL24B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_metabric_5shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_metabric_5shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral24b_metabric_5shot_conservateur.sh)
echo "  mistral24b_metabric_5shot_conservateur -> JID=$JID"
DEPS_MISTRAL24B="$DEPS_MISTRAL24B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_metabric_5shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_metabric_5shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral24b_metabric_5shot_standard.sh)
echo "  mistral24b_metabric_5shot_standard -> JID=$JID"
DEPS_MISTRAL24B="$DEPS_MISTRAL24B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_metabric_5shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_metabric_5shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral24b_metabric_5shot_exploratoire.sh)
echo "  mistral24b_metabric_5shot_exploratoire -> JID=$JID"
DEPS_MISTRAL24B="$DEPS_MISTRAL24B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_seer_0shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_seer_0shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral24b_seer_0shot_conservateur.sh)
echo "  mistral24b_seer_0shot_conservateur -> JID=$JID"
DEPS_MISTRAL24B="$DEPS_MISTRAL24B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_seer_0shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_seer_0shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral24b_seer_0shot_standard.sh)
echo "  mistral24b_seer_0shot_standard -> JID=$JID"
DEPS_MISTRAL24B="$DEPS_MISTRAL24B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_seer_0shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_seer_0shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral24b_seer_0shot_exploratoire.sh)
echo "  mistral24b_seer_0shot_exploratoire -> JID=$JID"
DEPS_MISTRAL24B="$DEPS_MISTRAL24B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_seer_5shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_seer_5shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral24b_seer_5shot_conservateur.sh)
echo "  mistral24b_seer_5shot_conservateur -> JID=$JID"
DEPS_MISTRAL24B="$DEPS_MISTRAL24B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_seer_5shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_seer_5shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral24b_seer_5shot_standard.sh)
echo "  mistral24b_seer_5shot_standard -> JID=$JID"
DEPS_MISTRAL24B="$DEPS_MISTRAL24B -a $JID"
JID=$(submit oarsub -l /gpu=2,walltime=90:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_seer_5shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/mistral24b_seer_5shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/mistral24b_seer_5shot_exploratoire.sh)
echo "  mistral24b_seer_5shot_exploratoire -> JID=$JID"
DEPS_MISTRAL24B="$DEPS_MISTRAL24B -a $JID"

echo "  -> submitting evaluation mistral24b (after all generation jobs)"
JID=$(submit oarsub $DEPS_MISTRAL24B -l /gpu=2,walltime=08:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/eval_mistral24b.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/eval_mistral24b.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/eval_mistral24b.sh)
echo "  eval_mistral24b -> JID=$JID"

# ────────────────────────────────────────────────────────────
# apertus8b  (swiss-ai/Apertus-8B-Instruct-2509)
# lig-gpu8.imag.fr / lig-gpu10.imag.fr  1 GPU  N=700
# ────────────────────────────────────────────────────────────
echo "=== Submitting apertus8b ==="
DEPS_APERTUS8B=""

JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_metabric_0shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_metabric_0shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/apertus8b_metabric_0shot_conservateur.sh)
echo "  apertus8b_metabric_0shot_conservateur -> JID=$JID"
DEPS_APERTUS8B="$DEPS_APERTUS8B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_metabric_0shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_metabric_0shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/apertus8b_metabric_0shot_standard.sh)
echo "  apertus8b_metabric_0shot_standard -> JID=$JID"
DEPS_APERTUS8B="$DEPS_APERTUS8B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_metabric_0shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_metabric_0shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/apertus8b_metabric_0shot_exploratoire.sh)
echo "  apertus8b_metabric_0shot_exploratoire -> JID=$JID"
DEPS_APERTUS8B="$DEPS_APERTUS8B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_metabric_5shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_metabric_5shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/apertus8b_metabric_5shot_conservateur.sh)
echo "  apertus8b_metabric_5shot_conservateur -> JID=$JID"
DEPS_APERTUS8B="$DEPS_APERTUS8B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_metabric_5shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_metabric_5shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/apertus8b_metabric_5shot_standard.sh)
echo "  apertus8b_metabric_5shot_standard -> JID=$JID"
DEPS_APERTUS8B="$DEPS_APERTUS8B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_metabric_5shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_metabric_5shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/apertus8b_metabric_5shot_exploratoire.sh)
echo "  apertus8b_metabric_5shot_exploratoire -> JID=$JID"
DEPS_APERTUS8B="$DEPS_APERTUS8B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_seer_0shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_seer_0shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/apertus8b_seer_0shot_conservateur.sh)
echo "  apertus8b_seer_0shot_conservateur -> JID=$JID"
DEPS_APERTUS8B="$DEPS_APERTUS8B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_seer_0shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_seer_0shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/apertus8b_seer_0shot_standard.sh)
echo "  apertus8b_seer_0shot_standard -> JID=$JID"
DEPS_APERTUS8B="$DEPS_APERTUS8B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_seer_0shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_seer_0shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/apertus8b_seer_0shot_exploratoire.sh)
echo "  apertus8b_seer_0shot_exploratoire -> JID=$JID"
DEPS_APERTUS8B="$DEPS_APERTUS8B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_seer_5shot_conservateur.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_seer_5shot_conservateur.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/apertus8b_seer_5shot_conservateur.sh)
echo "  apertus8b_seer_5shot_conservateur -> JID=$JID"
DEPS_APERTUS8B="$DEPS_APERTUS8B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_seer_5shot_standard.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_seer_5shot_standard.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/apertus8b_seer_5shot_standard.sh)
echo "  apertus8b_seer_5shot_standard -> JID=$JID"
DEPS_APERTUS8B="$DEPS_APERTUS8B -a $JID"
JID=$(submit oarsub -l /gpu=1,walltime=30:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_seer_5shot_exploratoire.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/apertus8b_seer_5shot_exploratoire.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/apertus8b_seer_5shot_exploratoire.sh)
echo "  apertus8b_seer_5shot_exploratoire -> JID=$JID"
DEPS_APERTUS8B="$DEPS_APERTUS8B -a $JID"

echo "  -> submitting evaluation apertus8b (after all generation jobs)"
JID=$(submit oarsub $DEPS_APERTUS8B -l /gpu=2,walltime=08:00:00 -p "host in ('lig-gpu8.imag.fr', 'lig-gpu10.imag.fr')" -q default -O /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/eval_apertus8b.stdout -E /home/getalp/bouarabs/Data_Generation_3/CAG/logs/oar/eval_apertus8b.stderr /home/getalp/bouarabs/Data_Generation_3/CAG/logs/jobscripts/eval_apertus8b.sh)
echo "  eval_apertus8b -> JID=$JID"
