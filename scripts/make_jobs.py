"""
Generate OAR batch scripts for the full campaign (generation + evaluation).

Per model:
  - 1 OAR job per combination (dataset × prompt_type × sampling_preset)
  - apertus70b: 2 jobs per combination (part1=400 CR, part2=300 CR, via --skip)
  - 1 evaluation job per model, triggered automatically after all generation jobs finish via OAR dependencies (-a <job_id>)

To reproduce : run from the OAR frontend (aker.imag.fr):

    ssh aker.imag.fr
    cd /home/getalp/bouarabs/Data_Generation_3/CAG
    conda activate synthgen
    python scripts/make_jobs.py   # writes logs/jobscripts/ + logs/submit_all.sh
    bash logs/submit_all.sh       # submits everything to OAR

Monitor:
    oarstat -u $(whoami)
    tail -f logs/oar/<tag>.stdout
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from campaign_config import (
    MODELS, MODEL_NODE, SAMPLING_PRESETS,
    DATASETS, PROMPT_TYPES, N_PER_RUN, N_SPLIT,
)

BASE_DIR   = Path(__file__).parent.parent
JOBS_DIR   = BASE_DIR / "logs" / "jobscripts"
LOGS_DIR   = BASE_DIR / "logs" / "oar"
OUTPUT_DIR = BASE_DIR / "outputs"

for d in (JOBS_DIR, LOGS_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

ACTIVE        = list(MODELS)
EVAL_WALLTIME = "08:00:00"
EVAL_GPU      = 2


def _p_constraint(nodes):
    if len(nodes) == 1:
        return f"host='{nodes[0]}'"
    nodes_q = ", ".join(f"'{n}'" for n in nodes)
    return f"host in ({nodes_q})"


def write_gen_job(tag, nodes, n_gpu, walltime, dtype, model_id, dataset,
                  prompt_type, temp, top_p, rep_pen, n, skip, out_jsonl):
    script_path = JOBS_DIR / f"{tag}.sh"
    stdout_path = LOGS_DIR / f"{tag}.stdout"
    stderr_path = LOGS_DIR / f"{tag}.stderr"

    script_path.write_text(f"""\
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate synthgen
cd "{BASE_DIR}"
echo "=== $(date) -- START {tag} on $(hostname) ==="
python scripts/generate_cag.py \\
    --dataset       {dataset} \\
    --prompt_type   {prompt_type} \\
    --model         {model_id} \\
    --n             {n} \\
    --skip          {skip} \\
    --dtype         {dtype} \\
    --temperature   {temp} \\
    --top_p         {top_p} \\
    --repetition_penalty {rep_pen} \\
    --out           {out_jsonl}
code=$?
echo "=== $(date) -- END {tag} (exit $code) ==="
exit $code
""")
    script_path.chmod(0o755)

    oarsub = (
        f"oarsub"
        f" -l /gpu={n_gpu},walltime={walltime}"
        f" -p \"{_p_constraint(nodes)}\""
        f" -q default"
        f" -O {stdout_path}"
        f" -E {stderr_path}"
        f" {script_path}"
    )
    return oarsub


def write_eval_job(short_name, nodes):
    tag         = f"eval_{short_name}"
    script_path = JOBS_DIR / f"{tag}.sh"
    stdout_path = LOGS_DIR / f"{tag}.stdout"
    stderr_path = LOGS_DIR / f"{tag}.stderr"

    script_path.write_text(f"""\
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate synthgen
cd "{BASE_DIR}"
echo "=== $(date) -- START {tag} on $(hostname) ==="
python scripts/evaluate_cag.py \\
    --input outputs/{short_name}_*.jsonl \\
    --out_per_cr  outputs/{short_name}_eval_per_cr.csv \\
    --out_summary outputs/{short_name}_eval_summary.csv
code=$?
echo "=== $(date) -- END {tag} (exit $code) ==="
exit $code
""")
    script_path.chmod(0o755)

    oarsub = (
        f"oarsub"
        f" -l /gpu={EVAL_GPU},walltime={EVAL_WALLTIME}"
        f" -p \"{_p_constraint(nodes)}\""
        f" -q default"
        f" -O {stdout_path}"
        f" -E {stderr_path}"
        f" {script_path}"
    )
    return oarsub


# submit_all.sh: for each model, submit all generation jobs, capture their IDs,
# then submit the evaluation job as a dependent of all of them.
submit_lines = [
    "#!/bin/bash",
    "# OAR submission — generation + automatic evaluation per model.",
    "# Each model's evaluation job starts only after all its generation jobs",
    "# have completed successfully (OAR dependencies via -a <job_id>).",
    "",
    "# Helper: submit a job and return its OAR job ID.",
    "submit() {",
    '    local jid',
    '    jid=$("$@" 2>&1 | grep OAR_JOB_ID | cut -d= -f2)',
    '    if [ -z "$jid" ]; then',
    '        echo "[ERROR] oarsub failed: $*" >&2',
    '        exit 1',
    '    fi',
    '    echo "$jid"',
    "}",
    "",
]

n_gen_scripts  = 0
n_eval_scripts = 0

for short_name in ACTIVE:
    model_id         = MODELS[short_name]
    nodes, n_gpu, wt, dtype = MODEL_NODE[short_name]
    splits           = N_SPLIT.get(short_name)
    n_label          = "+".join(str(s["n"]) for s in splits) if splits else str(N_PER_RUN)

    submit_lines += [
        f"# {'─' * 60}",
        f"# {short_name}  ({model_id})",
        f"# {' / '.join(nodes)}  {n_gpu} GPU  N={n_label}",
        f"# {'─' * 60}",
        f'echo "=== Submitting {short_name} ==="',
        f"DEPS_{short_name.upper()}=\"\"",
        "",
    ]

    for dataset in DATASETS:
        for prompt_type in PROMPT_TYPES:
            for preset_name, (temp, top_p, rep_pen) in SAMPLING_PRESETS.items():
                base_tag = f"{short_name}_{dataset}_{prompt_type}_{preset_name}"

                if splits:
                    for part in splits:
                        tag      = f"{base_tag}_{part['suffix']}"
                        out_file = OUTPUT_DIR / f"{tag}.jsonl"
                        oarsub   = write_gen_job(
                            tag, nodes, n_gpu, part["walltime"], dtype, model_id,
                            dataset, prompt_type, temp, top_p, rep_pen,
                            part["n"], part["skip"], out_file,
                        )
                        submit_lines += [
                            f'JID=$(submit {oarsub})',
                            f'echo "  {tag} -> JID=$JID"',
                            f'DEPS_{short_name.upper()}="$DEPS_{short_name.upper()} -a $JID"',
                        ]
                        n_gen_scripts += 1
                else:
                    out_file = OUTPUT_DIR / f"{base_tag}.jsonl"
                    oarsub   = write_gen_job(
                        base_tag, nodes, n_gpu, wt, dtype, model_id,
                        dataset, prompt_type, temp, top_p, rep_pen,
                        N_PER_RUN, 0, out_file,
                    )
                    submit_lines += [
                        f'JID=$(submit {oarsub})',
                        f'echo "  {base_tag} -> JID=$JID"',
                        f'DEPS_{short_name.upper()}="$DEPS_{short_name.upper()} -a $JID"',
                    ]
                    n_gen_scripts += 1

    eval_oarsub = write_eval_job(short_name, nodes)
    submit_lines += [
        "",
        f'echo "  -> submitting evaluation {short_name} (after all generation jobs)"',
        f'JID=$(submit {eval_oarsub.replace("oarsub ", "oarsub $DEPS_" + short_name.upper() + " ", 1)})',
        f'echo "  eval_{short_name} -> JID=$JID"',
        "",
    ]
    n_eval_scripts += 1

submit_path = BASE_DIR / "logs" / "submit_all.sh"
submit_path.write_text("\n".join(submit_lines) + "\n")
submit_path.chmod(0o755)

print(f"[OK] {n_gen_scripts} generation scripts")
print(f"[OK] {n_eval_scripts} evaluation scripts")
print(f"[OK] Submission script: {submit_path}")
print()
print("Review then run from aker.imag.fr:")
print(f"    bash {submit_path}")
