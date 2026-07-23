"""
Génère les scripts OAR pour la campagne complète (génération + évaluation).

Pour chaque modèle :
  - 1 job OAR par combinaison (dataset × prompt_type × sampling_preset)
  - apertus70b : 2 jobs par combinaison (part1=400 CR, part2=300 CR, via --skip)
  - 1 job d'évaluation par modèle, déclenché automatiquement après tous les
    jobs de génération via les dépendances OAR (-a <job_id>)

Usage :
    cd /home/getalp/bouarabs/Data_Generation_3/CAG
    python scripts/make_jobs.py
    bash logs/submit_all.sh
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

ACTIVE        = [m for m in MODELS if m not in ("_smoketest", "mixtral")]
EVAL_WALLTIME = "08:00:00"
EVAL_GPU      = 2


def _p_constraint(nodes):
    if len(nodes) == 1:
        return f"host='{nodes[0]}'"
    nodes_q = ", ".join(f"'{n}'" for n in nodes)
    return f"host in ({nodes_q})"


def write_gen_job(tag, nodes, n_gpu, walltime, model_id, dataset,
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
    --dtype         bf16 \\
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
        f" -l host=1/gpu={n_gpu},walltime={walltime}"
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
        f" -l host=1/gpu={EVAL_GPU},walltime={EVAL_WALLTIME}"
        f" -p \"{_p_constraint(nodes)}\""
        f" -q default"
        f" -O {stdout_path}"
        f" -E {stderr_path}"
        f" {script_path}"
    )
    return oarsub


# submit_all.sh : pour chaque modèle, soumet les jobs de génération,
# capture leurs IDs, puis soumet le job d'évaluation en dépendance.
submit_lines = [
    "#!/bin/bash",
    "# Soumission OAR — génération + évaluation automatique par modèle.",
    "# L'évaluation de chaque modèle démarre automatiquement quand tous ses",
    "# jobs de génération sont terminés (dépendances OAR via -a <job_id>).",
    "",
    "# Utilitaire : soumet un job et renvoie son ID OAR.",
    "submit() {",
    '    local jid',
    '    jid=$("$@" 2>&1 | grep OAR_JOB_ID | cut -d= -f2)',
    '    if [ -z "$jid" ]; then',
    '        echo "[ERREUR] oarsub a échoué : $*" >&2',
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
    nodes, n_gpu, wt = MODEL_NODE[short_name]
    splits           = N_SPLIT.get(short_name)
    n_label          = "+".join(str(s["n"]) for s in splits) if splits else str(N_PER_RUN)

    submit_lines += [
        f"# {'─' * 60}",
        f"# {short_name}  ({model_id})",
        f"# {' / '.join(nodes)}  {n_gpu} GPU  N={n_label}",
        f"# {'─' * 60}",
        f'echo "=== Soumission {short_name} ==="',
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
                            tag, nodes, n_gpu, part["walltime"], model_id,
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
                        base_tag, nodes, n_gpu, wt, model_id,
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
        f'echo "  -> lancement évaluation {short_name} (après tous les jobs de génération)"',
        f'JID=$(submit {eval_oarsub.replace("oarsub ", "oarsub $DEPS_" + short_name.upper() + " ", 1)})',
        f'echo "  eval_{short_name} -> JID=$JID"',
        "",
    ]
    n_eval_scripts += 1

submit_path = BASE_DIR / "logs" / "submit_all.sh"
submit_path.write_text("\n".join(submit_lines) + "\n")
submit_path.chmod(0o755)

print(f"[OK] {n_gen_scripts} scripts de génération")
print(f"[OK] {n_eval_scripts} scripts d'évaluation")
print(f"[OK] Script de soumission : {submit_path}")
print()
print("Vérifier puis lancer depuis aker.imag.fr :")
print(f"    bash {submit_path}")
