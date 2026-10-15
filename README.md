# synthetic-rt-reports-fr

Synthetic French breast cancer radiotherapy follow-up reports generated with a Context-Augmented Generation (CAG) pipeline. Seven open-source LLMs are compared across combinations of dataset, prompt strategy, and sampling temperature, producing a reproducibility-oriented evaluation campaign.

> **Data privacy notice.** The preprocessed patient profiles (`data/`, `data_processed/`) are derived from publicly available datasets (see below) but are **not included in this repository**. Real clinical reports (`data/CR_reels/`) must never be published. Only generated outputs and evaluation results are committed.

---

## Repository structure

```
synthetic-rt-reports-fr/
├── prompts/                        # Prompt templates (system + user, 0-shot / 5-shot × METABRIC / SEER)
├── référentiels/                   # Clinical reference PDFs (French breast cancer RT protocols)
├── scripts/
│   ├── preprocess_metabric.py      # METABRIC → data_processed/metabric_preprocessed.csv
│   ├── preprocess_seer.py          # SEER → data_processed/seer_preprocessed.csv
│   ├── campaign_config.py          # Central config: models, nodes, sampling presets, datasets
│   ├── make_jobs.py                # Generates OAR job scripts into logs/jobscripts/
│   ├── generate_cag.py             # CAG generation (one combination per run)
│   ├── evaluate_synthetic_reports.py  # Per-report + per-group automatic metrics
│   ├── add_bertscore_encoder.py    # Add BERTScore with a different encoder to existing results
│   ├── summarize_lexical_metrics.py   # Aggregate lexical metrics across all models
│   └── summarize_bertscore.py      # Aggregate BERTScore results across all models
├── jobs/
│   ├── submit_all.sh               # Submit the full generation + evaluation campaign to OAR
│   ├── submit_bertscore.sh         # Submit BERTScore recomputation jobs
│   └── bertscore_config.sh         # BERTScore job configuration
├── outputs/
│   ├── generated_CR/{model}/       # Generated reports: {model}_{dataset}_{prompt}_{preset}.jsonl
│   └── metrics_results/
│       ├── {model}/                # Per-model eval: *_eval_per_cr.csv + *_eval_summary.csv
│       ├── lexical_metrics_by_temperature.csv   # Cross-model lexical summary
│       └── bertscore_by_model.csv               # Cross-model BERTScore summary
├── data/                           # [gitignored] Raw METABRIC / SEER data + real CRs
├── data_processed/                 # [gitignored] Preprocessed patient profiles
└── logs/                           # [gitignored] OAR job scripts and cluster logs
```

---

## Models evaluated

| Short name     | HuggingFace ID                                 | Size  | dtype |
|----------------|------------------------------------------------|-------|-------|
| `qwen7b`       | Qwen/Qwen2.5-7B-Instruct                       | 7B    | fp16  |
| `qwen32b`      | Qwen/Qwen2.5-32B-Instruct                      | 32B   | bf16  |
| `mistral7b`    | mistralai/Mistral-7B-Instruct-v0.3             | 7B    | fp16  |
| `mistral12b`   | mistralai/Mistral-Nemo-Instruct-2407           | 12B   | fp16  |
| `mistral24b`   | mistralai/Mistral-Small-24B-Instruct-2501      | 24B   | bf16  |
| `apertus8b`    | swiss-ai/Apertus-8B-Instruct-2509              | 8B    | fp32  |
| `medgemma27b`  | google/medgemma-27b-it *(license required)*    | 27B   | bf16  |

**Experimental grid** — 84 combinations total:  
7 models × 2 datasets (METABRIC, SEER) × 2 prompts (0-shot, 5-shot) × 3 sampling presets (conservateur/standard/exploratoire)  
× 700 reports per combination

---

## Sampling presets

| Preset         | Temperature | top_p | repetition_penalty |
|----------------|-------------|-------|--------------------|
| conservateur   | 0.3         | 0.85  | 1.10               |
| standard       | 0.5         | 0.90  | 1.10               |
| exploratoire   | 0.8         | 0.95  | 1.15               |

---

## Automatic evaluation metrics

**Per report:** word count, MSTTR-100 (lexical richness), Flesch Reading Ease (FR), Gunning Fog (FR), perplexity (`asi/gpt-fr-cased-base`), BERTScore-max vs. 5-shot reference reports.

**Per group:** Self-BLEU-4, Distinct-1/2, mean pairwise cosine similarity (TF-IDF).

**BERTScore encoders:** CamemBERT-base, CamemBERT-bio, DrBERT — added via `add_bertscore_encoder.py`.

---

## Reproducing the pipeline

### Prerequisites

```bash
pip install transformers torch pandas faker textstat sacrebleu bert-score scikit-learn
# Python 3.10+, CUDA 11.8+ for GPU inference
```

### 1 — Download and preprocess patient data

**METABRIC** — download the clinical data CSV from [cBioPortal](https://www.cbioportal.org/study/clinicalData?id=brca_metabric) and place it in `data/`.

**SEER** — download the breast cancer dataset from [IEEE DataPort](https://ieee-dataport.org/open-access/seer-breast-cancer-data) and place it in `data/`.

```bash
python scripts/preprocess_metabric.py
python scripts/preprocess_seer.py
# → data_processed/metabric_preprocessed.csv
# → data_processed/seer_preprocessed.csv
```

### 2 — Smoke-test locally (CPU, tiny model)

```bash
python scripts/generate_cag.py \
    --dataset metabric --prompt_type 0shot \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --n 5 --out outputs/smoketest.jsonl
```

### 3 — Generate the full campaign on the OAR cluster

```bash
# From the LIG cluster frontend (aker.imag.fr)
python scripts/make_jobs.py          # generates logs/jobscripts/
bash jobs/submit_all.sh             # submits all OAR jobs (generation + evaluation chained)
```

Each job generates 700 reports for one (model × dataset × prompt × preset) combination and immediately chains an evaluation job.

### 4 — Evaluate manually (single combination)

```bash
python scripts/evaluate_synthetic_reports.py \
    --input outputs/generated_CR/qwen7b/*.jsonl \
    --out_per_cr  outputs/metrics_results/qwen7b/qwen7b_eval_per_cr.csv \
    --out_summary outputs/metrics_results/qwen7b/qwen7b_eval_summary.csv
```

### 5 — Add a BERTScore encoder

```bash
bash jobs/submit_bertscore.sh
# or directly:
# edit jobs/bertscore_config.sh to set MODEL and ENCODER, then run add_bertscore_encoder.py
```

### 6 — Summarize results

```bash
python scripts/summarize_lexical_metrics.py   # → outputs/metrics_results/lexical_metrics_by_temperature.csv
python scripts/summarize_bertscore.py         # → outputs/metrics_results/bertscore_by_model.csv
```

---

## Cluster notes (LIG OAR)

- Frontend: `aker.imag.fr`
- GPU nodes used: `lig-gpu8.imag.fr`, `lig-gpu10.imag.fr` (NVIDIA A6000, 48 GB each)
- GPU syntax in job scripts: `/gpu=N`
- Batch walltime ceiling: 100 h
- Pascal-architecture nodes (gpu1–gpu5) are **not compatible** with recent model dtypes

---

## Citation / reference

If you use this pipeline or the generated data, please cite the associated paper (reference TBD).
