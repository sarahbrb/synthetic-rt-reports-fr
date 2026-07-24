# CAG — Context-Augmented Generation of Synthetic Radiotherapy Reports

Pipeline for generating synthetic French breast cancer radiotherapy follow-up reports by conditioning open-weight LLMs on de-identified patient profiles from METABRIC and SEER.

Published at ...

---

## Overview

Training NLP models for French clinical text is hindered by a privacy constraint: patient data lives at the hospital (GHM Grenoble) while compute lives at the lab (LIG). CAG bypasses this by generating *synthetic* reports (fictitious but clinically realistic) using publicly available patient profiles validated against oncology guidelines.

**Pipeline steps:**
1. **Filtering** : 32 expert-validated coherence rules reduce METABRIC (1 173 → 702) and SEER (4 024 → 3 832) to clinically consistent profiles
2. **Generation** : each profile is formatted into a structured French clinical context and fed to an LLM
3. **Evaluation** : clinical field coverage, perplexity, lexical diversity (MSTTR-100), corpus diversity (Self-BLEU-4), anti-leakage (BERTScore) , + new metrics

---

## Repository structure

```
scripts/
  01_preprocessing_metabric.py   # METABRIC preprocessing and filtering
  02_filtering_metabric.py       # coherence rule filtering (METABRIC)
  02_filtering_seer.py           # coherence rule filtering (SEER)
  campaign_config.py             # models, nodes, sampling presets, dataset split config
  generate_cag.py                # generation script (one combo per call)
  evaluate_cag.py                # evaluation script (PPL, coverage, MSTTR, SBleu, BERTScore)
  evaluate_reports.py            # report-level evaluation utilities
  make_jobs.py                   # generates OAR job scripts + submit_all.sh

prompts/
  user_prompt_metabric_0shot.txt
  user_prompt_metabric_5shot.txt
  user_prompt_seer_0shot.txt
  user_prompt_seer_5shot.txt
```

---

## Setup

```bash
conda activate synthgen
pip install transformers accelerate torch evaluate bert-score
```

---

## Running the campaign (OAR cluster)

```bash
# Generate all OAR job scripts and the submission script
python scripts/make_jobs.py

# Submit from aker.imag.fr
bash logs/submit_all.sh
```

Each generation job covers one combination: `dataset × prompt_type × sampling_preset × model`.  
Evaluation jobs are chained automatically via OAR dependencies (`-a <job_id>`).

### Sampling presets

| Preset        | Temperature | top_p | repetition_penalty |
|---------------|-------------|-------|--------------------|
| conservateur  | 0.3         | 0.85  | 1.10               |
| standard      | 0.5         | 0.90  | 1.10               |
| exploratoire  | 0.8         | 0.95  | 1.15               |

### Models evaluated

| Short name   | HuggingFace ID                              | Size    |
|--------------|---------------------------------------------|---------|
| qwen7b       | Qwen/Qwen2.5-7B-Instruct                    | 7B      |
| qwen32b      | Qwen/Qwen2.5-32B-Instruct                   | 32B     |
| mistral7b    | mistralai/Mistral-7B-Instruct-v0.3          | 7B      |
| mistral12b   | mistralai/Mistral-Nemo-Instruct-2407        | 12B     |
| mistral24b   | mistralai/Mistral-Small-24B-Instruct-2501   | 24B     |
| apertus8b    | swiss-ai/Apertus-8B-Instruct-2509           | 8B      |
| apertus70b   | swiss-ai/Apertus-70B-Instruct-2509          | 70B     |

---

## OAR cluster notes

- Frontend: `aker.imag.fr`
- Pascal GPUs (lig-gpu1 to lig-gpu5, sm_61) are excluded : incompatible with bf16 kernels
- 7B/8B models: gpu6 or gpu7 | 12B: gpu7 or gpu8 | 24B/32B: gpu8 | 70B: gpu9 (apertus)
- Batch walltime limit: 100h, apertus70b is split into two jobs per combination (400 + 300 CRs)

---

## Citation

```bibtex
@inproceedings{bouaraba2026cag,
  title     = {Context-Augmented Generation of Synthetic Radiotherapy Follow-up Reports},
  author    = {Bouaraba, Sarah and Goeuriot, Lorraine and Mulhem, Philippe and
               Portet, Fran\c{c}ois and Schwab, Didier},
  booktitle = {Proceedings of FDIA 2026},
  year      = {2026}
}
```
