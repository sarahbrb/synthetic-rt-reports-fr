## Synthetic Radiotherapy Report Generation & Evaluation

This project provides a pipeline for:
* Generating synthetic end-of-treatment radiotherapy reports for breast cancer patients using Large Language Models.
* Conditioning generation on structured clinical variables.
* Automatically evaluating linguistic and medical quality of generated reports.
* Producing structured datasets and analysis reports.
* The system is designed for clinical NLP research, and controlled LLM generation experiments.

## Project Structure

Data_Generation/
│
├── prompts/
│   ├── system_prompt.txt
│   ├── user_prompt_v5.txt
│   └── prompt_v0.txt
│
├── scripts/
│   ├── generate_script_v5.py        # Synthetic report generation script
│   ├── evaluate_automatic.py        # Automatic linguistic + medical evaluation
│   ├── evaluate_llm_judge.py        # LLM-based qualitative evaluation
│   └── conversion.py                # Format conversion utilities
│
├── outputs/
│   ├── mistral_5shot_v5.jsonl       # Generated reports
│   └── reports_for_review.xlsx      # Manual review export
│
├── evaluation/
│   └── metrics_auto_v5.json         # Automatic evaluation results



## Installation

##### Create environment:
`conda create -n synthgen python=3.10
conda activate synthgen`


##### Install dependencies:
`pip install torch transformers faker numpy`




