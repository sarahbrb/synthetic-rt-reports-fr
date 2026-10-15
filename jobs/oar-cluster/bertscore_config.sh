#!/bin/bash
# Central BERTScore encoder configuration.
# Each entry: "output_column_name|HuggingFace_model_id"
#
# Note: camembert-base is already computed as bertscore_max by evaluate_synthetic_reports.py
# — only add encoders that need add_bertscore_encoder.py.
#
# To add a new encoder: append one line here, then re-run submit_bertscore.sh.

BERTSCORE_ENCODERS=(
    "bertscore_max_base|camembert-base"
    "bertscore_max_bio|almanach/camembert-bio-base"
    "bertscore_max_drbert|Dr-BERT/DrBERT-7GB"
)
