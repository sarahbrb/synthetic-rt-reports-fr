## Synthetic Radiotherapy Report Generation & Evaluation

📌 Overview

This repository contains the pipeline used to generate, validate, and evaluate synthetic end-of-radiotherapy reports for breast cancer using Large Language Models (LLMs).

The objective is to study to what extent a language model can produce clinically coherent medical reports when conditioned on structured clinical variables.

The pipeline combines:

* controlled synthetic patient profiles,
* structured prompting,
* automatic validation,
* expert medical assessment.

The full methodology is described in: [Méthodologie(Data_Generation).pdf)](https://gricad-gitlab.univ-grenoble-alpes.fr/data-generation-team/data-generation/-/blob/main/M%C3%A9thodologie__Data_Generation_.pdf)


## Installation

##### Create environment:
`conda create -n synthgen python=3.10
conda activate synthgen`


##### Install dependencies:
`pip install torch transformers faker numpy`




