# Green AI for Higher Education — Reproducibility Repository

This repository contains the Python implementation and reproducibility workflow for the manuscript:

**Green AI for Higher Education: A Sustainability-Oriented Comparative Evaluation of Transformer-Based Learning Systems**

## Repository purpose

The repository allows reviewers and readers to run the complete evaluation pipeline for:

- educational-feedback sentiment classification,
- traditional ML baselines,
- transformer-based models,
- parameter-efficient fine-tuning,
- optional open-source QLoRA,
- training and inference tracking,
- CodeCarbon-based sustainability measurement,
- repeated-seed statistical reporting,
- confusion matrices,
- ROC curves,
- precision–recall curves,
- sustainability and EPR plots.

## Critical dataset scope statement

This repository contains a **synthetic 120,000-comment educational-feedback dataset**:

```text
data/synthetic_educational_feedback_120k.csv
```

This file follows the same minimum schema and approximate domain/class structure used in the manuscript, but it is **synthetic**.

It must not be described as:

- the original institutional LMS dataset,
- real student feedback,
- the exact dataset used to produce the manuscript results.

The original anonymised LMS dataset cannot be publicly released because of privacy and ethical restrictions. The synthetic dataset is provided only for:

- checking the code,
- validating the input schema,
- testing the workflow at manuscript-like scale,
- regenerating figures from a runnable dataset.

The final manuscript results should be generated from actual held-out labels, model predictions, predicted probabilities, and CodeCarbon logs.

## Repository structure

```text
green-ai-higher-education-reproducibility/
├── .github/
│   └── workflows/
│       └── reproducibility-check.yml
├── data/
│   ├── sample_feedback.csv
│   ├── synthetic_educational_feedback_120k.csv
│   ├── data_dictionary.csv
│   ├── dataset_manifest.json
│   ├── README_DATASET_SCOPE.md
│   ├── efficiency_input_template.csv
│   └── generate_synthetic_dataset.py
├── green_ai_utils.py
├── train_baselines.py
├── train_transformer.py
├── train_qlora.py
├── run_all_local.py
├── evaluate_predictions.py
├── plot_sustainability.py
├── make_external_api_prediction_template.py
├── requirements.txt
├── requirements-ci.txt
├── requirements-qlora.txt
├── README.md
├── REVIEWER_RUN_INSTRUCTIONS.md
├── CITATION.cff
├── LICENSE
└── .gitignore
```

## Dataset format

The training scripts require at least the following CSV columns:

```csv
text,label
```

Accepted labels are:

```text
Positive
Neutral
Negative
```

or:

```text
0 = Positive
1 = Neutral
2 = Negative
```

The synthetic 120k dataset also includes:

```text
comment_id
label_id
domain
source_type
course_code
anonymised_user_id
split
synthetic_flag
```

## Install locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install lightweight packages for baseline testing:

```bash
pip install -r requirements-ci.txt
```

Install full packages for transformer experiments:

```bash
pip install -r requirements.txt
```

Optional QLoRA packages:

```bash
pip install -r requirements-qlora.txt
```

## Fast verification run

```bash
python train_baselines.py \
  --data data/sample_feedback.csv \
  --models logistic_regression linear_svm \
  --seeds 42 52 62 72 82 \
  --output_dir results/baselines

python evaluate_predictions.py \
  --input_dir results \
  --output_dir reports \
  --figure_seed 42
```

This generates:

```text
reports/performance_per_run.csv
reports/performance_summary_mean_sd_ci.csv
reports/normalised_confusion_matrices.png
reports/macro_average_roc_curves.png
reports/macro_average_pr_curves.png
```

## Run on the synthetic 120,000-comment dataset

```bash
python train_baselines.py \
  --data data/synthetic_educational_feedback_120k.csv \
  --models logistic_regression linear_svm \
  --seeds 42 52 62 72 82 \
  --output_dir results/baselines_120k

python evaluate_predictions.py \
  --input_dir results \
  --output_dir reports_120k \
  --figure_seed 42
```

## Run a transformer model

Example using DistilBERT:

```bash
python train_transformer.py \
  --data data/synthetic_educational_feedback_120k.csv \
  --model distilbert \
  --method full \
  --seed 42 \
  --output_dir results/transformers
```

Example using BERT with LoRA:

```bash
python train_transformer.py \
  --data data/synthetic_educational_feedback_120k.csv \
  --model bert \
  --method lora \
  --seed 42 \
  --output_dir results/transformers
```

Run the locally executable model set over five seeds:

```bash
python run_all_local.py \
  --data data/synthetic_educational_feedback_120k.csv \
  --seeds 42 52 62 72 82 \
  --include_bert_lora \
  --output_dir results/transformers
```

## Optional QLoRA

Important: GPT-3.5 cannot be locally fine-tuned with QLoRA because its model weights are not available.

Use QLoRA only with an explicitly identified open-source model:

```bash
python train_qlora.py \
  --data data/synthetic_educational_feedback_120k.csv \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --seed 42 \
  --output_dir results/qlora
```

## GitHub Actions

The repository includes:

```text
.github/workflows/reproducibility-check.yml
```

The workflow runs a fast five-seed baseline verification and generates output tables and figures as artifacts.

## Output files

The evaluation script generates:

```text
performance_per_run.csv
performance_summary_mean_sd_ci.csv
normalised_confusion_matrices.png
macro_average_roc_curves.png
macro_average_pr_curves.png
resources_per_run.csv
resources_summary_mean_sd.csv
```

## Sustainability figures

Example:

```bash
python plot_sustainability.py \
  --input data/efficiency_input_template.csv \
  --output_dir reports/sustainability
```

API-estimated values are visually distinguished from locally measured values.

## Recommended manuscript wording

Use the following wording in the manuscript:

> The Python implementation, dependency files, model-training scripts, evaluation scripts, figure-generation code, and GitHub Actions workflow are publicly available in the associated GitHub repository. The repository includes a synthetic 120,000-comment educational-feedback dataset that follows the same schema and approximate domain/class structure as the experimental data. This synthetic dataset is provided only to demonstrate code execution and workflow repeatability. It was not used to derive the reported manuscript results. The original anonymised institutional LMS data cannot be publicly released because of privacy and ethical restrictions.

## Recommended reviewer-response wording

> We have prepared a GitHub repository containing the complete Python implementation, dependency files, dataset schema, evaluation scripts, figure-generation scripts, and an automated GitHub Actions workflow. Because the institutional LMS dataset contains restricted student feedback, it cannot be publicly released. To enable reviewers and readers to execute the pipeline, we provide a synthetic 120,000-comment educational-feedback dataset with the same schema and approximate class/domain structure. This synthetic dataset is provided for workflow verification only and was not used to derive the reported experimental results.

## License

MIT License.
