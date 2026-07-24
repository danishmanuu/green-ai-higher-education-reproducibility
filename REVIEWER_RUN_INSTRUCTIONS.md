# Reviewer Run Instructions

This repository supports two levels of execution.

## 1. Fast GitHub Actions verification

The included GitHub Actions workflow uses the small `data/sample_feedback.csv` file.
This is intentional because GitHub-hosted CPU runners are not suitable for running
full transformer experiments or long GPU-based training jobs.

The workflow checks that:

- dependencies install correctly,
- baseline training runs,
- five-seed evaluation works,
- performance tables are generated,
- confusion-matrix, ROC, and precision–recall figures are generated.

## 2. Synthetic 120,000-comment workflow verification

The file `data/synthetic_educational_feedback_120k.csv` is a synthetic dataset
created only for workflow testing at the approximate scale of the manuscript.

Run the baseline pipeline on the 120k synthetic dataset:

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

## 3. Transformer execution

Transformer experiments require model downloads and GPU resources. Example:

```bash
python train_transformer.py \
  --data data/synthetic_educational_feedback_120k.csv \
  --model distilbert \
  --method full \
  --seed 42 \
  --output_dir results/transformers
```

For five-seed local transformer execution:

```bash
python run_all_local.py \
  --data data/synthetic_educational_feedback_120k.csv \
  --seeds 42 52 62 72 82 \
  --include_bert_lora \
  --output_dir results/transformers
```

## Important dataset note

The synthetic 120k dataset is **not** the original institutional LMS dataset.
It is included only to verify that the code pipeline can run on a dataset with
the same schema and approximate scale.

The original LMS dataset cannot be publicly shared because of privacy and
ethical restrictions.
