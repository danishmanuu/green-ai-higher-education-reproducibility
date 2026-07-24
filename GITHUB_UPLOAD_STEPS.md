# GitHub Upload Steps From Scratch

## Step 1: Create a GitHub repository

1. Go to https://github.com
2. Click the `+` icon in the top-right corner.
3. Click `New repository`.
4. Repository name:

```text
green-ai-higher-education-reproducibility
```

5. Description:

```text
Reproducibility code for Green AI evaluation of transformer-based models in higher education.
```

6. Select `Public`.
7. Do not add README, .gitignore, or license because these files are already included.
8. Click `Create repository`.

## Step 2: Extract this package

1. Download the ZIP package from ChatGPT.
2. Right-click the ZIP file on Windows.
3. Click `Extract All`.
4. Open the extracted folder.
5. Open the inner folder named:

```text
green-ai-higher-education-reproducibility
```

## Step 3: Upload files through GitHub website

1. Open your empty GitHub repository.
2. Click `uploading an existing file`.
3. Drag all files and folders from the extracted repository folder into GitHub.
4. Make sure these are visible in the upload list:

```text
.github/
data/
README.md
train_baselines.py
train_transformer.py
evaluate_predictions.py
requirements-ci.txt
```

5. Scroll down.
6. Commit message:

```text
Add Green AI reproducibility repository
```

7. Click `Commit changes`.

## Step 4: Run GitHub Actions

1. Click the `Actions` tab.
2. Click `Reproducibility Check`.
3. Click `Run workflow`.
4. Select the `main` or `master` branch.
5. Click the green `Run workflow` button.

## Step 5: Download generated outputs

1. Open the completed workflow run.
2. Scroll to `Artifacts`.
3. Download:

```text
reproducibility-check-results
```

The downloaded ZIP contains:

```text
performance_per_run.csv
performance_summary_mean_sd_ci.csv
normalised_confusion_matrices.png
macro_average_roc_curves.png
macro_average_pr_curves.png
```

## Step 6: Add the repository link in the manuscript

Use:

```text
The Python implementation, dependency files, model-training scripts, evaluation scripts,
figure-generation code, and GitHub Actions workflow are publicly available at:
https://github.com/YOUR_USERNAME/green-ai-higher-education-reproducibility
```

Replace `YOUR_USERNAME` with your GitHub username.

## Important

Do not claim the synthetic 120,000-comment dataset is the original LMS dataset.
It is included only for code execution and workflow verification.
