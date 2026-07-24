from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from green_ai_utils import (
    CLASS_NAMES,
    PROBABILITY_COLUMNS,
    classification_metrics,
    normalise_labels,
    validate_probabilities,
)


METRIC_COLUMNS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "macro_roc_auc",
    "macro_pr_auc",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate repeated runs and generate confusion-matrix, "
            "ROC, and precision-recall figures."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports"),
    )
    parser.add_argument("--figure_seed", type=int, default=42)
    parser.add_argument(
        "--confusion_models",
        nargs="*",
        default=[
            "BERT (Full)",
            "RoBERTa (Full)",
            "DistilBERT (Full)",
            "BERT (LoRA)",
        ],
    )
    return parser.parse_args()


def load_prediction_files(input_dir: Path) -> pd.DataFrame:
    frames = []

    for path in sorted(input_dir.rglob("predictions.csv")):
        frame = pd.read_csv(path)

        required = {
            "y_true",
            "y_pred",
            *PROBABILITY_COLUMNS,
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        if "model" not in frame.columns:
            frame["model"] = path.parent.parent.name
        if "seed" not in frame.columns:
            seed_text = path.parent.name.replace("seed_", "")
            frame["seed"] = int(seed_text)
        if "source_type" not in frame.columns:
            frame["source_type"] = "locally_measured"

        frame["source_file"] = str(path)
        frame["y_true"] = normalise_labels(frame["y_true"])
        frame["y_pred"] = normalise_labels(frame["y_pred"])
        probabilities = validate_probabilities(
            frame[PROBABILITY_COLUMNS].to_numpy()
        )
        frame[PROBABILITY_COLUMNS] = probabilities
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No predictions.csv files were found below {input_dir}."
        )

    return pd.concat(frames, ignore_index=True)


def calculate_per_run_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model, seed), group in predictions.groupby(
        ["model", "seed"],
        sort=True,
    ):
        probabilities = group[PROBABILITY_COLUMNS].to_numpy()
        metrics = classification_metrics(
            group["y_true"].to_numpy(),
            group["y_pred"].to_numpy(),
            probabilities,
        )
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "source_type": group["source_type"].iloc[0],
                "n_test": len(group),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def summarise_runs(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model, group in per_run.groupby("model", sort=True):
        row = {
            "model": model,
            "source_type": group["source_type"].iloc[0],
            "runs": len(group),
        }

        for metric in METRIC_COLUMNS:
            values = group[metric].dropna().to_numpy(dtype=float)
            count = len(values)

            if count == 0:
                mean = std = ci_low = ci_high = float("nan")
            else:
                mean = float(np.mean(values))
                std = (
                    float(np.std(values, ddof=1))
                    if count > 1
                    else 0.0
                )
                half_width = (
                    1.96 * std / math.sqrt(count)
                    if count > 1
                    else 0.0
                )
                ci_low = mean - half_width
                ci_high = mean + half_width

            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = std
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high

        rows.append(row)

    return pd.DataFrame(rows)


def select_figure_runs(
    predictions: pd.DataFrame,
    figure_seed: int,
) -> dict[str, pd.DataFrame]:
    selected = {}

    for model, group in predictions.groupby("model", sort=True):
        seed_group = group[group["seed"] == figure_seed]

        if seed_group.empty:
            first_seed = int(group["seed"].min())
            seed_group = group[group["seed"] == first_seed]

        selected[model] = seed_group.reset_index(drop=True)

    return selected


def macro_roc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    binary = label_binarize(y_true, classes=[0, 1, 2])
    fpr_values = {}
    tpr_values = {}

    for class_id in range(3):
        fpr_values[class_id], tpr_values[class_id], _ = roc_curve(
            binary[:, class_id],
            probabilities[:, class_id],
        )

    common_fpr = np.unique(
        np.concatenate(
            [fpr_values[class_id] for class_id in range(3)]
        )
    )
    mean_tpr = np.zeros_like(common_fpr)

    for class_id in range(3):
        mean_tpr += np.interp(
            common_fpr,
            fpr_values[class_id],
            tpr_values[class_id],
        )

    mean_tpr /= 3
    area = float(np.trapz(mean_tpr, common_fpr))
    return common_fpr, mean_tpr, area


def macro_pr(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    binary = label_binarize(y_true, classes=[0, 1, 2])
    recall_grid = np.linspace(0, 1, 501)
    precision_values = []

    for class_id in range(3):
        precision, recall, _ = precision_recall_curve(
            binary[:, class_id],
            probabilities[:, class_id],
        )
        precision_values.append(
            np.interp(
                recall_grid,
                recall[::-1],
                precision[::-1],
            )
        )

    macro_precision = np.mean(precision_values, axis=0)
    macro_ap = float(
        average_precision_score(
            binary,
            probabilities,
            average="macro",
        )
    )
    return recall_grid, macro_precision, macro_ap


def plot_confusion_matrices(
    selected_runs: dict[str, pd.DataFrame],
    requested_models: list[str],
    output_path: Path,
) -> None:
    available = [
        model for model in requested_models if model in selected_runs
    ]
    if not available:
        available = list(selected_runs)[:4]

    available = available[:4]
    rows = 2
    columns = 2
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(10, 8),
    )
    axes = np.asarray(axes).ravel()
    image = None

    for axis, model in zip(axes, available):
        group = selected_runs[model]
        matrix = confusion_matrix(
            group["y_true"],
            group["y_pred"],
            labels=[0, 1, 2],
            normalize="true",
        )
        image = axis.imshow(matrix, vmin=0, vmax=1)
        axis.set_title(model, fontweight="bold")
        axis.set_xticks(range(3))
        axis.set_yticks(range(3))
        axis.set_xticklabels(CLASS_NAMES)
        axis.set_yticklabels(CLASS_NAMES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")

        for row in range(3):
            for column in range(3):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.55 else "black",
                )

    for axis in axes[len(available):]:
        axis.axis("off")

    if image is not None:
        colorbar = figure.colorbar(
            image,
            ax=axes.tolist(),
            fraction=0.03,
            pad=0.04,
        )
        colorbar.set_label("Proportion")

    figure.suptitle(
        "Normalised Confusion Matrices",
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.08,
        top=0.90,
        wspace=0.30,
        hspace=0.35,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_roc(
    selected_runs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))

    for model, group in selected_runs.items():
        probabilities = group[PROBABILITY_COLUMNS].to_numpy()
        fpr, tpr, area = macro_roc(
            group["y_true"].to_numpy(),
            probabilities,
        )
        line_style = (
            "--"
            if group["source_type"].iloc[0].startswith("api")
            else "-"
        )
        axis.plot(
            fpr,
            tpr,
            linestyle=line_style,
            linewidth=2,
            label=f"{model} (AUC = {area:.3f})",
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle=":",
        linewidth=1.5,
        label="Random baseline",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(
        "Macro-average Receiver Operating Characteristic Curves",
        fontweight="bold",
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_pr(
    selected_runs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))

    prevalence_values = []

    for model, group in selected_runs.items():
        probabilities = group[PROBABILITY_COLUMNS].to_numpy()
        recall, precision, area = macro_pr(
            group["y_true"].to_numpy(),
            probabilities,
        )
        line_style = (
            "--"
            if group["source_type"].iloc[0].startswith("api")
            else "-"
        )
        axis.plot(
            recall,
            precision,
            linestyle=line_style,
            linewidth=2,
            label=f"{model} (PR-AUC = {area:.3f})",
        )
        frequencies = (
            np.bincount(group["y_true"], minlength=3) / len(group)
        )
        prevalence_values.append(float(np.mean(frequencies)))

    baseline = float(np.mean(prevalence_values))
    axis.axhline(
        baseline,
        linestyle=":",
        linewidth=1.5,
        label=f"No-skill baseline ({baseline:.2f})",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(
        "Macro-average Precision–Recall Curves",
        fontweight="bold",
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def aggregate_resources(input_dir: Path) -> pd.DataFrame:
    rows = []

    for path in sorted(input_dir.rglob("resources.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(data)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    numeric_columns = [
        column
        for column in frame.columns
        if column not in {
            "model",
            "model_key",
            "base_model",
            "method",
            "source_type",
        }
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def summarise_resources(resources: pd.DataFrame) -> pd.DataFrame:
    if resources.empty:
        return resources

    metric_columns = [
        column
        for column in [
            "training_seconds",
            "inference_seconds",
            "latency_ms_per_sample",
            "throughput_samples_per_second",
            "peak_gpu_memory_mb",
            "training_energy_kwh",
            "training_emissions_kg_co2eq",
            "inference_energy_kwh",
            "inference_emissions_kg_co2eq",
        ]
        if column in resources.columns
    ]

    rows = []

    for model, group in resources.groupby("model", sort=True):
        row = {
            "model": model,
            "source_type": group["source_type"].iloc[0],
            "runs": len(group),
        }

        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = (
                float(np.mean(values)) if len(values) else float("nan")
            )
            row[f"{metric}_sd"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0 if len(values) == 1 else float("nan")
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_prediction_files(args.input_dir)
    per_run = calculate_per_run_metrics(predictions)
    summary = summarise_runs(per_run)

    per_run.to_csv(
        args.output_dir / "performance_per_run.csv",
        index=False,
    )
    summary.to_csv(
        args.output_dir / "performance_summary_mean_sd_ci.csv",
        index=False,
    )

    selected = select_figure_runs(
        predictions,
        figure_seed=args.figure_seed,
    )
    plot_confusion_matrices(
        selected,
        args.confusion_models,
        args.output_dir / "normalised_confusion_matrices.png",
    )
    plot_roc(
        selected,
        args.output_dir / "macro_average_roc_curves.png",
    )
    plot_pr(
        selected,
        args.output_dir / "macro_average_pr_curves.png",
    )

    resources = aggregate_resources(args.input_dir)
    if not resources.empty:
        resources.to_csv(
            args.output_dir / "resources_per_run.csv",
            index=False,
        )
        summarise_resources(resources).to_csv(
            args.output_dir / "resources_summary_mean_sd.csv",
            index=False,
        )

    print(summary.to_string(index=False))
    print(f"\nReports saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
