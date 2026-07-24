from __future__ import annotations

import json
import random
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ["Positive", "Neutral", "Negative"]

LABEL_TO_ID = {
    "positive": 0,
    "neutral": 1,
    "negative": 2,
}

ID_TO_LABEL = {
    0: "Positive",
    1: "Neutral",
    2: "Negative",
}

PROBABILITY_COLUMNS = [
    "prob_positive",
    "prob_neutral",
    "prob_negative",
]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalise_labels(values: Iterable[Any]) -> np.ndarray:
    converted = []

    for value in values:
        if pd.isna(value):
            raise ValueError("Missing class label detected.")

        if isinstance(value, str):
            cleaned = value.strip().lower()

            if cleaned in LABEL_TO_ID:
                converted.append(LABEL_TO_ID[cleaned])
                continue

            try:
                numeric = int(float(cleaned))
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported label {value!r}. Use Positive, Neutral, "
                    "Negative, or numeric labels 0, 1, 2."
                ) from exc
        else:
            numeric = int(value)

        if numeric not in (0, 1, 2):
            raise ValueError("Only labels 0, 1, and 2 are allowed.")

        converted.append(numeric)

    return np.asarray(converted, dtype=int)


def load_feedback_data(
    path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    missing = {text_column, label_column}.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    dataframe = dataframe.copy()
    dataframe[text_column] = dataframe[text_column].astype(str).str.strip()
    dataframe = dataframe[dataframe[text_column].ne("")]
    dataframe = dataframe.dropna(subset=[label_column])
    dataframe = dataframe.drop_duplicates(subset=[text_column, label_column])
    dataframe["label_id"] = normalise_labels(dataframe[label_column])

    if dataframe["label_id"].nunique() != 3:
        raise ValueError("Dataset must contain Positive, Neutral, and Negative classes.")

    return dataframe.reset_index(drop=True)


def stratified_split(
    dataframe: pd.DataFrame,
    seed: int,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=seed,
        stratify=dataframe["label_id"],
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("Probability matrix must have shape (n_samples, 3).")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probability matrix contains NaN or infinity.")

    if (probabilities < 0).any():
        raise ValueError("Probability values cannot be negative.")

    row_sums = probabilities.sum(axis=1)

    if (row_sums <= 0).any():
        raise ValueError("Each probability row must have a positive sum.")

    if not np.allclose(row_sums, 1.0, atol=1e-5):
        probabilities = probabilities / row_sums[:, None]

    return probabilities


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    probabilities = validate_probabilities(probabilities)
    y_true_binary = label_binarize(y_true, classes=[0, 1, 2])

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

    try:
        metrics["macro_roc_auc"] = roc_auc_score(
            y_true_binary,
            probabilities,
            average="macro",
            multi_class="ovr",
        )
    except ValueError:
        metrics["macro_roc_auc"] = float("nan")

    try:
        metrics["macro_pr_auc"] = average_precision_score(
            y_true_binary,
            probabilities,
            average="macro",
        )
    except ValueError:
        metrics["macro_pr_auc"] = float("nan")

    return {key: float(value) for key, value in metrics.items()}


def prediction_dataframe(
    texts,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    seed: int,
    source_type: str = "locally_measured",
) -> pd.DataFrame:
    probabilities = validate_probabilities(probabilities)

    return pd.DataFrame(
        {
            "text": list(texts),
            "y_true": y_true,
            "y_pred": y_pred,
            "true_label": [ID_TO_LABEL[int(value)] for value in y_true],
            "predicted_label": [ID_TO_LABEL[int(value)] for value in y_pred],
            "prob_positive": probabilities[:, 0],
            "prob_neutral": probabilities[:, 1],
            "prob_negative": probabilities[:, 2],
            "model": model_name,
            "seed": seed,
            "source_type": source_type,
        }
    )


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def start_emissions_tracker(
    output_dir: str | Path,
    project_name: str,
    output_file: str,
):
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        warnings.warn(
            "CodeCarbon is not installed. Energy and carbon values will be null.",
            RuntimeWarning,
        )
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=str(output_dir),
        output_file=output_file,
        save_to_file=True,
        log_level="error",
        measure_power_secs=1,
    )
    tracker.start()
    return tracker


def stop_emissions_tracker(
    tracker,
    output_dir: str | Path,
    output_file: str,
) -> dict[str, float | None]:
    if tracker is None:
        return {
            "energy_kwh": None,
            "emissions_kg_co2eq": None,
        }

    emissions = tracker.stop()
    csv_path = Path(output_dir) / output_file

    energy = None
    emissions_from_file = None

    if csv_path.exists():
        try:
            row = pd.read_csv(csv_path).iloc[-1]

            if "energy_consumed" in row:
                energy = float(row["energy_consumed"])

            if "emissions" in row:
                emissions_from_file = float(row["emissions"])
        except Exception:
            pass

    return {
        "energy_kwh": energy,
        "emissions_kg_co2eq": (
            emissions_from_file
            if emissions_from_file is not None
            else float(emissions)
            if emissions is not None
            else None
        ),
    }


class Timer:
    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed_seconds = time.perf_counter() - self.started
