from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from green_ai_utils import (
    Timer,
    classification_metrics,
    load_feedback_data,
    prediction_dataframe,
    save_json,
    set_global_seed,
    start_emissions_tracker,
    stop_emissions_tracker,
    stratified_split,
)


MODEL_BUILDERS = {
    "logistic_regression": lambda seed: Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    ),
    "linear_svm": lambda seed: Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                SVC(
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    ),
}

DISPLAY_NAMES = {
    "logistic_regression": "TF-IDF + Logistic Regression",
    "linear_svm": "TF-IDF + Linear SVM",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train traditional text-classification baselines."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_BUILDERS),
        default=["logistic_regression", "linear_svm"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 52, 62, 72, 82],
    )
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/baselines"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataframe = load_feedback_data(
        args.data,
        text_column=args.text_col,
        label_column=args.label_col,
    )

    for model_key in args.models:
        model_name = DISPLAY_NAMES[model_key]

        for seed in args.seeds:
            set_global_seed(seed)
            train_df, test_df = stratified_split(
                dataframe,
                seed=seed,
                test_size=args.test_size,
            )

            run_dir = args.output_dir / model_key / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            model = MODEL_BUILDERS[model_key](seed)

            training_tracker = start_emissions_tracker(
                run_dir,
                project_name=f"{model_key}_training_seed_{seed}",
                output_file="codecarbon_training.csv",
            )

            with Timer() as training_timer:
                model.fit(
                    train_df[args.text_col],
                    train_df["label_id"],
                )

            training_energy = stop_emissions_tracker(
                training_tracker,
                run_dir,
                "codecarbon_training.csv",
            )

            inference_tracker = start_emissions_tracker(
                run_dir,
                project_name=f"{model_key}_inference_seed_{seed}",
                output_file="codecarbon_inference.csv",
            )

            with Timer() as inference_timer:
                probabilities = model.predict_proba(
                    test_df[args.text_col]
                )
                predictions = np.argmax(probabilities, axis=1)

            inference_energy = stop_emissions_tracker(
                inference_tracker,
                run_dir,
                "codecarbon_inference.csv",
            )

            metrics = classification_metrics(
                test_df["label_id"].to_numpy(),
                predictions,
                probabilities,
            )
            metrics.update(
                {
                    "model": model_name,
                    "model_key": model_key,
                    "seed": seed,
                    "train_samples": len(train_df),
                    "test_samples": len(test_df),
                }
            )

            latency_ms = (
                inference_timer.elapsed_seconds
                / max(len(test_df), 1)
                * 1000
            )
            throughput = (
                len(test_df) / inference_timer.elapsed_seconds
                if inference_timer.elapsed_seconds > 0
                else None
            )

            resources = {
                "model": model_name,
                "model_key": model_key,
                "seed": seed,
                "source_type": "locally_measured",
                "training_seconds": training_timer.elapsed_seconds,
                "inference_seconds": inference_timer.elapsed_seconds,
                "latency_ms_per_sample": latency_ms,
                "throughput_samples_per_second": throughput,
                "training_energy_kwh": training_energy["energy_kwh"],
                "training_emissions_kg_co2eq": training_energy[
                    "emissions_kg_co2eq"
                ],
                "inference_energy_kwh": inference_energy["energy_kwh"],
                "inference_emissions_kg_co2eq": inference_energy[
                    "emissions_kg_co2eq"
                ],
            }

            prediction_dataframe(
                test_df[args.text_col],
                test_df["label_id"].to_numpy(),
                predictions,
                probabilities,
                model_name=model_name,
                seed=seed,
            ).to_csv(run_dir / "predictions.csv", index=False)

            save_json(metrics, run_dir / "metrics.json")
            save_json(resources, run_dir / "resources.json")

            print(
                f"Completed {model_name}, seed={seed}, "
                f"macro-F1={metrics['macro_f1']:.4f}"
            )


if __name__ == "__main__":
    main()
