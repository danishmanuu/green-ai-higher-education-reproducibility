from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np

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


MODEL_REGISTRY = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
    "albert": "albert-base-v2",
}

DISPLAY_NAMES = {
    "bert": "BERT",
    "roberta": "RoBERTa",
    "distilbert": "DistilBERT",
    "tinybert": "TinyBERT",
    "albert": "ALBERT",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a Hugging Face transformer for three-class "
            "educational-feedback sentiment classification."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY),
        default="bert",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help="Override the default Hugging Face model identifier.",
    )
    parser.add_argument(
        "--method",
        choices=["full", "lora"],
        default="full",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/transformers"),
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the fine-tuned model checkpoint.",
    )
    return parser.parse_args()


def lora_target_modules(model_type: str) -> list[str]:
    mapping = {
        "bert": ["query", "value"],
        "roberta": ["query", "value"],
        "albert": ["query", "value"],
        "distilbert": ["q_lin", "v_lin"],
    }
    return mapping.get(model_type, ["query", "value"])


def build_training_arguments(
    TrainingArguments,
    output_dir: Path,
    args: argparse.Namespace,
):
    kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 20,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    try:
        import torch

        if torch.cuda.is_available():
            kwargs["fp16"] = True
    except ImportError:
        pass

    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_arguments()
    set_global_seed(args.seed)

    try:
        import torch
        from datasets import Dataset
        from sklearn.metrics import f1_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install the full dependencies first: "
            "pip install -r requirements.txt"
        ) from exc

    dataframe = load_feedback_data(
        args.data,
        text_column=args.text_col,
        label_column=args.label_col,
    )
    train_df, test_df = stratified_split(
        dataframe,
        seed=args.seed,
        test_size=args.test_size,
    )

    model_id = args.model_name_or_path or MODEL_REGISTRY[args.model]
    method_suffix = "LoRA" if args.method == "lora" else "Full"
    model_name = f"{DISPLAY_NAMES[args.model]} ({method_suffix})"

    run_dir = (
        args.output_dir
        / f"{args.model}_{args.method}"
        / f"seed_{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=3,
        id2label={0: "Positive", 1: "Neutral", 2: "Negative"},
        label2id={"Positive": 0, "Neutral": 1, "Negative": 2},
    )

    if args.method == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise SystemExit(
                "PEFT is required for LoRA. Install requirements.txt."
            ) from exc

        model_type = getattr(model.config, "model_type", args.model)
        modules_to_save = ["classifier"]

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.10,
            bias="none",
            target_modules=lora_target_modules(model_type),
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    def tokenize(batch):
        return tokenizer(
            batch[args.text_col],
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = Dataset.from_pandas(
        train_df[[args.text_col, "label_id"]].rename(
            columns={"label_id": "labels"}
        ),
        preserve_index=False,
    )
    test_dataset = Dataset.from_pandas(
        test_df[[args.text_col, "label_id"]].rename(
            columns={"label_id": "labels"}
        ),
        preserve_index=False,
    )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=[args.text_col],
    )
    test_dataset = test_dataset.map(
        tokenize,
        batched=True,
        remove_columns=[args.text_col],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
        return {"macro_f1": float(macro_f1)}

    training_args = build_training_arguments(
        TrainingArguments,
        run_dir,
        args,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": test_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    }

    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    training_tracker = start_emissions_tracker(
        run_dir,
        project_name=(
            f"{args.model}_{args.method}_training_seed_{args.seed}"
        ),
        output_file="codecarbon_training.csv",
    )

    with Timer() as training_timer:
        trainer.train()

    training_energy = stop_emissions_tracker(
        training_tracker,
        run_dir,
        "codecarbon_training.csv",
    )

    inference_tracker = start_emissions_tracker(
        run_dir,
        project_name=(
            f"{args.model}_{args.method}_inference_seed_{args.seed}"
        ),
        output_file="codecarbon_inference.csv",
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with Timer() as inference_timer:
        prediction_output = trainer.predict(test_dataset)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_energy = stop_emissions_tracker(
        inference_tracker,
        run_dir,
        "codecarbon_inference.csv",
    )

    logits = prediction_output.predictions
    logits = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    probabilities = exponential / exponential.sum(axis=1, keepdims=True)
    predictions = np.argmax(probabilities, axis=1)
    y_true = test_df["label_id"].to_numpy()

    metrics = classification_metrics(
        y_true,
        predictions,
        probabilities,
    )
    metrics.update(
        {
            "model": model_name,
            "model_key": f"{args.model}_{args.method}",
            "base_model": model_id,
            "method": args.method,
            "seed": args.seed,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        }
    )

    latency_ms = (
        inference_timer.elapsed_seconds / max(len(test_df), 1) * 1000
    )
    throughput = (
        len(test_df) / inference_timer.elapsed_seconds
        if inference_timer.elapsed_seconds > 0
        else None
    )

    peak_gpu_memory_mb = None
    if torch.cuda.is_available():
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )

    resources = {
        "model": model_name,
        "model_key": f"{args.model}_{args.method}",
        "base_model": model_id,
        "method": args.method,
        "seed": args.seed,
        "source_type": "locally_measured",
        "training_seconds": training_timer.elapsed_seconds,
        "inference_seconds": inference_timer.elapsed_seconds,
        "latency_ms_per_sample": latency_ms,
        "throughput_samples_per_second": throughput,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
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
        y_true,
        predictions,
        probabilities,
        model_name=model_name,
        seed=args.seed,
    ).to_csv(run_dir / "predictions.csv", index=False)

    save_json(metrics, run_dir / "metrics.json")
    save_json(resources, run_dir / "resources.json")

    if args.save_model:
        trainer.save_model(run_dir / "saved_model")
        tokenizer.save_pretrained(run_dir / "saved_model")

    print(
        f"Completed {model_name}, seed={args.seed}, "
        f"macro-F1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
