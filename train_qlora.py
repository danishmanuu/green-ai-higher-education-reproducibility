"""
Optional QLoRA experiment for an OPEN-SOURCE language model.

Important scientific note:
GPT-3.5 is a proprietary API model and cannot be locally fine-tuned with
QLoRA. Do not describe this script or its results as "GPT-3.5 + QLoRA".
Use an explicit open-source base-model name in the manuscript.
"""

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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for an open-source LLM."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/qlora"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    set_global_seed(args.seed)

    try:
        import torch
        from datasets import Dataset
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from sklearn.metrics import f1_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install requirements.txt and requirements-qlora.txt."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA requires a CUDA-capable GPU.")

    dataframe = load_feedback_data(
        args.data,
        text_column=args.text_col,
        label_column=args.label_col,
    )
    train_df, test_df = stratified_split(
        dataframe,
        seed=args.seed,
        test_size=0.20,
    )

    safe_model_name = args.model_name_or_path.replace("/", "_")
    model_name = f"{args.model_name_or_path} + QLoRA"
    run_dir = (
        args.output_dir
        / safe_model_name
        / f"seed_{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=3,
        quantization_config=quantization_config,
        device_map="auto",
        id2label={0: "Positive", 1: "Neutral", 2: "Negative"},
        label2id={"Positive": 0, "Neutral": 1, "Negative": 2},
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        modules_to_save=["score"],
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
    ).map(tokenize, batched=True, remove_columns=[args.text_col])

    test_dataset = Dataset.from_pandas(
        test_df[[args.text_col, "label_id"]].rename(
            columns={"label_id": "labels"}
        ),
        preserve_index=False,
    ).map(tokenize, batched=True, remove_columns=[args.text_col])

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return {
            "macro_f1": float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )
        }

    kwargs = {
        "output_dir": str(run_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**kwargs)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": test_dataset,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
    }

    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    training_tracker = start_emissions_tracker(
        run_dir,
        project_name=f"qlora_training_seed_{args.seed}",
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
        project_name=f"qlora_inference_seed_{args.seed}",
        output_file="codecarbon_inference.csv",
    )
    torch.cuda.synchronize()
    with Timer() as inference_timer:
        output = trainer.predict(test_dataset)
    torch.cuda.synchronize()
    inference_energy = stop_emissions_tracker(
        inference_tracker,
        run_dir,
        "codecarbon_inference.csv",
    )

    logits = output.predictions
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
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
            "model_key": safe_model_name + "_qlora",
            "base_model": args.model_name_or_path,
            "method": "qlora",
            "seed": args.seed,
        }
    )

    resources = {
        "model": model_name,
        "model_key": safe_model_name + "_qlora",
        "base_model": args.model_name_or_path,
        "method": "qlora",
        "seed": args.seed,
        "source_type": "locally_measured",
        "training_seconds": training_timer.elapsed_seconds,
        "inference_seconds": inference_timer.elapsed_seconds,
        "latency_ms_per_sample": (
            inference_timer.elapsed_seconds / len(test_df) * 1000
        ),
        "throughput_samples_per_second": (
            len(test_df) / inference_timer.elapsed_seconds
        ),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        ),
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

    print(
        f"Completed {model_name}, seed={args.seed}, "
        f"macro-F1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
