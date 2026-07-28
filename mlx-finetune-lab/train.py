"""LoRA fine-tune of Qwen2.5-1.5B-Instruct on the toy JSON-extraction task,
with every run tracked in MLflow: hyperparams, loss curves, and the
resulting adapter as an artifact.

This is the MLOps layer bolted onto mlx-lm's training loop via its
TrainingCallback hook, rather than scraping stdout.
"""

from types import SimpleNamespace

import mlflow

from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.trainer import TrainingCallback
from mlx_lm.utils import load

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "qwen2.5-1.5b-json-extraction"

HYPERPARAMS = {
    "model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "data": "data",
    "train": True,
    "fine_tune_type": "lora",
    "optimizer": "adamw",
    "num_layers": 8,
    "batch_size": 4,
    "iters": 200,
    "val_batches": 5,
    "learning_rate": 1e-4,
    "steps_per_report": 10,
    "steps_per_eval": 25,
    "save_every": 200,
    "adapter_path": "adapters",
    "max_seq_length": 512,
    "mask_prompt": True,
    "seed": 0,
    "grad_accumulation_steps": 1,
    "grad_checkpoint": False,
    "resume_adapter_file": None,
    "lr_schedule": None,
    "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
}


class MLflowCallback(TrainingCallback):
    def on_train_loss_report(self, train_info: dict):
        mlflow.log_metrics(
            {
                "train_loss": train_info["train_loss"],
                "learning_rate": train_info["learning_rate"],
                "tokens_per_second": train_info["tokens_per_second"],
                "peak_memory_gb": train_info["peak_memory"],
            },
            step=train_info["iteration"],
        )

    def on_val_loss_report(self, val_info: dict):
        mlflow.log_metric("val_loss", val_info["val_loss"], step=val_info["iteration"])


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    args = SimpleNamespace(**{**CONFIG_DEFAULTS, **HYPERPARAMS})

    with mlflow.start_run():
        mlflow.log_params(HYPERPARAMS)

        print(f"Loading base model: {args.model}")
        model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})

        print("Loading dataset")
        train_set, valid_set, _ = load_dataset(args, tokenizer)
        mlflow.log_params({"train_examples": len(train_set), "valid_examples": len(valid_set)})

        train_model(args, model, train_set, valid_set, training_callback=MLflowCallback())

        mlflow.log_artifacts(args.adapter_path, artifact_path="adapter")
        print(f"Run complete. Adapter saved to {args.adapter_path}/, logged to MLflow.")


if __name__ == "__main__":
    main()
