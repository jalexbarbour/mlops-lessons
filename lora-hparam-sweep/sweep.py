"""Grid search over LoRA rank and learning rate, each combination logged as
its own MLflow run. This is the systematic version of "try a config, eyeball
the loss" — every run is comparable afterward instead of only the last one
you happened to look at.

Same toy task as mlx-finetune-lab (freeform note -> structured JSON) so the
sweep's results are directly comparable to that lab's single baseline run.
"""

import itertools
from pathlib import Path
from types import SimpleNamespace

import mlflow

from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.trainer import TrainingCallback
from mlx_lm.utils import load

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "qwen2.5-1.5b-json-extraction-sweep"
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

RANKS = [4, 8, 16]
LEARNING_RATES = [3e-5, 1e-4, 3e-4]

FIXED = {
    "model": BASE_MODEL,
    "data": "data",
    "train": True,
    "fine_tune_type": "lora",
    "optimizer": "adamw",
    "num_layers": 8,
    "batch_size": 4,
    "iters": 80,
    "val_batches": 5,
    "steps_per_report": 10,
    "steps_per_eval": 20,
    "save_every": 80,
    "max_seq_length": 512,
    "mask_prompt": True,
    "seed": 0,
    "grad_accumulation_steps": 1,
    "grad_checkpoint": False,
    "resume_adapter_file": None,
    "lr_schedule": None,
}


class MLflowCallback(TrainingCallback):
    def on_train_loss_report(self, train_info: dict):
        mlflow.log_metrics(
            {
                "train_loss": train_info["train_loss"],
                "peak_memory_gb": train_info["peak_memory"],
            },
            step=train_info["iteration"],
        )

    def on_val_loss_report(self, val_info: dict):
        mlflow.log_metric("val_loss", val_info["val_loss"], step=val_info["iteration"])


def run_one(train_set, valid_set, rank: float, learning_rate: float):
    adapter_path = f"adapters/r{rank}_lr{learning_rate:.0e}"
    Path(adapter_path).mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        **{
            **CONFIG_DEFAULTS,
            **FIXED,
            "learning_rate": learning_rate,
            "adapter_path": adapter_path,
            "lora_parameters": {"rank": rank, "dropout": 0.0, "scale": 20.0},
        }
    )

    with mlflow.start_run(run_name=f"r{rank}-lr{learning_rate:.0e}"):
        mlflow.log_params(
            {
                "rank": rank,
                "learning_rate": learning_rate,
                "num_layers": FIXED["num_layers"],
                "iters": FIXED["iters"],
                "adapter_path": adapter_path,
            }
        )
        # train_model mutates the model in place (freezes/unfreezes layers,
        # converts linear layers to LoRA layers); a fresh load per run keeps
        # configs from leaking into each other.
        run_model, _ = load(BASE_MODEL, tokenizer_config={"trust_remote_code": True})
        train_model(args, run_model, train_set, valid_set, training_callback=MLflowCallback())
        mlflow.log_artifacts(adapter_path, artifact_path="adapter")


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Loading tokenizer from: {BASE_MODEL}")
    _, tokenizer = load(BASE_MODEL, tokenizer_config={"trust_remote_code": True})

    print("Loading dataset")
    args_for_data = SimpleNamespace(**{**CONFIG_DEFAULTS, **FIXED})
    train_set, valid_set, _ = load_dataset(args_for_data, tokenizer)

    configs = list(itertools.product(RANKS, LEARNING_RATES))
    print(f"Running {len(configs)} configs: ranks={RANKS} x learning_rates={LEARNING_RATES}")

    for i, (rank, lr) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] rank={rank} learning_rate={lr:.0e}")
        run_one(train_set, valid_set, rank, lr)

    print("\nSweep complete. Run compare.py to evaluate and rank all configs.")


if __name__ == "__main__":
    main()
