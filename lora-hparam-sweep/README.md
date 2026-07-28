# lora-hparam-sweep

Systematic hyperparameter comparison for LoRA fine-tuning, building on
[`mlx-finetune-lab`](../mlx-finetune-lab). Same toy task and base model
(`mlx-community/Qwen2.5-1.5B-Instruct-4bit`, freeform note -> structured
JSON) so results are directly comparable to that lab's single baseline run
-- the question here isn't "does fine-tuning work" but "which config wins,
and how do you know."

## Why this exists

A single fine-tuning run tells you whether *a* config works. It doesn't
tell you whether a different rank or learning rate would have converged
faster, generalized better, or been more stable. This lab runs a grid,
logs every config as its own MLflow run, and evaluates every resulting
adapter on the same held-out test set -- so "best" is a ranked table, not
a guess from one val_loss curve.

## Sweep

Grid: LoRA rank &times; learning rate, `num_layers=8` and `iters=80` fixed.

| | |
|---|---|
| rank | 4, 8, 16 |
| learning_rate | 3e-5, 1e-4, 3e-4 |

```
uv sync
uv run python data/generate_dataset.py
uv run python sweep.py     # ~9 runs, logged to the *-sweep MLflow experiment
uv run python compare.py   # evaluates every adapter on the test set, ranks them
```

`compare.py` logs `test_exact_match` / `test_field_accuracy` back onto each
sweep run in MLflow (so val_loss and test-set accuracy live side by side
per run) and prints a ranked leaderboard with the winning config.

## Inspect

```
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Layout

- `data/generate_dataset.py` -- same generator as mlx-finetune-lab (copied for self-containment)
- `sweep.py` -- grid search, one MLflow run per (rank, learning_rate) pair
- `compare.py` -- test-set evaluation + ranked leaderboard across all sweep runs
- `adapters/`, `mlruns/`, `mlflow.db` -- generated, gitignored
