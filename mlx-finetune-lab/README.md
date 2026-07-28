# mlx-finetune-lab

Local LoRA fine-tuning pipeline for small LLMs on Apple Silicon, built to
learn the MLOps loop end to end: data -> train -> track -> eval -> serve.

Toy task: extract structured JSON (`task`, `owner`, `due_date`, `priority`)
from a freeform note. Base model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.

## Pipeline

```
uv sync                                # install deps into .venv
uv run python data/generate_dataset.py # regenerate train/valid/test.jsonl
uv run python train.py                 # LoRA fine-tune, tracked in MLflow
uv run python eval.py                  # base vs fine-tuned on test set
uv run python -m mlx_lm fuse \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path adapters --save-path fused-model
uv run python -m mlx_lm server --model fused-model --port 8765
```

## Inspect experiment tracking

```
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Layout

- `data/generate_dataset.py` — synthetic dataset generator (deterministic, seeded)
- `train.py` — LoRA fine-tune via `mlx_lm.lora.train_model`, custom `MLflowCallback`
  logs params/loss curves/adapter artifact per run
- `eval.py` — exact-match and field-accuracy scoring, base vs fine-tuned
- `adapters/`, `fused-model/`, `mlruns/`, `mlflow.db` — generated, gitignored
