# mlops-lessons

Exercises learning MLOps and the tooling around it, run locally on a
16GB Apple Silicon MacBook Air. Each subfolder is a self-contained project.

## Lessons

- [`mlx-finetune-lab`](mlx-finetune-lab/) — LoRA fine-tuning pipeline for
  small LLMs on Apple Silicon (MLX + uv + MLflow): data, train, track,
  eval, serve. See its [docs](mlx-finetune-lab/docs/index.html) or
  [README](mlx-finetune-lab/README.md).
- [`lora-hparam-sweep`](lora-hparam-sweep/) — systematic hyperparameter
  comparison: grid search over LoRA rank and learning rate, every config
  logged as its own MLflow run, evaluated and ranked on a held-out test
  set. See its [README](lora-hparam-sweep/README.md).
