"""Compare base vs fine-tuned model on the held-out test set.

Two metrics, logged to MLflow so results are comparable across future runs:
  - exact_match: assistant output string matches target JSON exactly
  - field_accuracy: fraction of the 4 schema fields that match, per example
    (JSON parse failures count all 4 fields wrong rather than crashing)
"""

import json

import mlflow
from mlx_lm import generate, load

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "qwen2.5-1.5b-json-extraction"
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ADAPTER_PATH = "adapters"
TEST_FILE = "data/test.jsonl"
FIELDS = ["task", "owner", "due_date", "priority"]


def load_test_set():
    rows = []
    with open(TEST_FILE) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def score(model, tokenizer, rows):
    exact_matches = 0
    field_hits = 0
    field_total = 0
    for row in rows:
        messages = row["messages"]
        system_and_user = messages[:2]
        target = messages[2]["content"]

        prompt = tokenizer.apply_chat_template(system_and_user, add_generation_prompt=True)
        output = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False).strip()

        if output == target:
            exact_matches += 1

        try:
            predicted = json.loads(output)
        except json.JSONDecodeError:
            predicted = {}
        expected = json.loads(target)
        for field in FIELDS:
            field_total += 1
            if predicted.get(field) == expected.get(field):
                field_hits += 1

    return {
        "exact_match": exact_matches / len(rows),
        "field_accuracy": field_hits / field_total,
    }


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    rows = load_test_set()

    with mlflow.start_run(run_name="eval-base-vs-finetuned"):
        print("Loading base model (no adapter)...")
        base_model, tokenizer = load(BASE_MODEL, tokenizer_config={"trust_remote_code": True})
        print(f"Scoring base model on {len(rows)} test examples...")
        base_scores = score(base_model, tokenizer, rows)
        print("Base:", base_scores)
        mlflow.log_metrics({f"base_{k}": v for k, v in base_scores.items()})

        del base_model

        print("Loading fine-tuned model (adapter attached)...")
        ft_model, tokenizer = load(
            BASE_MODEL, tokenizer_config={"trust_remote_code": True}, adapter_path=ADAPTER_PATH
        )
        print(f"Scoring fine-tuned model on {len(rows)} test examples...")
        ft_scores = score(ft_model, tokenizer, rows)
        print("Fine-tuned:", ft_scores)
        mlflow.log_metrics({f"finetuned_{k}": v for k, v in ft_scores.items()})

        mlflow.log_params({"test_examples": len(rows), "adapter_path": ADAPTER_PATH})


if __name__ == "__main__":
    main()
