"""Evaluate every adapter from the sweep on the held-out test set and rank
them. val_loss during training tells you what converged; it doesn't tell
you which config actually nails the task -- this does.
"""

import json

import mlflow
from mlx_lm import generate, load

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "qwen2.5-1.5b-json-extraction-sweep"
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
TEST_FILE = "data/test.jsonl"
FIELDS = ["task", "owner", "due_date", "priority"]


def load_test_set():
    with open(TEST_FILE) as f:
        return [json.loads(line) for line in f]


def score(model, tokenizer, rows):
    exact_matches = 0
    field_hits = 0
    field_total = 0
    for row in rows:
        messages = row["messages"]
        target = messages[2]["content"]
        prompt = tokenizer.apply_chat_template(messages[:2], add_generation_prompt=True)
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
    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"No experiment '{EXPERIMENT_NAME}' found. Run sweep.py first.")

    runs = client.search_runs([exp.experiment_id], order_by=["start_time ASC"])
    print(f"Found {len(runs)} runs to evaluate.\n")

    rows = load_test_set()
    results = []

    for run in runs:
        params = run.data.params
        adapter_path = params["adapter_path"]
        rank, lr = params["rank"], float(params["learning_rate"])

        print(f"Evaluating rank={rank} lr={lr:.0e} ({adapter_path})...")
        model, tokenizer = load(
            BASE_MODEL, tokenizer_config={"trust_remote_code": True}, adapter_path=adapter_path
        )
        scores = score(model, tokenizer, rows)
        print(f"  exact_match={scores['exact_match']:.3f} field_accuracy={scores['field_accuracy']:.3f}")

        client.log_metric(run.info.run_id, "test_exact_match", scores["exact_match"])
        client.log_metric(run.info.run_id, "test_field_accuracy", scores["field_accuracy"])

        results.append(
            {
                "run_id": run.info.run_id,
                "rank": rank,
                "learning_rate": lr,
                "final_val_loss": run.data.metrics.get("val_loss"),
                **scores,
            }
        )

    results.sort(key=lambda r: (-r["field_accuracy"], -r["exact_match"], r["final_val_loss"]))

    print("\n" + "=" * 78)
    print(f"{'rank':>4} {'lr':>9} {'val_loss':>10} {'exact_match':>13} {'field_acc':>11}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['rank']:>4} {r['learning_rate']:>9.0e} {r['final_val_loss']:>10.4f} "
            f"{r['exact_match']:>13.3f} {r['field_accuracy']:>11.3f}"
        )
    print("=" * 78)

    winner = results[0]
    print(
        f"\nBest config: rank={winner['rank']} learning_rate={winner['learning_rate']:.0e} "
        f"(exact_match={winner['exact_match']:.3f}, field_accuracy={winner['field_accuracy']:.3f})"
    )


if __name__ == "__main__":
    main()
