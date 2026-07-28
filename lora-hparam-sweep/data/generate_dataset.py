"""Generate a small synthetic dataset for the toy fine-tuning task:
freeform note -> structured JSON {task, owner, due_date, priority}.

This exists purely to exercise the MLOps pipeline (data -> train -> track ->
eval -> serve) end to end on a task simple enough to eval by exact match.
"""

import json
import random
from pathlib import Path

random.seed(0)

SYSTEM_PROMPT = (
    "Extract a JSON object with exactly these keys from the note: "
    "task (string), owner (string or null), due_date (string or null), "
    "priority (one of \"low\", \"medium\", \"high\"). "
    "Respond with only the JSON object, no other text."
)

TASKS = [
    "fix the login bug", "write the release notes", "review the PR",
    "update the API docs", "renew the SSL cert", "deploy the staging build",
    "clean up the old feature flags", "onboard the new contractor",
    "migrate the database schema", "restock the office snacks",
    "send the invoice to the client", "prep slides for the demo",
    "rotate the API keys", "archive last quarter's tickets",
    "test the payment webhook", "refactor the auth middleware",
]

OWNERS = ["Alex", "Priya", "Sam", "Jordan", "Maya", "Chris", None, None]

DUE_PHRASES = [
    ("by Friday", "Friday"),
    ("by tomorrow", "tomorrow"),
    ("next week", "next week"),
    ("before the end of the month", "end of month"),
    ("by 3pm today", "today"),
    ("sometime next sprint", "next sprint"),
    (None, None),
    (None, None),
]

PRIORITY_PHRASES = [
    ("this is urgent", "high"),
    ("high priority", "high"),
    ("asap", "high"),
    ("no rush", "low"),
    ("whenever you get a chance", "low"),
    ("low priority", "low"),
    (None, "medium"),
    (None, "medium"),
]

TEMPLATES = [
    "{owner_clause}need to {task}{due_clause}{priority_clause}.",
    "Reminder: {task}{due_clause}{priority_clause}{owner_clause_trailing}.",
    "{owner_clause}Please {task}{due_clause}. {priority_sentence}",
    "Note to self: {task}{due_clause}{priority_clause}.",
    "{owner_clause}Can you {task}{due_clause}? {priority_sentence}",
]


def build_example():
    task = random.choice(TASKS)
    owner = random.choice(OWNERS)
    due_phrase, due_value = random.choice(DUE_PHRASES)
    priority_phrase, priority_value = random.choice(PRIORITY_PHRASES)

    due_clause = f" {due_phrase}" if due_phrase else ""
    priority_clause = f", {priority_phrase}" if priority_phrase else ""
    priority_sentence = priority_phrase.capitalize() + "." if priority_phrase else ""

    if owner:
        owner_clause = f"{owner}, "
        owner_clause_trailing = f" ({owner})"
    else:
        owner_clause = ""
        owner_clause_trailing = ""

    template = random.choice(TEMPLATES)
    note = template.format(
        owner_clause=owner_clause,
        owner_clause_trailing=owner_clause_trailing,
        task=task,
        due_clause=due_clause,
        priority_clause=priority_clause,
        priority_sentence=priority_sentence,
    ).strip()
    # collapse accidental double spaces/punctuation from empty clauses
    note = " ".join(note.split()).replace(" .", ".").replace(" ?", "?")

    target = {
        "task": task,
        "owner": owner,
        "due_date": due_value,
        "priority": priority_value,
    }

    return note, json.dumps(target, sort_keys=True)


def make_split(n):
    rows = []
    for _ in range(n):
        note, target_json = build_example()
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": note},
                    {"role": "assistant", "content": target_json},
                ]
            }
        )
    return rows


def write_jsonl(rows, path: Path):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    write_jsonl(make_split(200), out_dir / "train.jsonl")
    write_jsonl(make_split(30), out_dir / "valid.jsonl")
    write_jsonl(make_split(30), out_dir / "test.jsonl")
    print("Wrote train.jsonl (200), valid.jsonl (30), test.jsonl (30)")
