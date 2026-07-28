"""Assemble the GitHub Pages site from the repo's lab subfolders.

Any top-level directory containing docs/index.html is treated as a lab
with a published page -- no hardcoded lab list to keep in sync. The hub
page's card text comes straight from each lab page's own <title> and
<meta name="description">, so there's one source of truth per lab, not two.
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE_TEMPLATE = ROOT / "site" / "index.html"
OUT_DIR = ROOT / "_site"
SKIP_DIRS = {".git", ".github", "site", "_site"}


def extract(html: str, pattern: str, default: str) -> str:
    m = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else default


def find_labs():
    labs = []
    for entry in sorted(ROOT.iterdir()):
        if not entry.is_dir() or entry.name in SKIP_DIRS:
            continue
        docs_index = entry / "docs" / "index.html"
        if docs_index.exists():
            labs.append((entry.name, docs_index))
    return labs


def main():
    labs = find_labs()
    if not labs:
        print("No labs with docs/index.html found.")

    OUT_DIR.mkdir(exist_ok=True)

    cards = []
    for name, docs_index in labs:
        html = docs_index.read_text()
        title = extract(html, r"<title>(.*?)</title>", name)
        description = extract(
            html, r'<meta\s+name="description"\s+content="(.*?)"', ""
        )
        cards.append(
            f'<a class="lab-card" href="./{name}/">'
            f'<span class="arrow">&rarr;</span>'
            f'<p class="lab-name">{title}</p>'
            f'<p class="lab-desc">{description}</p>'
            f"</a>"
        )

        dest = OUT_DIR / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(docs_index.parent, dest)
        print(f"Included lab: {name} -> /{name}/")

    template = SITE_TEMPLATE.read_text()
    hub_html = template.replace(
        "<!--LABS-->",
        "\n      ".join(cards) if cards else '<p class="lab-desc">No lessons published yet.</p>',
    )
    (OUT_DIR / "index.html").write_text(hub_html)
    print(f"Wrote hub page with {len(cards)} lab card(s) to {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
