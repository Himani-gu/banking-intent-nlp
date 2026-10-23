"""Collect results/*/metrics.json into a markdown table + figures and refresh the README.

    python scripts/make_report.py

Writes:
    results/results_table.md
    results/figures/model_comparison.png      (full-data runs)
    results/figures/data_efficiency.png       (accuracy vs. training-set size)
and replaces the block between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FAMILY_ORDER = {"tfidf_lr": 0, "transformer": 1}


def load_runs():
    runs = []
    for path in sorted(RESULTS.glob("*/metrics.json")):
        runs.append(json.loads(path.read_text()))
    runs.sort(key=lambda r: (FAMILY_ORDER.get(r.get("family"), 9), r.get("display_name", ""), r.get("fraction", 1)))
    return runs


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_table(runs) -> str:
    header = (
        "| Model | Training data | Accuracy | Macro-F1 | Top-3 acc. | Acc. with 5% typos | "
        "Typo drop | Latency p50 / p95 (ms) | Device |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in runs:
        frac = r.get("fraction", 1.0)
        rows.append(
            f"| {r.get('display_name', r['name'])} "
            f"| {frac * 100:g}% ({r['n_train']:,}) "
            f"| {pct(r['accuracy'])} | {pct(r['macro_f1'])} | {pct(r['top3_accuracy'])} "
            f"| {pct(r['noisy_accuracy'])} | -{r['noisy_accuracy_drop'] * 100:.1f} pts "
            f"| {r['latency_ms_p50']:.1f} / {r['latency_ms_p95']:.1f} "
            f"| {r.get('device', '?')} |"
        )
    return header + "\n".join(rows) + "\n"


def make_figures(runs) -> None:
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    full = [r for r in runs if r.get("fraction", 1.0) == 1.0]
    if full:
        metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("noisy_accuracy", "Acc. with 5% typos")]
        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.8 / len(full)
        for i, r in enumerate(full):
            xs = [j + i * width for j in range(len(metrics))]
            vals = [r[m] * 100 for m, _ in metrics]
            bars = ax.bar(xs, vals, width, label=r.get("display_name", r["name"]))
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=8)
        ax.set_xticks([j + width * (len(full) - 1) / 2 for j in range(len(metrics))])
        ax.set_xticklabels([label for _, label in metrics])
        lo = min(r[m] for r in full for m, _ in metrics) * 100
        ax.set_ylim(max(0, lo - 8), 100)
        ax.set_ylabel("%")
        ax.set_title("Banking77 test set (3,080 queries, 77 intents)")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(fig_dir / "model_comparison.png", dpi=150)
        plt.close(fig)

    by_family = {}
    for r in runs:
        by_family.setdefault(r.get("display_name", r["name"]), []).append(r)
    if any(len(v) > 1 for v in by_family.values()):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for name, rs in by_family.items():
            rs = sorted(rs, key=lambda r: r["n_train"])
            ax.plot([r["n_train"] for r in rs], [r["accuracy"] * 100 for r in rs], marker="o", label=name)
        ax.set_xscale("log")
        ax.set_xlabel("Training examples (log scale)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Data efficiency")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "data_efficiency.png", dpi=150)
        plt.close(fig)


def update_readme(table: str) -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    block = f"<!-- RESULTS:START -->\n{table}<!-- RESULTS:END -->"
    text = readme.read_text()
    new, n = re.subn(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: block, text, flags=re.S)
    if n:
        readme.write_text(new)
    else:
        print("README.md has no RESULTS markers; skipped.", file=sys.stderr)


def main() -> None:
    runs = load_runs()
    if not runs:
        sys.exit("No results found. Run the baseline / transformer training first.")
    table = build_table(runs)
    (RESULTS / "results_table.md").write_text(table)
    make_figures(runs)
    update_readme(table)
    print(table)


if __name__ == "__main__":
    main()
