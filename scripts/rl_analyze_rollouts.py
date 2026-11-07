"""Training curves and reward-hacking checks from a verl run.

  python scripts/rl_analyze_rollouts.py results/rl_train/<EXP> --group 8

Reads <run>/rollouts/<step>.jsonl (verl rollout dumps) and <run>/train.log (verl
console metrics) and writes <run>/curves.csv, <run>/verl_metrics.csv,
<run>/hacking.json (counts per step + 3 examples per pattern) and <run>/curves.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentos.rl.reward import exact_match, f1_score, stats_from_transcript
from agentos.rl.tools import TOOL_NAMES

STEP_LINE = re.compile(r"step:(\d+) - (.*)")


def enrich(row: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Recompute episode fields from the rollout text and gold answers.

    verl 0.9.1's rollout dumps keep only input/output/gts/score, not the reward
    function's extra fields, so the same transcript parser as the reward fills them in.
    """
    stats = stats_from_transcript(row["output"])
    gts = row.get("gts") or []
    answers = [gts] if isinstance(gts, str) else [str(a) for a in gts]
    row.setdefault("tool_calls", stats.tool_calls)
    row.setdefault("invalid_calls", stats.invalid_calls)
    row.setdefault("obs_tokens", stats.obs_tokens)
    row.setdefault("assistant_turns", stats.assistant_turns)
    row.setdefault("no_answer", float(stats.answer is None))
    row.setdefault("r_em", exact_match(stats.answer, answers))
    row.setdefault("r_f1", f1_score(stats.answer, answers))
    for name in TOOL_NAMES:
        row.setdefault(f"calls_{name}", stats.tool_counts.get(name, 0))
    return row, stats.answer


def hacking_patterns(row: dict[str, Any], answer: str | None) -> list[str]:
    """Heuristic red flags; each match is inspected by hand before being called hacking."""
    flags = []
    if row.get("tool_calls", 0) >= 6:
        flags.append("tool_spam")  # many calls (mostly read-only search) per question
    if answer is not None and row.get("tool_calls", 0) == 0:
        flags.append("answer_without_search")
    if answer is not None and (len(answer.split()) > 8 or re.search(r"\bor\b|/|;", answer)):
        flags.append("hedged_or_long_answer")  # listing candidates to game matching
    if row.get("invalid_calls", 0) >= 3:
        flags.append("invalid_call_spam")
    return flags


def analyze(run: Path, group: int) -> dict[str, Any]:
    steps = sorted((run / "rollouts").glob("*.jsonl"), key=lambda p: int(p.stem))
    curves, counts, examples = [], [], defaultdict(list)
    for path in steps:
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        by_prompt: dict[str, list[float]] = defaultdict(list)
        flag_count: dict[str, int] = defaultdict(int)
        rows = [enrich(row)[0] for row in rows]
        for row in rows:
            by_prompt[row["input"]].append(row["score"])
            answer = stats_from_transcript(row["output"]).answer
            for flag in hacking_patterns(row, answer):
                flag_count[flag] += 1
                examples[flag].append({"step": row["step"], "score": row["score"], "gts": row["gts"], "input": row["input"][-400:], "output": row["output"]})
        groups = [scores for scores in by_prompt.values() if len(scores) >= 2]
        n = len(rows)
        mean = lambda key: sum(r.get(key, 0.0) for r in rows) / max(n, 1)  # noqa: E731
        curves.append(
            {
                "step": int(path.stem),
                "n": n,
                "reward": mean("score"),
                "em": mean("r_em"),
                "f1": mean("r_f1"),
                "no_answer": mean("no_answer"),
                "tool_calls": mean("tool_calls"),
                "obs_tokens": mean("obs_tokens"),
                "turns": mean("assistant_turns"),
                "zero_variance_groups": sum(len(set(s)) == 1 for s in groups) / max(len(groups), 1),
                "groups": len(groups),
                **{f"calls_{t}": mean(f"calls_{t}") for t in ("bm25_search", "dense_search", "get_document")},
            }
        )
        counts.append({"step": int(path.stem), **flag_count})
        if groups and len(groups) * group != n:
            print(f"note: step {path.stem} has {n} rollouts in {len(groups)} groups (expected group size {group})")
    return {"curves": curves, "counts": counts, "examples": {k: v[-3:] for k, v in examples.items()}}


def parse_verl_log(log: Path) -> list[dict[str, Any]]:
    rows = []
    if not log.exists():
        return rows
    for line in log.read_text(errors="ignore").splitlines():
        match = STEP_LINE.search(line)
        if not match:
            continue
        row: dict[str, Any] = {"step": int(match.group(1))}
        for item in match.group(2).split(" - "):
            key, _, value = item.partition(":")
            try:
                row[key.strip()] = float(value)
            except ValueError:
                continue
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row}, key=lambda k: (k != "step", k))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(run: Path, curves: list[dict[str, Any]], verl_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("reward", "reward"), ("em", "EM"), ("tool_calls", "tool calls"), ("zero_variance_groups", "zero-variance groups")]
    verl_keys = [k for k in ("actor/entropy", "actor/kl_loss", "response_length/mean", "num_turns/mean") if any(k in r for r in verl_rows)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    steps = [c["step"] for c in curves]
    for ax, (key, title) in zip(axes[0], panels):
        ax.plot(steps, [c[key] for c in curves])
        ax.set_title(title)
    for ax, key in zip(axes[1], verl_keys):
        points = [(r["step"], r[key]) for r in verl_rows if key in r]
        ax.plot([p[0] for p in points], [p[1] for p in points])
        ax.set_title(key)
    for ax in axes.flat:
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(run / "curves.png", dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--group", type=int, default=8)
    args = parser.parse_args()
    result = analyze(args.run, args.group)
    verl_rows = parse_verl_log(args.run / "train.log")
    write_csv(args.run / "curves.csv", result["curves"])
    write_csv(args.run / "verl_metrics.csv", verl_rows)
    (args.run / "hacking.json").write_text(json.dumps({"counts": result["counts"], "examples": result["examples"]}, indent=2, ensure_ascii=False))
    if result["curves"]:
        plot(args.run, result["curves"], verl_rows)
        last = result["curves"][-1]
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in last.items()}, indent=2))
    else:
        print("no rollout dumps found under", args.run / "rollouts")


if __name__ == "__main__":
    main()
