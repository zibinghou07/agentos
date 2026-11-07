"""Recompute every reported RL number from saved trajectories and verl dumps.

  .venv/bin/python scripts/rl_summarize.py

Writes results/rl_summary.json. Besides strict EM (answer inside <answer> tags) it
reports lenient EM: if no tag was written, the final reply counts when it contains the
gold answer (yes/no must be the first word). Lenient EM separates "learned the answer
format" from "found the right answer".
"""

from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentos.rl.actions import TOOL_CALL
from agentos.rl.data import normalize_answer
from agentos.rl.reward import exact_match, f1_score, stats_from_transcript

EVAL = Path("results/rl_eval")
TRAIN = Path("results/rl_train")


def lenient_hit(answer: str | None, final_text: str, gold: list[str]) -> float:
    if answer is not None:
        return exact_match(answer, gold)
    text = normalize_answer(TOOL_CALL.sub(" ", final_text or ""))
    for g in gold:
        g = normalize_answer(g)
        if not g:
            continue
        if g in {"yes", "no"}:
            if text.split()[:1] == [g]:
                return 1.0
        elif re.search(rf"(^| ){re.escape(g)}( |$)", text):
            return 1.0
    return 0.0


def final_reply(trajectory: dict[str, Any]) -> str:
    turns = trajectory.get("turns") or []
    return (turns[-1]["assistant"].get("content") or "") if turns else ""


def summarize_eval(tag: str, split: str) -> dict[str, Any] | None:
    files = sorted((EVAL / tag / split).glob("seed*.jsonl"))
    if not files:
        return None
    per_seed = []
    for path in files:
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        n = len(rows)
        em = [(r["reward"] or {}).get("em", 0.0) for r in rows]
        per_seed.append(
            {
                "seed": path.stem,
                "n": n,
                "em": sum(em) / n,
                "em_count": f"{int(sum(em))}/{n}",
                "lenient_em": sum(lenient_hit(r["answer"], final_reply(r), r["gold"]) for r in rows) / n,
                "f1": st.mean((r["reward"] or {}).get("f1", 0.0) for r in rows),
                "no_answer": sum(r["answer"] is None for r in rows) / n,
                "tool_calls": st.mean(r["stats"]["tool_calls"] for r in rows),
                "obs_tokens": st.mean(r["stats"]["obs_tokens"] for r in rows),
                "turns": st.mean(r["stats"]["assistant_turns"] for r in rows),
                "api_errors": sum(r.get("error") is not None for r in rows),
            }
        )
    out: dict[str, Any] = {"tag": tag, "split": split, "seeds": len(per_seed), "n": per_seed[0]["n"], "per_seed": per_seed}
    for key in ("em", "lenient_em", "f1", "no_answer", "tool_calls", "obs_tokens", "turns"):
        values = [s[key] for s in per_seed]
        out[key] = {"mean": st.mean(values), "std": st.stdev(values) if len(values) > 1 else 0.0}
    return out


def summarize_verl_validation(run: str) -> list[dict[str, Any]]:
    rows_out = []
    for path in sorted((TRAIN / run / "validation").glob("*.jsonl"), key=lambda p: int(p.stem)):
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        stats = [stats_from_transcript(r["output"]) for r in rows]
        gold = [[r["gts"]] if isinstance(r["gts"], str) else [str(g) for g in r["gts"]] for r in rows]
        n = len(rows)
        rows_out.append(
            {
                "step": int(path.stem),
                "n": n,
                "em": sum(exact_match(s.answer, g) for s, g in zip(stats, gold)) / n,
                "f1": st.mean(f1_score(s.answer, g) for s, g in zip(stats, gold)),
                "no_answer": sum(s.answer is None for s in stats) / n,
                "tool_calls": st.mean(s.tool_calls for s in stats),
                "reward": st.mean(r["score"] for r in rows),
            }
        )
    return rows_out


def summarize_training(run: str) -> list[dict[str, float]]:
    path = TRAIN / run / "curves.csv"
    if not path.exists():
        return []
    import csv

    with path.open() as handle:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def main() -> None:
    evals = defaultdict(dict)
    for tag_dir in sorted(EVAL.iterdir()):
        for split_dir in sorted(p for p in tag_dir.iterdir() if p.is_dir()):
            summary = summarize_eval(tag_dir.name, split_dir.name)
            if summary:
                evals[tag_dir.name][split_dir.name] = summary
    report = {
        "evaluations": evals,
        "verl_validation": {run: summarize_verl_validation(run) for run in ("grpo_full", "grpo_v11")},
        "training_curves": {run: summarize_training(run) for run in ("smoke", "grpo_full", "grpo_v11")},
    }
    Path("results/rl_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for tag in ("pre_rl_v2", "post_rl_v11_step50_v2", "pre_rl_closed_book"):
        for split, s in evals.get(tag, {}).items():
            print(
                f"{tag:22s} {split:12s} n={s['n']} seeds={s['seeds']} | EM {s['em']['mean']:.3f}+/-{s['em']['std']:.3f}"
                f" | lenient {s['lenient_em']['mean']:.3f} | F1 {s['f1']['mean']:.3f} | no_ans {s['no_answer']['mean']:.2f}"
                f" | calls {s['tool_calls']['mean']:.2f} | obs {s['obs_tokens']['mean']:.0f}"
            )
    for run, rows in report["verl_validation"].items():
        for r in rows:
            print(f"verl-val {run:10s} step {r['step']:3d} n={r['n']} | EM {r['em']:.3f} | F1 {r['f1']:.3f} | no_ans {r['no_answer']:.2f} | calls {r['tool_calls']:.2f} | reward {r['reward']:.3f}")


if __name__ == "__main__":
    main()
