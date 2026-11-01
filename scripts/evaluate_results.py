from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "data/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from agentos.benchmarks.metrics import mcnemar

directory = Path("experiments/results/benchmark")
frame = pd.read_csv(directory / "runs.csv")
means = frame.groupby("method")["success"].mean().sort_values()
ax = means.plot.barh(title="Tool-routing microbenchmark (offline, no LLM)")
ax.set_xlabel("Task success rate")
plt.tight_layout()
plt.savefig(directory / "success_rate.png", dpi=160)
plt.close()

pivot = frame.loc[frame["run_id"] == frame["run_id"].min()].pivot_table(
    index="task_id", columns="method", values="success", aggfunc="first"
)
stats = mcnemar(pivot["keyword_rules"].astype(bool).tolist(), pivot["clause_routing"].astype(bool).tolist())
(directory / "statistical_tests.json").write_text(json.dumps({"mcnemar_keyword_rules_vs_clause_routing": stats}, indent=2), encoding="utf-8")
print(json.dumps(stats, indent=2))

