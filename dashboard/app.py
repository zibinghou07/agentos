from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AgentOS", layout="wide")
st.title("AgentOS experiment dashboard")
result_path = Path("experiments/results/benchmark/runs.csv")
if not result_path.exists():
    st.info("Run `make benchmark` first. No numbers are hard-coded in this dashboard.")
else:
    runs = pd.read_csv(result_path)
    st.subheader("Method comparison")
    summary = runs.groupby("method").agg(
        success_rate=("success", "mean"), p95_latency_ms=("latency_ms", lambda x: x.quantile(0.95)),
        average_tool_calls=("tool_calls", "mean"), average_input_tokens=("input_tokens", "mean"),
    )
    st.dataframe(summary)
    st.bar_chart(summary["success_rate"])
    st.subheader("Failure analysis")
    st.dataframe(runs.loc[~runs["success"], ["method", "task_id", "family", "tool_selection_correct", "recovered"]])

