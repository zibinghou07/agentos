PYTHON ?= .venv/bin/python
UV ?= .venv/bin/uv

.PHONY: install test demo benchmark evaluate taskbench-data taskbench-dry taskbench-smoke taskbench real-data real-benchmark real-api-example e2e-example api api-check mcp-smoke
install:
	python3.11 -m venv .venv
	.venv/bin/python -m pip install uv==0.9.7
	$(UV) sync --extra dev --python .venv/bin/python

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) -m agentos.cli demo

benchmark:
	$(PYTHON) scripts/run_benchmark.py --repeats 3

evaluate:
	$(PYTHON) scripts/evaluate_results.py

real-data:
	$(PYTHON) scripts/download_real_benchmark.py

real-benchmark:
	$(PYTHON) scripts/run_real_benchmark.py

real-api-example:
	$(PYTHON) scripts/run_real_api_example.py

e2e-example:
	$(PYTHON) scripts/run_e2e_example.py

api:
	$(PYTHON) -m uvicorn agentos.api:app --reload

api-check:
	$(PYTHON) scripts/check_bailian.py

mcp-smoke:
	$(PYTHON) scripts/mcp_smoke.py


taskbench-data:
	$(PYTHON) -c "from agentos.taskbench.generator import write_taskbench; print(write_taskbench('benchmarks/taskbench'))"

taskbench-dry:
	$(PYTHON) scripts/run_taskbench.py --dry-run

taskbench-smoke:
	AGENTOS_ENABLE_THINKING=false $(PYTHON) scripts/run_taskbench.py --split dev --per-family 1 --output experiments/results/taskbench/dev_smoke/runs.jsonl

taskbench:
	AGENTOS_ENABLE_THINKING=false $(PYTHON) scripts/run_taskbench.py --split test --methods react,fixed_roles,agentos,agentos_static_routing,agentos_no_recovery
