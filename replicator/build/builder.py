"""Stage 3 (Build): a bounded manual tool loop. The orchestrator owns the clock and the stop:
the model cannot end the stage by claiming done (DESIGN.md §9.4 add-back 2). The stage ends
when the smoke run passes its invariants or the budget runs out.

Tools (typed, cacheable, logged): list_files, read_file, write_file, run_python, run_smoke,
describe_data. The model never receives data rows; describe_data returns aggregates.
"""
from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from secprim import MalformedInput, contained_path

from ..data.adapters import describe
from ..llm import LLM
from ..schema import Spec
from .sandbox import Sandbox

BUILDER_SYSTEM = """You are implementing a replication pipeline from a frozen spec, inside a work
directory, using the family template already installed there. You work piecemeal: read the
template and the spec, write pipeline.py, run the smoke test, fix what fails, repeat.

Contract you must satisfy (the orchestrator checks it; saying you are done does nothing):
- Finance families: pipeline.py defines
    run(config: dict, seed: int) -> dict of metrics (keys named exactly as the claims' metric field),
    raw_data() -> pandas DataFrame of the raw input panel (columns include 'date'),
    build_features(raw: DataFrame) -> DataFrame with a 'date' column and the signal column(s).
  run() must honor config['_shuffle_labels'] by shuffling returns within each period.
- CS families: write reproduce.sh (bash) and the code it calls. reproduce.sh reads env SEED,
  SMOKE ('1' => tiny subset, one epoch), SCALE (float, default 1.0: multiply training steps /
  epochs / data by it; SCALE=0.1 must run the same pipeline at one tenth the cost), RUN_TIMEOUT_S,
  REPLICATOR_CONFIG (json), and writes METRICS_OUT as
  {"seed": int, "split": str, "n_examples": int, "scale": <the SCALE it ran at>, "shuffled": bool,
   "matched_scale": bool, "metrics": {<metric>: value, "chance_level": <no-skill value>}}.
  run_smoke also runs the script with SCALE=0.1 and with _shuffle_labels=true and checks these
  fields; a build cannot finish until they are right.
  For EVERY target claim in the spec, metrics must contain an entry keyed by the claim id holding
  the value that claim is compared against (e.g. "c_dim_tetrahedron": 0.75), in the claim's units.
  Several claims often share one metric name and differ only in what they measure; the claim-id
  key is what makes the comparison unambiguous.
  When REPLICATOR_CONFIG has "_shuffle_labels": true, shuffle the training labels (or targets /
  rewards where that is meaningful) within each batch or period BEFORE training and set
  "shuffled": true; if shuffling is meaningless for the method (no labels or rewards: toy models,
  causal tracing, generative-only), set "shuffle_not_applicable": true and say why in a comment.
  Always report a no-skill reference next to the headline metric so the shuffle test can judge:
  "chance_level" (e.g. 0.5 or 50 for balanced binary accuracy), or "<metric>_random_policy" for
  RL returns, or "<metric>_baseline" for a trivial baseline, in the same units as the metric.
  The full run (SCALE=1) MUST finish inside RUN_TIMEOUT_S on CPU; run_smoke times a SCALE=0.1 run
  and rejects the build if the extrapolated full run does not fit. Size the default accordingly.
  If the full run uses the paper's OWN configuration for a claim (same dataset, model size, and
  training length, e.g. LeNet-300-100 on MNIST, a 3-layer MLP on MNIST, CartPole), add
  "matched_scale": true to metrics.json so the comparison to the paper's number is made; leave it
  false when you scaled anything down.
- Every choice in the ambiguity list is a config key read from `config`, defaulting to the spec default.
- Use pandas/numpy/statsmodels/torch. No network calls except through the provided data adapters.
- Do not read the paper authors' code or clone repositories in re-implementation mode.
- Keep intermediates: return them under metrics['_intermediates'] when cheap (row counts, dates).
Work order: (1) read the template and README, (2) write reproduce.sh and the script it calls with
SMOKE=1 support first, (3) call run_smoke immediately, before polishing anything, (4) fix what it
reports, repeat. Calling run_smoke early is how you find contract mismatches cheaply; the stage
ends only when it passes. Keep each tool call small; do not write long explanations."""

TOOLS = [
    {"name": "list_files", "description": "List files in the work directory.", "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "read_file", "description": "Read a file in the work directory.", "strict": True,
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}},
    {"name": "write_file", "description": "Write (overwrite) a file in the work directory.", "strict": True,
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False}},
    {"name": "run_python", "description": "Run a python file in the work directory (timeout 5 min). Returns stdout/stderr tails.", "strict": True,
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"}}}, "required": ["path", "args"], "additionalProperties": False}},
    {"name": "run_smoke", "description": "Run the orchestrator's smoke test against pipeline.py / reproduce.sh: contract check, invariants, and metrics. This is the check that ends the stage when it passes.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "describe_data", "description": "Aggregates (row counts, date range, means, stds) of a cached data source. Never rows.", "strict": True,
     "input_schema": {"type": "object", "properties": {"source": {"type": "string"}, "params_json": {"type": "string"}}, "required": ["source", "params_json"], "additionalProperties": False}},
]


class Builder:
    def __init__(self, llm: LLM, spec: Spec, workdir: Path, smoke_fn: Callable[[], dict[str, Any]], budget_minutes: int):
        self.llm, self.spec, self.workdir, self.smoke_fn = llm, spec, workdir, smoke_fn
        self.sandbox = Sandbox(workdir)
        self.budget_s = budget_minutes * 60
        self.smoke_passed = False
        self.last_smoke: dict[str, Any] = {}
        self.turns = 0
        self.timeouts = 0

    # ------------------------------------------------------------------ tools
    def _tool(self, name: str, inp: dict) -> str:
        wd = self.workdir
        if name == "list_files":
            return "\n".join(str(p.relative_to(wd)) for p in sorted(wd.rglob("*")) if p.is_file() and "__pycache__" not in str(p))
        if name == "read_file":
            p = (wd / inp["path"]).resolve()
            if wd.resolve() not in p.parents and p != wd.resolve():
                return "error: path outside work dir"
            return p.read_text()[:60000] if p.exists() else "error: no such file"
        if name == "write_file":
            p = (wd / inp["path"]).resolve()
            if wd.resolve() not in p.parents:
                return "error: path outside work dir"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(inp["content"])
            return f"wrote {len(inp['content'])} chars to {inp['path']}"
        if name == "run_python":
            try:
                rel = str(contained_path(wd, inp["path"]).relative_to(wd))
            except MalformedInput as e:
                return f"error: {e}"
            r = self.sandbox.python_file(rel, inp.get("args") or [], timeout_s=300, tag="builder")
            return f"rc={r.returncode} ({r.seconds:.1f}s)\n--- stdout ---\n{r.stdout[-6000:]}\n--- stderr ---\n{r.stderr[-6000:]}"
        if name == "run_smoke":
            res = self.smoke_fn()
            self.last_smoke = res
            self.smoke_passed = bool(res.get("passed"))
            return json.dumps(res, indent=1, default=str)[:12000]
        if name == "describe_data":
            from ..data.adapters import ADAPTERS
            mod = ADAPTERS.get(inp["source"])
            if mod is None:
                return f"error: unknown source {inp['source']}"
            df, key = mod.fetch(json.loads(inp["params_json"]))
            return json.dumps({"cache_key": key, **describe(df)}, indent=1, default=str)
        return f"error: unknown tool {name}"

    # ------------------------------------------------------------------ loop
    def run(self) -> dict[str, Any]:
        t0 = time.time()
        spec_text = self.spec.model_dump_json(indent=1)
        from ..triage import COMPUTE_ENVELOPE as _env
        hw = (f"GPU {_env['gpu_name']} ({_env['gpu_mem_gb']} GB), use device='cuda'" if _env["gpu"] else "CPU only (no GPU)")
        envelope = (f"Compute envelope: {hw}, {_env['cores']} cores. Each full reproduce.sh run must "
                    f"finish within {self.spec.plan.budget.run_timeout_s // 60} minutes and the SMOKE=1 run within 5. "
                    f"Compute tier {self.spec.plan.compute_tier}: scale down (subset, fewer steps, smaller model) and "
                    f"record what you scaled in metrics['_intermediates']['scale']. Compute tier 1 means run the paper's own "
                    f"configuration and set matched_scale true. transformers/datasets/peft/"
                    f"gymnasium/ale-py are installed. Datasets and HF models cache under data_cache/.")
        messages: list[dict] = [{"role": "user", "content": textwrap.dedent(f"""
            Work directory contents are available via list_files. {envelope}
            Plan hint from triage: {self.spec.plan.success_criteria}
            The frozen spec:
            {spec_text}

            Start by reading the template and any README, then write pipeline.py (or reproduce.sh),
            then call run_smoke. Continue until run_smoke reports passed=true.""")}]
        while time.time() - t0 < self.budget_s:
            self.turns += 1
            resp = self.llm.tool_turn("build", BUILDER_SYSTEM, messages, TOOLS, effort="medium")
            if resp is None:  # timed out or connection dropped: an empty turn, not a lost stage
                self.timeouts += 1
                if self.timeouts >= 3:
                    break
                continue
            messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                # The model stopped calling tools. The stage does not end on its say-so.
                if self.smoke_passed:
                    break
                messages.append({"role": "user", "content": "The smoke test has not passed. Call run_smoke, read its output, and fix pipeline.py. Do not stop."})
                continue
            results = []
            for tu in tool_uses:
                try:
                    out = self._tool(tu.name, tu.input)
                except Exception as e:  # noqa: BLE001
                    out = f"error: {e}"
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out})
            messages.append({"role": "user", "content": results})
            if self.smoke_passed:
                break
        if not self.smoke_passed:
            # Budget expired (or the model stopped) without ever passing the gate: run it once on
            # whatever exists, so partial work is judged by the same check rather than lost.
            try:
                self.last_smoke = self.smoke_fn()
                self.smoke_passed = bool(self.last_smoke.get("passed"))
            except Exception as e:  # noqa: BLE001
                self.last_smoke = {"passed": False, "problems": [f"{type(e).__name__}: {e}"]}
        return {"smoke_passed": self.smoke_passed, "turns": self.turns, "timeouts": self.timeouts, "seconds": time.time() - t0,
                "blacklist_hits": self.sandbox.blacklist_hits, "last_smoke": self.last_smoke}
