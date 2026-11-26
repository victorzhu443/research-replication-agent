"""CS/ML template. The contract is a `reproduce.sh` in the work dir that writes `metrics.json`:
    {"seed": int, "split": str, "n_examples": int, "scale": float, "shuffled": bool,
     "matched_scale": bool, "metrics": {"accuracy": 98.2, "chance_level": 10.0, ...}}
The smoke gate runs the script three ways (SMOKE=1; SCALE=0.1; SMOKE=1 with _shuffle_labels) and
checks the fields the verify stage depends on before the build may finish.
One invocation per seed. The builder writes the code; this module only runs and parses.
Oracle-first: if the repo already has an eval script for released weights, `reproduce.sh` calls it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..build.sandbox import Sandbox


def run_reproduce(workdir: Path, config: dict[str, Any], seed: int, timeout_s: int = 1800,
                  python: str | None = None, sandbox: "Sandbox | None" = None) -> dict[str, Any]:
    """One `bash reproduce.sh` run through the Sandbox — the same boundary as every other exec path:
    allow-listed env, rlimits + RSS watchdog (local) or the container, and a wall clock that kills
    the whole process group (a grandchild holding the pipe cannot hang the parent)."""
    from ..build.sandbox import Sandbox
    workdir = workdir.resolve()
    sb = sandbox or Sandbox(workdir)
    out = workdir / f"metrics_seed{seed}.json"
    if out.exists():
        out.unlink()
    extra = {"SEED": str(seed), "REPLICATOR_CONFIG": json.dumps(config),
             "SMOKE": "1" if config.get("_smoke") else "0", "RUN_TIMEOUT_S": str(timeout_s),
             "SCALE": str(config.get("_scale", 1.0)), "TOKENIZERS_PARALLELISM": "false",
             "METRICS_OUT": str(out)}
    if python:
        extra["PYTHON"] = python
    r = sb.run(["bash", "reproduce.sh"], timeout_s=timeout_s, env=extra, tag=f"reproduce:seed{seed}")
    if r.returncode == 124:
        raise RuntimeError(f"reproduce.sh timed out after {timeout_s}s (seed {seed}); process group killed")
    if r.returncode != 0:
        raise RuntimeError(f"reproduce.sh failed (seed {seed}):\n{r.stderr[-3000:]}")
    if not out.exists():
        raise RuntimeError("reproduce.sh did not write METRICS_OUT")
    return json.loads(out.read_text())


def aggregate_seeds(results: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({k for r in results for k in r["metrics"]})
    out: dict[str, float] = {"n_seeds": float(len(results))}
    for k in keys:
        vals = np.array([r["metrics"][k] for r in results if k in r["metrics"]], dtype=float)
        out[k] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    if results and "n_examples" in results[0]:
        out["n_examples"] = float(results[0]["n_examples"])
    return out


def split_size_checkpoint(reported: int | None, ours: int | None, rel_tol: float = 0.02) -> tuple[bool | None, str]:
    if reported is None or ours is None:
        return None, "no split size to compare"
    gap = abs(ours - reported) / max(reported, 1)
    return gap <= rel_tol, f"reported {reported}, ours {ours}, rel gap {gap:.3%}"
