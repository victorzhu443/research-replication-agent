"""The Sandbox is the one execution boundary for model-authored code (DESIGN §3.3).

`Sandbox.call` replaces importing `pipeline.py` into the orchestrator process: arguments and
results cross as JSON/parquet, the child sees an allow-listed environment, memory is bounded by a
watchdog that works on macOS, and a local backend announces itself in the run log."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from replicator.build import sandbox as sb_mod
from replicator.build.sandbox import (BANNER_LOCAL, BLACKLIST, PROVENANCE_MONITOR, Sandbox, SandboxCallError,
                                      sandbox_env)

PIPELINE = '''
import os
import numpy as np
import pandas as pd

def run(config, seed=0):
    return {"sharpe": np.float64(1.25), "n_periods": np.int64(120), "seed": seed, "k": config.get("k"),
            "_intermediates": {"first_date": "1990-01-31"}}

def raw_data():
    return pd.DataFrame({"date": pd.date_range("2000-01-31", periods=6, freq="ME"), "x": [1.0, 2, 3, 4, 5, 6]})

def build_features(raw):
    df = raw.sort_values("date").copy()
    df["x_lag"] = df["x"].shift(1)
    return df[["date", "x_lag"]]

def env_keys():
    return sorted(os.environ)

def boom():
    raise ValueError("boom")
'''


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setenv("REPLICATOR_SANDBOX_BACKEND", "local")
    wd = tmp_path / "work"
    wd.mkdir()
    (wd / "pipeline.py").write_text(PIPELINE)
    return wd


def test_call_round_trips_dict_and_dataframe(work):
    sb = Sandbox(work)
    metrics = sb.call("pipeline", "run", {}, args=[{"k": 3}, 7])
    assert metrics["sharpe"] == 1.25 and metrics["n_periods"] == 120 and metrics["seed"] == 7 and metrics["k"] == 3
    assert metrics["_intermediates"] == {"first_date": "1990-01-31"}
    raw = sb.call("pipeline", "raw_data", {})
    assert isinstance(raw, pd.DataFrame) and list(raw.columns) == ["date", "x"] and len(raw) == 6
    feats = sb.call("pipeline", "build_features", {}, args=[raw])  # a DataFrame argument travels as parquet
    assert list(feats.columns) == ["date", "x_lag"] and feats["x_lag"].iloc[1] == 1.0
    assert sb.call("_runner", "has_fns", {"module": "pipeline", "fns": ["run", "raw_data", "nope"]}) == \
        {"run": True, "raw_data": True, "nope": False}
    assert not list(work.glob("_call_*")) and not list(work.glob("_result_*"))  # per-call files are cleaned up


def test_call_reports_child_exception(work):
    with pytest.raises(SandboxCallError, match="ValueError: boom"):
        Sandbox(work).call("pipeline", "boom", {})


def test_concurrent_calls_do_not_cross(work):
    """The verify grid runs flips in a thread pool; each call must get its own result."""
    sb = Sandbox(work)
    with ThreadPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(lambda i: sb.call("pipeline", "run", {}, args=[{"k": i}, i]), range(6)))
    assert [o["k"] for o in outs] == list(range(6)) and [o["seed"] for o in outs] == list(range(6))


def test_child_env_is_allow_listed(work, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SENTINEL")
    monkeypatch.setenv("MY_SERVICE_TOKEN", "SENTINEL")       # a secret nobody thought to deny
    monkeypatch.setenv("RANDOM_PARENT_VAR", "leaks-by-default")
    keys = set(Sandbox(work).call("pipeline", "env_keys", {}))
    assert {"ANTHROPIC_API_KEY", "MY_SERVICE_TOKEN", "RANDOM_PARENT_VAR"}.isdisjoint(keys)
    assert {"PATH", "HOME", "PYTHONPATH", "PYTHONHASHSEED"} <= keys


def test_sandbox_env_rejects_secret_looking_extra(tmp_path):
    with pytest.raises(AssertionError):
        sandbox_env(tmp_path, {"HF_TOKEN": "x"})
    env = sandbox_env(tmp_path, {"SEED": "1", "TOKENIZERS_PARALLELISM": "false"})
    assert env["SEED"] == "1" and "ANTHROPIC_API_KEY" not in env


def test_local_backend_writes_banner(work):
    sb = Sandbox(work)
    assert sb.backend == "local"
    assert BANNER_LOCAL in sb.log_path.read_text()
    assert "NOT restricted" in BANNER_LOCAL


def test_banner_reaches_run_logger(work):
    lines: list[str] = []
    Sandbox(work, logger=lines.append)
    assert lines == [BANNER_LOCAL]


def test_rss_watchdog_kills_over_limit(work):
    """Enforced on macOS too: RLIMIT_AS is advisory there, the watchdog is not."""
    sb = Sandbox(work, mem_gb=0.06, timeout_s=30)
    r = sb.python("b = bytearray(300 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096])\nimport time; time.sleep(20)")
    assert r.returncode != 0 and "memory" in r.stderr and r.seconds < 15


def test_timeout_kills_process_group(work):
    r = Sandbox(work).python("import subprocess, time; subprocess.Popen(['sleep', '60']); time.sleep(60)", timeout_s=1)
    assert r.returncode == 124 and "timeout" in r.stderr


def test_backend_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("REPLICATOR_SANDBOX_BACKEND", raising=False)
    monkeypatch.setattr(sb_mod, "docker_image_available", lambda image: False)
    assert Sandbox(tmp_path / "a").backend == "local"
    with pytest.raises(RuntimeError, match="container"):
        Sandbox(tmp_path / "b", backend="container")
    monkeypatch.setattr(sb_mod, "docker_image_available", lambda image: True)
    assert Sandbox(tmp_path / "c").backend == "container"
    monkeypatch.setenv("REPLICATOR_SANDBOX_BACKEND", "local")
    assert Sandbox(tmp_path / "d", backend="container").backend == "local"  # the flag is the rollback


def test_container_command_shape(monkeypatch, tmp_path):
    monkeypatch.setenv("REPLICATOR_SANDBOX_BACKEND", "local")
    sb = Sandbox(tmp_path / "w", mem_gb=2)
    argv = sb._container_argv([os.sys.executable, "_runner.py", "_call_x.json"], sb.env({"SEED": "1"}))
    joined = " ".join(argv)
    for flag in ("--rm", "--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                 "--memory 2048m", "--pids-limit 64", "--tmpfs /tmp", "-w /work", "--env-file"):
        assert flag in joined, flag
    assert f"{(tmp_path / 'w').resolve()}:/work:rw" in argv
    assert argv[-3:] == ["python", "_runner.py", "_call_x.json"]
    env_file = argv[argv.index("--env-file") + 1]
    lines = open(env_file).read().splitlines()
    assert "SEED=1" in lines and "PYTHONPATH=/work:/app" in lines and not any(l.startswith("ANTHROPIC") for l in lines)


def test_provenance_monitor_keeps_old_name_and_behavior(work):
    assert BLACKLIST is PROVENANCE_MONITOR
    sb = Sandbox(work)
    sb.run(["echo", "git clone https://github.com/x/y"], tag="t")
    assert sb.blacklist_hits and "github.com" in sb.blacklist_hits[0]
    rec = [json.loads(l) for l in sb.log_path.read_text().splitlines()][-1]
    assert rec["tag"] == "t" and rec["backend"] == "local"
