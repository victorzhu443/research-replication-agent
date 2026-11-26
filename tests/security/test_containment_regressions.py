"""Containment (P3) and boundary (P1) regressions the first mutation run showed
nothing caught (see security/mutations.yaml): rlimits actually reach the child, a
timeout kills the whole process group, the builder's `run_python` cannot leave the
work dir, and a sandboxed result cannot name a parquet side file by path.
"""
from __future__ import annotations

import time

import pandas as pd
import psutil
import pytest

from replicator.build import _runner
from replicator.build.builder import Builder
from replicator.build.sandbox import Sandbox


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setenv("REPLICATOR_SANDBOX_BACKEND", "local")
    wd = tmp_path / "work"
    wd.mkdir()
    return wd


# ---------------------------------------------------------------- P3 rlimits

def test_child_cannot_write_past_the_file_size_limit(work):
    """RLIMIT_FSIZE is set in the child (not advisory anywhere): a 4 MB write under a 1 MB cap fails."""
    sb = Sandbox(work, max_file_gb=1 / 1024, timeout_s=30)
    r = sb.python("open('big.bin', 'wb').write(b'x' * (4 * 1024 * 1024)); print('WROTE', flush=True)")
    assert "WROTE" not in r.stdout and r.returncode != 0, (r.returncode, r.stdout, r.stderr[-300:])
    assert (work / "big.bin").stat().st_size <= 1024 * 1024


def test_child_sees_finite_cpu_and_process_limits(work):
    sb = Sandbox(work, timeout_s=20, nproc=64)
    r = sb.python("import resource, json; print(json.dumps({n: resource.getrlimit(getattr(resource, n))[0] "
                  "for n in ('RLIMIT_CPU', 'RLIMIT_NPROC', 'RLIMIT_FSIZE')}))")
    assert r.returncode == 0, r.stderr
    import json
    lim = json.loads(r.stdout.strip().splitlines()[-1])
    assert 0 < lim["RLIMIT_CPU"] <= 20 * 64 + 30
    assert 0 < lim["RLIMIT_NPROC"] <= 64
    assert 0 < lim["RLIMIT_FSIZE"] <= sb.max_file_bytes


# ---------------------------------------------------------------- P3 process group

def test_timeout_kills_grandchildren_not_just_the_child(work):
    """A grandchild holding the stdout pipe must die with the group; killing only the direct child
    leaves it running (and the parent waiting on the pipe)."""
    pid_file = work / "grandchild.pid"
    r = Sandbox(work).python(
        "import subprocess, time, pathlib\n"
        "p = subprocess.Popen(['sleep', '120'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid))\n"
        "time.sleep(120)\n", timeout_s=1)
    assert r.returncode == 124
    assert r.seconds < 10, f"timeout took {r.seconds:.1f}s: the parent waited on a surviving grandchild"
    pid = int(pid_file.read_text())
    for _ in range(40):  # launchd reaps the orphan asynchronously
        if not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(0.05)
    else:
        psutil.Process(pid).kill()
        pytest.fail(f"grandchild {pid} survived the sandbox timeout")


# ---------------------------------------------------------------- P1 run_python containment

def _builder(work) -> Builder:
    b = Builder.__new__(Builder)  # the tool dispatcher needs only these two
    b.workdir = work
    b.sandbox = Sandbox(work)
    return b


@pytest.mark.parametrize("path", ["../evil.py", "/tmp/evil.py", "sub/../../evil.py"])
def test_run_python_refuses_paths_outside_the_work_dir(work, tmp_path, path):
    marker = tmp_path / "escaped.marker"
    evil = tmp_path / "evil.py"
    evil.write_text(f"open({str(marker)!r}, 'w').write('escaped')\n")
    if path.startswith("/"):
        path = str(evil)
    out = _builder(work)._tool("run_python", {"path": path})
    assert out.startswith("error:"), out
    assert not marker.exists(), "generated code outside the work dir was executed"


def test_run_python_still_runs_files_inside_the_work_dir(work):
    (work / "ok.py").write_text("print('inside')\n")
    out = _builder(work)._tool("run_python", {"path": "ok.py"})
    assert out.startswith("rc=0") and "inside" in out


# ---------------------------------------------------------------- P1 parquet side files

def test_result_parquet_side_file_cannot_point_outside_the_work_dir(work, tmp_path):
    outside = tmp_path / "outside.parquet"
    pd.DataFrame({"secret": [1, 2, 3]}).to_parquet(outside)
    for ref in ("../outside.parquet", str(outside)):
        with pytest.raises(FileNotFoundError):  # looked up as a bare name inside work/, where it is absent
            _runner.decode({"__parquet__": ref}, work)
    # and when a file of that bare name exists in the work dir, that is the one read
    pd.DataFrame({"x": [1]}).to_parquet(work / "outside.parquet")
    assert list(_runner.decode({"__parquet__": "../outside.parquet"}, work).columns) == ["x"]
