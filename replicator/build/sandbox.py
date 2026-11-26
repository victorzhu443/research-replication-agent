"""Subprocess sandbox for the builder. No Docker on this machine, so isolation is: a dedicated
work dir, a resource-limited subprocess, a wall-clock timeout, and a log of every command for
the author-code blacklist monitor. Swap for Docker/gVisor via the same interface in production."""
from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BLACKLIST = re.compile(r"(github\.com|gitlab\.com|git clone|huggingface\.co/[\w\-]+/[\w\-]+/blob)", re.I)

# The orchestrator's own LLM credentials. Model-authored code runs under the orchestrator's
# identity (subprocess inherits its env; the finance path runs it in-process outright), and
# network egress is open, so without scrubbing a generated program could read the Anthropic
# key out of os.environ and exfiltrate it. No replication pipeline ever needs these, so
# dropping them from every sandboxed/generated-code environment is free defense-in-depth.
SECRET_ENV_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def scrub_secrets(env: dict) -> dict:
    """Copy of env with the orchestrator's LLM-provider credentials removed."""
    return {k: v for k, v in env.items() if k not in SECRET_ENV_KEYS}


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    seconds: float


class Sandbox:
    def __init__(self, workdir: Path, mem_gb: float = 8.0, log_path: Path | None = None):
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.mem_bytes = int(mem_gb * 1024 ** 3)
        self.log_path = log_path or (workdir / "sandbox_log.jsonl")
        self.blacklist_hits: list[str] = []

    def _limits(self):
        try:
            resource.setrlimit(resource.RLIMIT_AS, (self.mem_bytes, self.mem_bytes))
        except (ValueError, OSError):
            pass

    def run(self, argv: list[str], timeout_s: int = 300, env: dict | None = None, tag: str = "") -> ExecResult:
        cmd = " ".join(argv)
        if BLACKLIST.search(cmd):
            self.blacklist_hits.append(cmd)
        t0 = time.time()
        try:
            p = subprocess.run(argv, cwd=self.workdir, capture_output=True, text=True, timeout=timeout_s,
                               env=scrub_secrets({**os.environ, **(env or {}), "PYTHONHASHSEED": "0"}),
                               preexec_fn=self._limits if sys.platform != "darwin" else None)
            res = ExecResult(p.returncode, p.stdout[-20000:], p.stderr[-20000:], time.time() - t0)
        except subprocess.TimeoutExpired as e:
            res = ExecResult(124, (e.stdout or b"")[-20000:].decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")[-20000:],
                             f"timeout after {timeout_s}s", time.time() - t0)
        with self.log_path.open("a") as f:
            f.write(json.dumps({"tag": tag, "cmd": cmd, "rc": res.returncode, "secs": round(res.seconds, 2)}) + "\n")
        return res

    def python(self, script: str, timeout_s: int = 300, tag: str = "") -> ExecResult:
        return self.run([sys.executable, "-c", script], timeout_s=timeout_s, tag=tag)

    def python_file(self, rel: str, args: list[str] | None = None, timeout_s: int = 600, tag: str = "") -> ExecResult:
        return self.run([sys.executable, rel, *(args or [])], timeout_s=timeout_s, tag=tag)
