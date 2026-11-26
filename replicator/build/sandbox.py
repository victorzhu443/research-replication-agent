"""The one execution boundary for model-authored code.

Every path that runs generated code (the builder's `run_python` tool, the finance
`pipeline.py`, the CS `reproduce.sh`) goes through this module. What it enforces:

- an **allow-listed environment** built from nothing (`secprim.child_env`), so no
  credential reaches the child by omission; `scrub_secrets` + `assert_no_secrets`
  remain as a second check right before exec;
- a throwaway HOME per sandbox, with the ML caches pinned to `data_cache/`;
- resource limits in the child (`RLIMIT_CPU/NPROC/FSIZE/AS` where the platform
  supports each) plus an **RSS watchdog thread** that kills the process group when
  resident memory exceeds `mem_gb` — enforced on macOS, where rlimits are not;
- a wall-clock timeout that kills the whole process group;
- a `container` backend (`docker run --network none --read-only ...`) chosen by
  `auto` when Docker and the image exist; otherwise `local`, announced with a
  one-line banner in the run log so the lack of network restriction is never silent;
- `.call()`: a file-based RPC (JSON + parquet via `_runner.py`) that replaces
  importing `pipeline.py` into the orchestrator's own process.

`PROVENANCE_MONITOR` is not a security control: it only records commands that
look like they fetch author code so the verdict can be downgraded from
re-implementation to reproduction.
"""
from __future__ import annotations

import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import psutil
from secprim import assert_no_secrets, child_env
from secprim.child_env import DEFAULT_ALLOW

from . import _runner

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CACHE = REPO_ROOT / "data_cache"

# Author-code isolation signal (see module docstring). Same pattern as before, honest name.
PROVENANCE_MONITOR = re.compile(r"(github\.com|gitlab\.com|git clone|huggingface\.co/[\w\-]+/[\w\-]+/blob)", re.I)
BLACKLIST = PROVENANCE_MONITOR  # compatibility alias

# The orchestrator's own LLM credentials. The allow-list below already excludes them; this
# denylist is kept as the belt-and-braces check (and for callers that build their own env).
SECRET_ENV_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

# What a replication child may see, beyond secprim's baseline (PATH, locale, TZ, TERM):
# cache locations, thread/device knobs, and the per-run values callers set explicitly.
SANDBOX_ALLOW = frozenset(DEFAULT_ALLOW | {
    "TMPDIR", "VIRTUAL_ENV", "HF_HOME", "HF_HUB_OFFLINE", "TORCH_HOME", "XDG_CACHE_HOME",
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM",
})
# Keys that match secprim's secret-looking regex but carry no credential.
_BENIGN_SECRET_LOOKING = frozenset({"TOKENIZERS_PARALLELISM"})

BANNER_LOCAL = "sandbox backend=local: network is NOT restricted"
DEFAULT_IMAGE = "replicator-sandbox:latest"
Backend = Literal["auto", "container", "local"]
_TAIL = 20000


def scrub_secrets(env: dict) -> dict:
    """Copy of env with the orchestrator's LLM-provider credentials removed."""
    return {k: v for k, v in env.items() if k not in SECRET_ENV_KEYS}


def sandbox_env(workdir: Path, extra: dict[str, str] | None = None, *, home: str | Path | None = None) -> dict[str, str]:
    """Allow-listed environment for generated code. `extra` values are explicit per-run settings
    (SEED, REPLICATOR_CONFIG, ...); nothing else from the parent gets through."""
    defaults = {"PYTHONPATH": os.pathsep.join([str(workdir), str(REPO_ROOT)]), "PYTHONHASHSEED": "0"}
    for k, v in (("HF_HOME", DATA_CACHE / "hf"), ("TORCH_HOME", DATA_CACHE / "torch")):
        if k not in os.environ:
            defaults[k] = str(v)
    env = child_env(allow=SANDBOX_ALLOW, set={**defaults, **(extra or {})}, home=str(home) if home else None)
    env = scrub_secrets(env)
    assert_no_secrets({k: v for k, v in env.items() if k not in _BENIGN_SECRET_LOOKING})
    return env


def _killpg(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


class _RssWatchdog(threading.Thread):
    """Polls resident memory of the child and its descendants; kills the group over `limit`."""

    def __init__(self, proc: subprocess.Popen, limit: int, interval: float = 0.25):
        super().__init__(daemon=True)
        self.proc, self.limit, self.interval = proc, limit, interval
        self.tripped: int | None = None
        self._stop = threading.Event()

    def run(self) -> None:
        try:
            root = psutil.Process(self.proc.pid)
        except psutil.Error:
            return
        while not self._stop.wait(self.interval):
            try:
                rss = root.memory_info().rss
                for c in root.children(recursive=True):
                    try:
                        rss += c.memory_info().rss
                    except psutil.Error:
                        continue
            except psutil.Error:
                return
            if rss > self.limit:
                self.tripped = rss
                _killpg(self.proc)
                return

    def stop(self) -> None:
        self._stop.set()


_image_cache: dict[str, bool] = {}


def docker_image_available(image: str) -> bool:
    """True when the docker CLI, a reachable daemon, and `image` all exist. Cached per process."""
    if image not in _image_cache:
        ok = False
        if shutil.which("docker"):
            try:
                ok = subprocess.run(["docker", "image", "inspect", image], capture_output=True,
                                    timeout=10).returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                ok = False
        _image_cache[image] = ok
    return _image_cache[image]


def resolve_backend(requested: str, image: str) -> str:
    """`REPLICATOR_SANDBOX_BACKEND` overrides the constructor; `auto` prefers the container."""
    req = os.environ.get("REPLICATOR_SANDBOX_BACKEND", requested).strip().lower()
    if req == "local":
        return "local"
    if req == "container":
        if not docker_image_available(image):
            raise RuntimeError(f"sandbox backend=container requested but docker image {image!r} is not available")
        return "container"
    if req == "auto":
        return "container" if docker_image_available(image) else "local"
    raise ValueError(f"unknown sandbox backend {req!r} (expected auto|container|local)")


class SandboxCallError(RuntimeError):
    """The function called through `Sandbox.call` raised, or the sandboxed process died."""


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    seconds: float


class Sandbox:
    def __init__(self, workdir: Path, mem_gb: float = 8.0, log_path: Path | None = None, *,
                 timeout_s: int = 300, network: bool = False, backend: Backend = "auto",
                 image: str | None = None, logger: Callable[[str], None] | None = None,
                 nproc: int = 2048, max_file_gb: float = 16.0):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.mem_bytes = int(mem_gb * 1024 ** 3)
        self.timeout_s = timeout_s
        self.network = network
        self.nproc = nproc
        self.max_file_bytes = int(max_file_gb * 1024 ** 3)
        self.image = image or os.environ.get("REPLICATOR_SANDBOX_IMAGE", DEFAULT_IMAGE)
        # The log lives OUTSIDE the work dir: generated code can write anything under workdir.
        self.log_path = log_path or (self.workdir.parent / "sandbox_log.jsonl")
        self.logger = logger
        self.blacklist_hits: list[str] = []
        self.home = Path(tempfile.mkdtemp(prefix="replicator-home-"))
        self._lock = threading.Lock()
        self.backend = resolve_backend(backend, self.image)
        self.banner = (BANNER_LOCAL if self.backend == "local" else
                       f"sandbox backend=container: image={self.image}, network={'bridge' if network else 'none'}")
        self._log({"tag": "sandbox", "event": "backend", "backend": self.backend, "banner": self.banner})
        if logger:
            logger(self.banner)

    # ------------------------------------------------------------------ plumbing
    def _log(self, rec: dict) -> None:
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        return sandbox_env(self.workdir, extra, home=self.home)

    def _limits(self, cpu_s: int) -> None:
        """preexec_fn: each limit is attempted on its own; an unsupported one is skipped, never
        the whole set. RLIMIT_AS is advisory on macOS, which is why the RSS watchdog exists."""
        cpu = int(cpu_s)
        for name, soft, hard in (("RLIMIT_AS", self.mem_bytes, self.mem_bytes),
                                 ("RLIMIT_CPU", cpu, cpu + 30),
                                 ("RLIMIT_NPROC", self.nproc, self.nproc),
                                 ("RLIMIT_FSIZE", self.max_file_bytes, self.max_file_bytes)):
            rid = getattr(resource, name, None)
            if rid is None:
                continue
            try:
                _, cur_hard = resource.getrlimit(rid)
                if cur_hard != resource.RLIM_INFINITY:
                    soft, hard = min(soft, cur_hard), min(hard, cur_hard)
                resource.setrlimit(rid, (soft, hard))
            except (ValueError, OSError):
                continue

    def _container_argv(self, argv: list[str], env: dict[str, str], name: str, cpu_s: int | None = None) -> list[str]:
        """The `docker run` command for one call. The work dir is the only writable mount; the
        adapter cache (`data_cache/`, pre-staged by the orchestrator outside the sandbox) is mounted
        read-only at the path `replicator.data.cache.CACHE_ROOT` resolves to inside the image.
        Memory is bounded by `--memory` (cgroup), not the RSS watchdog, in this mode; processes by
        `--pids-limit`; file size and CPU by the same rlimits local mode sets, passed as `--ulimit`.
        The process runs as the host uid:gid (`--user`), not the image's `sandbox` user: the work dir
        is a host directory owned by the orchestrator's user, and on a Linux host a different uid in
        the container cannot write it (the containment matrix's preflight caught exactly that)."""
        env = {**env, "PYTHONPATH": "/work:/app", "HOME": "/tmp",
               "HF_HOME": "/app/data_cache/hf", "TORCH_HOME": "/app/data_cache/torch"}
        env_file = self.home / f"env-{name}.list"
        env_file.write_text("".join(f"{k}={v}\n" for k, v in env.items() if "\n" not in v))
        inner = ["python" if a == sys.executable else a for a in argv]
        ulimits = ["--ulimit", f"fsize={self.max_file_bytes}:{self.max_file_bytes}"]
        if cpu_s is not None:
            ulimits += ["--ulimit", f"cpu={int(cpu_s)}:{int(cpu_s) + 30}"]
        return ["docker", "run", "--rm", "--name", name, "--network", "bridge" if self.network else "none",
                "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--memory", f"{max(1, self.mem_bytes // (1024 ** 2))}m", "--pids-limit", "64", *ulimits,
                "--tmpfs", "/tmp", "-v", f"{self.workdir.resolve()}:/work:rw",
                "-v", f"{DATA_CACHE}:/app/data_cache:ro", "-w", "/work",
                "--env-file", str(env_file), self.image, *inner]

    @staticmethod
    def _docker_kill(name: str) -> None:
        """Killing the `docker run` client leaves the container running; kill it by name."""
        try:
            subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass

    # ------------------------------------------------------------------ exec
    def run(self, argv: list[str], timeout_s: int | None = None, env: dict | None = None, tag: str = "") -> ExecResult:
        timeout_s = timeout_s or self.timeout_s
        cmd = " ".join(argv)
        if PROVENANCE_MONITOR.search(cmd):
            self.blacklist_hits.append(cmd)
        penv = self.env(env)
        t0 = time.time()
        container = f"sandbox-{uuid.uuid4().hex[:12]}" if self.backend == "container" else None
        cpu_s = timeout_s * max(1, os.cpu_count() or 1)
        if container:
            popen_argv, popen_env, preexec = self._container_argv(argv, penv, container, cpu_s), None, None
        else:
            popen_argv, popen_env, preexec = argv, penv, (lambda: self._limits(cpu_s))
        proc = subprocess.Popen(popen_argv, cwd=self.workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=popen_env, preexec_fn=preexec, start_new_session=True)
        watchdog = _RssWatchdog(proc, self.mem_bytes) if self.backend == "local" else None
        if watchdog:
            watchdog.start()
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            if container:
                self._docker_kill(container)
            _killpg(proc)
            try:
                out, err = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = "", ""
        finally:
            if watchdog:
                watchdog.stop()
        rc = proc.returncode
        if timed_out:
            rc, err = 124, f"timeout after {timeout_s}s"
        elif watchdog and watchdog.tripped:
            rc = 137
            err = (err or "")[-_TAIL:] + (f"\nkilled: resident memory {watchdog.tripped / 1024 ** 3:.2f} GB exceeded "
                                          f"the {self.mem_bytes / 1024 ** 3:.2f} GB sandbox limit")
        res = ExecResult(rc, (out or "")[-_TAIL:], (err or "")[-_TAIL:], time.time() - t0)
        self._log({"tag": tag, "cmd": cmd, "rc": res.returncode, "secs": round(res.seconds, 2), "backend": self.backend})
        return res

    def python(self, script: str, timeout_s: int | None = None, tag: str = "") -> ExecResult:
        return self.run([sys.executable, "-c", script], timeout_s=timeout_s, tag=tag)

    def python_file(self, rel: str, args: list[str] | None = None, timeout_s: int | None = 600, tag: str = "") -> ExecResult:
        return self.run([sys.executable, rel, *(args or [])], timeout_s=timeout_s, tag=tag)

    # ------------------------------------------------------------------ file-based RPC
    def call(self, module: str, fn: str, kwargs: dict[str, Any] | None = None, *, args: list[Any] | None = None,
             timeout_s: int | None = None, tag: str = "call") -> Any:
        """Run `module.fn(*args, **kwargs)` inside the sandbox and return its result.

        Arguments and results cross as JSON; DataFrames as parquet side files (see `_runner`).
        Raises `SandboxCallError` with the child's exception text if the call fails.
        """
        call_id = uuid.uuid4().hex[:12]
        self._ship_runner()
        call_path, result_path = _runner.call_paths(self.workdir, call_id)
        payload = {"id": call_id, "module": module, "fn": fn,
                   "args": _runner.encode(list(args or []), self.workdir, f"_arg_{call_id}"),
                   "kwargs": _runner.encode(dict(kwargs or {}), self.workdir, f"_kwarg_{call_id}")}
        call_path.write_text(json.dumps(payload, allow_nan=True))
        try:
            r = self.run([sys.executable, "_runner.py", call_path.name], timeout_s=timeout_s, tag=f"{tag}:{module}.{fn}")
            if not result_path.exists():
                raise SandboxCallError(f"{module}.{fn} produced no result (rc={r.returncode}): {r.stderr[-3000:]}")
            res = json.loads(result_path.read_text())
            if res.get("id") != call_id:
                raise SandboxCallError(f"{module}.{fn}: result id mismatch")
            if not res.get("ok"):
                raise SandboxCallError(res.get("error", "unknown error"))
            return _runner.decode(res.get("result"), self.workdir)
        finally:
            for p in self.workdir.glob(f"_*_{call_id}*"):
                try:
                    p.unlink()
                except OSError:
                    pass

    def _ship_runner(self) -> None:
        """Keep an exact copy of the trusted runner in the work dir (a model edit does not persist)."""
        src = Path(_runner.__file__).read_bytes()
        dst = self.workdir / "_runner.py"
        with self._lock:
            if not dst.exists() or dst.read_bytes() != src:
                tmp = dst.with_suffix(".py.tmp")
                tmp.write_bytes(src)
                os.replace(tmp, dst)
