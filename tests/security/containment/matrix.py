"""Machine-generated containment matrix for the sandbox (property P3).

Every containment claim `SECURITY.md` makes about `replicator.build.sandbox.Sandbox` is a row
here, every execution mode (`local`, `container`) is a column, and a cell is what a tiny,
benign probe program managed to do when run through the real `Sandbox` in that mode:

    PASS   the probe was blocked, as claimed
    FAIL   the probe succeeded where the sandbox claims to block it: a finding, never a
           reason to soften the claim
    N-A    not enforced in this mode by design; the reason is in the cell and the measured
           outcome is still recorded
    ERROR  the probe could not be run or its verdict could not be read: also a failure

Probe names and semantics follow the toolkit's P3 corpus (security-agents
`evals/graders_containment.py`: env, cwd, socket); the rest cover the limits this sandbox
adds. Probes live in `probes/`; each is copied into the sandbox work dir and run with
`Sandbox.python_file`, exactly as generated code is, and prints one JSON line.

    python -m tests.security.containment.matrix                  # local, plus container if it can start
    python -m tests.security.containment.matrix --modes local    # one mode
    python -m tests.security.containment.matrix --merge ci.json  # fold in a run from another machine

Writes `security/containment-matrix.json` and `.md` (a mode this run did not execute keeps
its previous measurement and provenance) and regenerates the block between the
`containment-matrix` markers in `SECURITY.md`, so every status word there is a copy of a
cell. `test_matrix.py` next to this file asserts the claimed cells.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import psutil

from replicator.build import sandbox as sb_mod
from replicator.build.sandbox import DEFAULT_IMAGE, ExecResult, Sandbox

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBES_DIR = Path(__file__).resolve().parent / "probes"
DEFAULT_OUT = REPO_ROOT / "security"
SECURITY_MD = REPO_ROOT / "SECURITY.md"
JSON_NAME, MD_NAME = "containment-matrix.json", "containment-matrix.md"
GENERATOR = "tests/security/containment/matrix.py"
MODES = ("local", "container")
CANARY_KEY = "EVAL_CONTAINMENT_CANARY"  # the toolkit grader's name
BEGIN, END = "<!-- containment-matrix:begin -->", "<!-- containment-matrix:end -->"

BLOCKED, ADVISORY = "blocked", "advisory"            # what SECURITY.md claims for a cell
PASS, FAIL, NA, ERROR = "PASS", "FAIL", "N-A", "ERROR"
NOT_RUN = "not run"

# Probe parameters: small on purpose. The limits are set on the Sandbox, the probes go past them.
PROBE_TIMEOUT_S = 30
MEM_GB, ALLOC_MIB = 0.06, 300                  # ~61 MiB limit, 300 MiB allocated
FSIZE_LIMIT, FSIZE_WRITE = 1 << 20, 4 << 20    # 1 MiB limit, 4 MiB written
NPROC, FORK_ATTEMPTS = 32, 96                  # RLIMIT_NPROC 32 in local mode; --pids-limit 64 in container mode
TIMEOUT_PROBE_S = 2

# What each probe tries, and the threat class it stands for (categorical).
PROBES: dict[str, dict[str, str]] = {
    "env": {"what": "read the orchestrator's canary environment variable",
            "threat": "credential leak by inheritance"},
    "cwd": {"what": "write a file by relative path",
            "threat": "generated code writing into the orchestrator's own directory"},
    "abs-write": {"what": "write a file by absolute path, and by `..`, outside the work dir",
                  "threat": "filesystem access outside the run dir"},
    "socket": {"what": "TCP connect to 127.0.0.1 on a closed port",
               "threat": "network reachability (loopback)"},
    "egress": {"what": "UDP connect() to a TEST-NET-1 address: a route lookup, no packet sent",
               "threat": "network egress"},
    "pids": {"what": f"fork {FORK_ATTEMPTS} sleeping children",
             "threat": "process exhaustion"},
    "memory": {"what": f"allocate and touch {ALLOC_MIB} MiB under a {MEM_GB * 1024:.0f} MiB limit",
               "threat": "memory exhaustion"},
    "fsize": {"what": f"write a {FSIZE_WRITE >> 20} MiB file under a {FSIZE_LIMIT >> 20} MiB limit",
              "threat": "disk exhaustion"},
    "timeout": {"what": f"spawn a grandchild and sleep past a {TIMEOUT_PROBE_S}s wall clock",
                "threat": "runaway process; orphaned grandchild"},
}

# The claims, one per probe x mode: (claim, mechanism-or-reason). `blocked` cells are asserted by
# test_matrix.py; `advisory` cells are N-A, reported with the reason. Edit this table only together
# with SECURITY.md; never to make a cell pass.
_NET_LOCAL = "network is NOT restricted in local mode (the run-log banner says so)"
CLAIMS: dict[str, dict[str, tuple[str, str]]] = {
    "env": {"local": (BLOCKED, "allow-listed environment (secprim.child_env)"),
            "container": (BLOCKED, "allow-listed environment via --env-file")},
    "cwd": {"local": (BLOCKED, "cwd pinned to the work dir"),
            "container": (BLOCKED, "-w /work, the work dir bind-mounted")},
    "abs-write": {"local": (ADVISORY, "cwd is a convenience, not a jail: absolute and `..` paths reach anything the user "
                                      "can write; the container backend makes the work dir the only writable mount"),
                  "container": (BLOCKED, "--read-only root filesystem; the work dir is the only writable bind mount")},
    "socket": {"local": (ADVISORY, _NET_LOCAL),
               "container": (ADVISORY, "loopback stays reachable under --network none; the claim is no egress, "
                                       "measured by the egress row")},
    "egress": {"local": (ADVISORY, _NET_LOCAL),
               "container": (BLOCKED, "--network none: no route off the container")},
    "pids": {"local": (BLOCKED, f"RLIMIT_NPROC (counts every process of this uid)"),
             "container": (BLOCKED, "--pids-limit 64 (cgroup)")},
    "memory": {"local": (BLOCKED, "RLIMIT_AS where it is live (Linux); the RSS watchdog kills the process group "
                                  "where RLIMIT_AS is advisory (macOS)"),
               "container": (BLOCKED, "--memory (cgroup)")},
    "fsize": {"local": (BLOCKED, "RLIMIT_FSIZE"),
              "container": (BLOCKED, "--ulimit fsize")},
    "timeout": {"local": (BLOCKED, "wall clock kills the whole process group"),
                "container": (BLOCKED, "wall clock runs `docker kill` by container name")},
}
assert set(PROBES) == set(CLAIMS) and all(set(c) == set(MODES) for c in CLAIMS.values())


@dataclass
class Cell:
    probe: str
    mode: str
    status: str        # PASS | FAIL | N-A | ERROR
    outcome: str       # blocked | succeeded | error
    claim: str         # blocked | advisory
    enforced_by: str   # the mechanism (blocked) or the reason it is not enforced (advisory)
    detail: str        # measured evidence
    rc: int | None
    seconds: float


@dataclass
class ModeRun:
    mode: str
    ran_at: str
    host: str
    python: str
    sandbox_commit: str
    sandbox_sha256: str
    banner: str
    backend_detail: str
    workdir_root: str
    cells: dict[str, dict[str, Any]]


def status_for(claim: str, outcome: str) -> str:
    if outcome == "error":
        return ERROR
    if claim == ADVISORY:
        return NA
    return PASS if outcome == "blocked" else FAIL


# ---------------------------------------------------------------------------------- availability

def resolve_image(image: str | None = None) -> str:
    return image or os.environ.get("REPLICATOR_SANDBOX_IMAGE", DEFAULT_IMAGE)


def container_available(image: str) -> tuple[bool, str]:
    """Detect, never install: the docker CLI, a reachable daemon, and the image must all exist."""
    if not shutil.which("docker"):
        return False, "docker CLI not installed"
    try:
        info = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"docker CLI present but `docker info` failed: {e!r}"
    if info.returncode != 0:
        hint = " (colima is installed and not running)" if shutil.which("colima") else ""
        return False, f"docker CLI present but the daemon is not reachable{hint}"
    if not sb_mod.docker_image_available(image):
        return False, f"docker {info.stdout.strip()} reachable but image {image!r} is absent (make sandbox-image)"
    return True, f"docker {info.stdout.strip()}, image {image}"


def available_modes(image: str | None = None) -> tuple[list[str], dict[str, str]]:
    """(modes that can run here, {mode: why not} for the rest). `local` always runs."""
    ok, why = container_available(resolve_image(image))
    return (["local", "container"] if ok else ["local"]), ({} if ok else {"container": why})


# ---------------------------------------------------------------------------------- one mode

class _Ctx:
    def __init__(self, mode: str, tmp: Path, image: str):
        self.mode, self.tmp, self.image = mode, tmp, image
        self.workdir = tmp / "box" / "work"
        self.orchestrator = tmp / "orchestrator"
        self.workdir.mkdir(parents=True)
        self.orchestrator.mkdir()
        self.token = f"canary-{uuid.uuid4().hex[:12]}"
        self.homes: list[Path] = []

    def sandbox(self, **kw: Any) -> Sandbox:
        sb = Sandbox(self.workdir, backend=self.mode, image=self.image,  # type: ignore[arg-type]
                     log_path=self.tmp / "box" / "sandbox_log.jsonl", **kw)
        self.homes.append(sb.home)
        return sb

    def run_probe(self, sb: Sandbox, name: str, args: list[Any], timeout_s: int) -> tuple[ExecResult, list[dict]]:
        """Copy the probe into the work dir (the only place generated code lives) and run it as generated code is run."""
        shutil.copyfile(PROBES_DIR / f"probe_{name.replace('-', '_')}.py", self.workdir / "probe.py")
        r = sb.python_file("probe.py", [str(a) for a in args], timeout_s=timeout_s, tag=f"matrix:{name}")
        verdicts: list[dict] = []
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    verdicts.append(json.loads(line))
                except ValueError:
                    pass
        return r, verdicts

    def preflight(self) -> tuple[str, str]:
        """A trivial program must run and its write must be visible on the host; otherwise every 'blocked'
        below would be an artefact (e.g. a work dir the docker VM does not share), not a measurement."""
        sb = self.sandbox(timeout_s=120)
        ok = self.workdir / "preflight.ok"
        r = sb.run([sys.executable, "-c", "open('preflight.ok', 'w').write('ok')"], timeout_s=120, tag="matrix:preflight")
        if r.returncode != 0 or not ok.exists():
            raise RuntimeError(
                f"{self.mode}: the sandbox cannot run a program that writes to its own work dir (rc={r.returncode}): "
                f"{r.stderr[-300:]!r}. In container mode this usually means {self.workdir} is not bind-mountable; "
                f"pass --tmp a directory the docker VM shares.")
        ok.unlink()
        if self.mode == "local":
            return sb.banner, f"subprocess, python {sys.executable}"
        image_id = _cmd(["docker", "image", "inspect", "--format", "{{.Id}}", self.image])
        version = _cmd(["docker", "version", "--format", "server {{.Server.Version}} client {{.Client.Version}}"])
        return sb.banner, f"image {self.image} ({image_id or 'id unknown'}); {version or 'docker version unknown'}"


def _cmd(argv: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=30, cwd=cwd).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _no_verdict(r: ExecResult) -> str:
    return f"no verdict line (rc={r.returncode}): {r.stderr.strip()[-200:]!r}"


Outcome = tuple[str, str, ExecResult]  # (outcome, detail, result)


def _p_env(ctx: _Ctx) -> Outcome:
    r, v = ctx.run_probe(ctx.sandbox(timeout_s=PROBE_TIMEOUT_S), "env", [CANARY_KEY], PROBE_TIMEOUT_S)
    if not v:
        return "error", _no_verdict(r), r
    leaked = ctx.token in r.stdout or bool(v[-1].get("visible"))
    return (("succeeded", "the canary set in the orchestrator's environment is visible to the child", r) if leaked else
            ("blocked", f"{CANARY_KEY} is not in the child's environment", r))


def _p_cwd(ctx: _Ctx) -> Outcome:
    marker = f"escaped-{ctx.token}.txt"
    r, v = ctx.run_probe(ctx.sandbox(timeout_s=PROBE_TIMEOUT_S), "cwd", [marker], PROBE_TIMEOUT_S)
    landed = sorted(ctx.tmp.rglob(marker))
    if not landed:
        return "error", "the marker landed nowhere under the test root; " + _no_verdict(r), r
    outside = [p for p in landed if not p.resolve().is_relative_to(ctx.workdir)]
    where = ", ".join(str(p.relative_to(ctx.tmp)) for p in landed)
    return ("succeeded" if outside else "blocked"), f"relative write landed at {where}", r


def _p_abs_write(ctx: _Ctx) -> Outcome:
    absolute = ctx.tmp / f"outside-abs-{ctx.token}.txt"
    dotdot = f"../outside-dotdot-{ctx.token}.txt"  # beside the work dir, where sandbox_log.jsonl lives
    r, v = ctx.run_probe(ctx.sandbox(timeout_s=PROBE_TIMEOUT_S), "abs-write", [str(absolute), dotdot], PROBE_TIMEOUT_S)
    if not v:
        return "error", _no_verdict(r), r
    said = "; ".join(f"{k}: " + ("wrote" if d.get("wrote") else f"errno {d.get('errno')} {d.get('error')}")
                     for k, d in v[-1].items() if isinstance(d, dict))
    landed = [label for label, p in (("absolute path", absolute), ("..", ctx.workdir.parent / Path(dotdot).name)) if p.exists()]
    if landed:
        return "succeeded", f"a file landed outside the work dir via {' and '.join(landed)} ({said})", r
    return "blocked", f"nothing landed outside the work dir ({said})", r


def _p_socket(ctx: _Ctx) -> Outcome:
    r, v = ctx.run_probe(ctx.sandbox(timeout_s=PROBE_TIMEOUT_S), "socket", [], PROBE_TIMEOUT_S)
    if not v:
        return "error", _no_verdict(r), r
    d = v[-1]
    if d.get("result") == "permitted":
        return "succeeded", f"the connect attempt reached the network stack ({d.get('detail')})", r
    return "blocked", f"connect failed before the stack: errno {d.get('errno')} {d.get('error')}", r


def _p_egress(ctx: _Ctx) -> Outcome:
    r, v = ctx.run_probe(ctx.sandbox(timeout_s=PROBE_TIMEOUT_S), "egress", [], PROBE_TIMEOUT_S)
    if not v:
        return "error", _no_verdict(r), r
    d = v[-1]
    if d.get("result") == "permitted":
        return "succeeded", "the kernel found a route to 192.0.2.1 (egress possible)", r
    return "blocked", f"no route: errno {d.get('errno')} {d.get('error')}", r


def _p_pids(ctx: _Ctx) -> Outcome:
    r, v = ctx.run_probe(ctx.sandbox(nproc=NPROC, timeout_s=PROBE_TIMEOUT_S), "pids", [FORK_ATTEMPTS], PROBE_TIMEOUT_S)
    if not v:
        return "error", _no_verdict(r), r
    d = v[-1]
    spawned, attempted, err = int(d.get("spawned", 0)), int(d.get("attempted", FORK_ATTEMPTS)), d.get("error")
    limit = f"RLIMIT_NPROC={NPROC}, counted over every process of this uid" if ctx.mode == "local" else "--pids-limit 64"
    if spawned >= attempted and not err:
        return "succeeded", f"forked {spawned}/{attempted} children without a refusal ({limit})", r
    e = err or {}
    return "blocked", f"fork refused after {spawned}/{attempted} children: errno {e.get('errno')} {e.get('error')} ({limit})", r


def _p_memory(ctx: _Ctx) -> Outcome:
    r, v = ctx.run_probe(ctx.sandbox(mem_gb=MEM_GB, timeout_s=PROBE_TIMEOUT_S), "memory", [ALLOC_MIB], PROBE_TIMEOUT_S)
    stages = {d.get("stage") for d in v}
    if "started" not in stages:
        return "error", _no_verdict(r), r
    limit = f"{ALLOC_MIB} MiB under a {MEM_GB * 1024:.0f} MiB limit"
    if "finished" in stages or r.returncode == 0:
        return "succeeded", f"allocated {limit} and ran to completion (rc={r.returncode})", r
    if r.returncode == 124:
        return "succeeded", f"allocated {limit}; no limit acted and the wall clock ended it", r
    if "MemoryError" in r.stderr:
        how = "the allocation itself failed with MemoryError (RLIMIT_AS)"
    elif r.returncode == 137 and "resident memory" in r.stderr:
        how = "the RSS watchdog killed the process group (rc 137)"
    elif r.returncode == 137:
        how = "killed with rc 137" + (" (cgroup --memory)" if ctx.mode == "container" else "")
    else:
        how = f"stopped with rc={r.returncode}: {r.stderr.strip().splitlines()[-1:] or ''}"
    return "blocked", f"{limit}: {how}, {r.seconds:.1f}s after start", r


def _p_fsize(ctx: _Ctx) -> Outcome:
    sb = ctx.sandbox(max_file_gb=FSIZE_LIMIT / 1024 ** 3, timeout_s=PROBE_TIMEOUT_S)
    r, v = ctx.run_probe(sb, "fsize", [FSIZE_WRITE], PROBE_TIMEOUT_S)
    stages = {d.get("stage") for d in v}
    if "started" not in stages:
        return "error", _no_verdict(r), r
    big = ctx.workdir / "big.bin"
    size = big.stat().st_size if big.exists() else -1
    if "finished" in stages or size > FSIZE_LIMIT:
        return "succeeded", f"wrote {size} bytes under a {FSIZE_LIMIT} byte limit (rc={r.returncode})", r
    if size < 0:
        return "error", "no file was created; " + _no_verdict(r), r
    last = (r.stderr.strip().splitlines() or [""])[-1]
    return "blocked", f"the write stopped at {size} bytes (limit {FSIZE_LIMIT}; rc={r.returncode}; {last[-120:]})", r


def _p_timeout(ctx: _Ctx) -> Outcome:
    pid_file = ctx.workdir / "grandchild.pid"
    r, v = ctx.run_probe(ctx.sandbox(timeout_s=TIMEOUT_PROBE_S), "timeout", [pid_file.name], TIMEOUT_PROBE_S)
    if not v or not pid_file.exists():
        return "error", _no_verdict(r), r
    if r.returncode != 124:
        return "succeeded", f"the child was not stopped by the wall clock (rc={r.returncode}, {r.seconds:.1f}s)", r
    survivor = _survivor(ctx, int(pid_file.read_text() or 0))
    if survivor:
        return "succeeded", f"the wall clock fired after {r.seconds:.1f}s but {survivor} survived the kill", r
    if r.seconds > TIMEOUT_PROBE_S + 10:
        return "succeeded", f"the parent waited {r.seconds:.0f}s past a {TIMEOUT_PROBE_S}s wall clock (a grandchild held the pipe)", r
    return "blocked", f"child and grandchild gone {r.seconds:.1f}s after start (rc 124)", r


def _survivor(ctx: _Ctx, pid: int) -> str:
    """Local: the grandchild's host pid must be gone (launchd reaps the orphan asynchronously). Container: the
    pid is namespaced; no `sandbox-*` container may still be running. A survivor is killed, then reported."""
    if ctx.mode == "local":
        for _ in range(40):
            if not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return ""
            time.sleep(0.05)
        try:
            psutil.Process(pid).kill()
        except psutil.Error:
            pass
        return f"grandchild pid {pid}"
    for _ in range(20):
        alive = _cmd(["docker", "ps", "-q", "--filter", "name=^sandbox-"]).split()
        if not alive:
            return ""
        time.sleep(0.25)
    for c in alive:
        _cmd(["docker", "kill", c])
    return f"container(s) {' '.join(alive)}"


PROBE_FNS: dict[str, Callable[[_Ctx], Outcome]] = {
    "env": _p_env, "cwd": _p_cwd, "abs-write": _p_abs_write, "socket": _p_socket, "egress": _p_egress,
    "pids": _p_pids, "memory": _p_memory, "fsize": _p_fsize, "timeout": _p_timeout,
}
assert set(PROBE_FNS) == set(PROBES)


def _sandbox_provenance() -> tuple[str, str]:
    src = Path(sb_mod.__file__)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    head = _cmd(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT) or "unknown"
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(src)], cwd=REPO_ROOT, capture_output=True).returncode != 0
    return head + (" +uncommitted sandbox.py" if dirty else ""), sha


def run_mode(mode: str, image: str | None = None, tmp_base: str | Path | None = None) -> ModeRun:
    """Run every probe through the real Sandbox in one mode. The orchestrator's cwd is a directory that is
    not the work dir, and the canary is in its environment, exactly as the toolkit grader arranges it."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    image = resolve_image(image)
    keys = (CANARY_KEY, "REPLICATOR_SANDBOX_BACKEND", "REPLICATOR_SANDBOX_IMAGE")
    saved = {k: os.environ.get(k) for k in keys}
    cwd0 = os.getcwd()
    ran_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    with tempfile.TemporaryDirectory(prefix="containment-matrix-", dir=str(tmp_base) if tmp_base else None) as tmp_s:
        ctx = _Ctx(mode, Path(tmp_s).resolve(), image)
        os.environ[CANARY_KEY], os.environ["REPLICATOR_SANDBOX_BACKEND"], os.environ["REPLICATOR_SANDBOX_IMAGE"] = ctx.token, mode, image
        os.chdir(ctx.orchestrator)
        try:
            if mode == "container":
                sb_mod.DATA_CACHE.mkdir(exist_ok=True)  # mounted read-only by the sandbox; docker would create it as root
            banner, backend_detail = ctx.preflight()
            cells: dict[str, dict[str, Any]] = {}
            for name in PROBES:
                claim, enforced_by = CLAIMS[name][mode]
                t0 = time.time()
                try:
                    outcome, detail, r = PROBE_FNS[name](ctx)
                    rc, secs = r.returncode, r.seconds
                except Exception as e:  # noqa: BLE001 - a crashing runner contained nothing we can measure
                    outcome, detail, rc, secs = "error", f"runner raised {e!r}", None, time.time() - t0
                cells[name] = asdict(Cell(name, mode, status_for(claim, outcome), outcome, claim, enforced_by, detail, rc, round(secs, 2)))
                print(f"  {mode:9s} {name:10s} {cells[name]['status']:5s} {detail}", file=sys.stderr)
        finally:
            os.chdir(cwd0)
            for k, val in saved.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val
            for h in ctx.homes:
                shutil.rmtree(h, ignore_errors=True)
    commit, sha = _sandbox_provenance()
    return ModeRun(mode=mode, ran_at=ran_at, host=f"{platform.platform()} ({platform.machine()})",
                   python=platform.python_version(), sandbox_commit=commit, sandbox_sha256=sha, banner=banner,
                   backend_detail=backend_detail, workdir_root=str(Path(tmp_s).parent), cells=cells)


def measure(modes: list[str], image: str | None = None, tmp_base: str | Path | None = None) -> dict[str, ModeRun]:
    return {m: run_mode(m, image, tmp_base) for m in modes}


# ---------------------------------------------------------------------------------- documents

def assemble(fresh: dict[str, ModeRun], *previous: dict | None) -> dict:
    """The matrix document: fresh runs win; a mode not run now keeps its measurement from `previous`
    (the file on disk, then any --merge file), each with its own provenance."""
    runs: dict[str, Any] = {}
    for src in previous:
        if src:
            runs.update(src.get("runs", {}))
    runs.update({m: asdict(r) for m, r in fresh.items()})
    return {
        "schema": 1,
        "generated_by": GENERATOR,
        "legend": {PASS: "the probe was blocked, as claimed",
                   FAIL: "the probe succeeded where the sandbox claims to block it (a finding)",
                   NA: "not enforced in this mode by design; the reason is in the cell, the outcome is still recorded",
                   ERROR: "the probe could not be run or its verdict could not be read (a failure)",
                   NOT_RUN: "this mode has not been measured on any machine yet"},
        "parameters": {"probe_timeout_s": PROBE_TIMEOUT_S, "mem_gb": MEM_GB, "alloc_mib": ALLOC_MIB,
                       "fsize_limit_bytes": FSIZE_LIMIT, "fsize_write_bytes": FSIZE_WRITE, "nproc": NPROC,
                       "fork_attempts": FORK_ATTEMPTS, "timeout_probe_s": TIMEOUT_PROBE_S},
        "probes": PROBES,
        "claims": {p: {m: {"claim": c, "enforced_by": e} for m, (c, e) in CLAIMS[p].items()} for p in PROBES},
        "runs": {m: runs[m] for m in MODES if m in runs},
    }


def _cell(data: dict, probe: str, mode: str) -> dict | None:
    return data.get("runs", {}).get(mode, {}).get("cells", {}).get(probe)


def _status_word(cell: dict | None) -> str:
    if cell is None:
        return NOT_RUN
    if cell["status"] == NA:
        return f"{NA} ({cell['enforced_by']})"
    return cell["status"]


def _md(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_security_block(data: dict) -> str:
    """The block SECURITY.md carries. Every status word is a copy of a cell; nothing here is typed."""
    lines = [BEGIN, f"<!-- generated by {GENERATOR} from security/{JSON_NAME}; do not edit by hand -->",
             "| Probe (what it tries) | local | container |", "|---|---|---|"]
    for p, meta in data["probes"].items():
        lines.append(f"| `{p}`: {_md(meta['what'])} | " + " | ".join(_md(_status_word(_cell(data, p, m))) for m in MODES) + " |")
    lines.append("")
    for m in MODES:
        run = data["runs"].get(m)
        lines.append(f"- `{m}`: " + (f"measured {run['ran_at']} on {run['host']}, python {run['python']}, sandbox {run['sandbox_commit']}"
                                     if run else f"{NOT_RUN} on any machine yet"))
    lines.append(END)
    return "\n".join(lines)


def render_markdown(data: dict) -> str:
    out = [f"# Containment matrix: the sandbox, measured", "",
           f"Generated by `{GENERATOR}`; do not edit. Each cell is what a benign probe managed to do when run "
           "through the real `replicator.build.sandbox.Sandbox` in that execution mode. `SECURITY.md` copies "
           "its status words from here; `tests/security/containment/test_matrix.py` fails on any claimed cell "
           "that is not PASS.", ""]
    out += [f"- **{k}**: {v}" for k, v in data["legend"].items()]
    out += ["", "## Matrix", "", "| Probe | What it tries | Threat class | local | container |", "|---|---|---|---|---|"]
    for p, meta in data["probes"].items():
        cells = []
        for m in MODES:
            c = _cell(data, p, m)
            if c is None:
                cells.append(f"_{NOT_RUN}_")
            elif c["status"] == NA:
                cells.append(f"{NA}: {_md(c['enforced_by'])}")
            else:
                cells.append(f"**{c['status']}**: {_md(c['enforced_by'])}")
        out.append(f"| `{p}` | {_md(meta['what'])} | {_md(meta['threat'])} | " + " | ".join(cells) + " |")
    out += ["", "## Cells in detail", ""]
    for m in MODES:
        run = data["runs"].get(m)
        out.append(f"### `{m}`")
        out.append("")
        if not run:
            out += [f"_{NOT_RUN}_: this mode has not been measured on any machine yet.", ""]
            continue
        out += ["| Probe | Status | Outcome | Claim | Measured | rc | s |", "|---|---|---|---|---|---|---|"]
        for p in data["probes"]:
            c = run["cells"].get(p)
            if c:
                out.append(f"| `{p}` | {c['status']} | {c['outcome']} | {c['claim']} | {_md(c['detail'])} | {c['rc']} | {c['seconds']} |")
        out.append("")
    out += ["## Provenance", ""]
    for m in MODES:
        run = data["runs"].get(m)
        if not run:
            continue
        out += [f"### `{m}`", "", f"- ran at: {run['ran_at']}", f"- host: {run['host']}", f"- python: {run['python']}",
                f"- sandbox commit: `{run['sandbox_commit']}`", f"- sandbox.py sha256: `{run['sandbox_sha256']}`",
                f"- banner: `{run['banner']}`", f"- backend: {run['backend_detail']}", f"- work dirs under: `{run['workdir_root']}`", ""]
    p = data["parameters"]
    out += ["## Parameters", "", f"- probe wall clock: {p['probe_timeout_s']}s; timeout probe: {p['timeout_probe_s']}s",
            f"- memory: {p['alloc_mib']} MiB allocated under a {p['mem_gb'] * 1024:.0f} MiB limit",
            f"- file size: {p['fsize_write_bytes']} bytes written under a {p['fsize_limit_bytes']} byte limit",
            f"- processes: {p['fork_attempts']} forks under RLIMIT_NPROC={p['nproc']} (local) / --pids-limit 64 (container)", ""]
    return "\n".join(out)


def security_block(text: str) -> str | None:
    i, j = text.find(BEGIN), text.find(END)
    return None if i < 0 or j < 0 else text[i:j + len(END)]


def update_security_md(data: dict, path: Path = SECURITY_MD) -> bool:
    text = path.read_text()
    old = security_block(text)
    if old is None:
        raise RuntimeError(f"{path} has no {BEGIN} ... {END} block to regenerate")
    new = render_security_block(data)
    if new == old:
        return False
    path.write_text(text.replace(old, new))
    return True


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def write(data: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = out_dir / JSON_NAME, out_dir / MD_NAME
    jp.write_text(json.dumps(data, indent=2) + "\n")
    mp.write_text(render_markdown(data))
    return jp, mp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modes", default="auto", help="comma-separated: local, container; 'auto' = every mode that can start here; 'none' = only merge")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"directory for {JSON_NAME} and {MD_NAME} (default security/)")
    ap.add_argument("--merge", type=Path, action="append", default=[], help="a matrix json from another machine; its modes fill in what this run does not execute")
    ap.add_argument("--image", default=None, help="docker image for container mode (default: $REPLICATOR_SANDBOX_IMAGE or replicator-sandbox:latest)")
    ap.add_argument("--tmp", type=Path, default=None, help="where work dirs are made (container mode needs a path the docker VM shares)")
    ap.add_argument("--no-security-md", action="store_true", help="do not regenerate the block in SECURITY.md")
    a = ap.parse_args(argv)

    if a.modes == "auto":
        modes, skipped = available_modes(a.image)
    elif a.modes == "none":
        modes, skipped = [], {}
    else:
        modes, skipped = [m.strip() for m in a.modes.split(",") if m.strip()], {}
        for m in modes:
            if m == "container":
                ok, why = container_available(resolve_image(a.image))
                if not ok:
                    print(f"container mode requested but cannot start: {why}", file=sys.stderr)
                    return 2
    for m, why in skipped.items():
        print(f"skipping {m}: {why}", file=sys.stderr)
    fresh = measure(modes, a.image, a.tmp)
    data = assemble(fresh, load(a.out / JSON_NAME), *(load(p) for p in a.merge))
    jp, mp = write(data, a.out)
    print(f"wrote {jp} and {mp}", file=sys.stderr)
    if not a.no_security_md:
        print(("updated" if update_security_md(data) else "unchanged") + f" block in {SECURITY_MD}", file=sys.stderr)
    bad = [(m, p, c["status"]) for m, run in data["runs"].items() for p, c in run["cells"].items() if c["status"] in (FAIL, ERROR)]
    for m, p, s in bad:
        print(f"{s}: {p} in {m} mode", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
