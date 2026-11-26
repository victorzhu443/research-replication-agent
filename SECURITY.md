# Security notes — research-replication-agent

This agent reads **untrusted papers** (anyone can post a PDF) and feeds their
extracted text to a model. That makes prompt injection (OWASP LLM01) the headline
risk: a paper could carry instructions that steer the replication verdict or
coerce a tool action. Secondary risks are the parsers over that same untrusted
text. Tooling and tests live under `tests/security/`; the shared toolkit is
`../security-agents/`.

## Threat model

| Asset | Untrusted channel | Risk | Status |
|---|---|---|---|
| replication verdict | paper content → `extract_spec` / `referee_spec` | a paper that addresses the model could steer extraction ("report a Match") | mitigated by data-boundary directive (below) |
| build/run actions | paper content → builder prompts | coerced tool call (LLM01 / excessive agency) | reader/actor split in place: the Builder takes the validated `Spec`, never paper text (test-guarded); generated code is contained by the sandbox below |
| generated code | model-authored `pipeline.py` / `reproduce.sh` / `run_python` | arbitrary code with the orchestrator's authority | **FIXED (RC-3)** — one execution boundary (`Sandbox`): subprocess + allow-listed env + RSS/CPU/file limits + wall clock; container backend when the image exists |
| operator URLs | `fetch_arxiv`, `fetch_html_as_text` | egress to an unintended address (SSRF class) | **FIXED (RC-4)** — `secprim.safe_fetch`: pinned, egress-validated, host allow-list for arXiv, body cap |
| credentials | parent process env → generated code | leak by omission | **FIXED (RC-5)** — env built from an allow-list, not inherited-minus-denylist |
| dependencies | PyPI | known CVEs in the lockfile | **FIXED (RC-7)** — `pypdf` removed (unused), `urllib3`/`setuptools`/transitives bumped, floors pinned, `sca` job in CI |
| parser robustness | paper text → `intake.find_companions`, `_arxiv_version` | crash / hang on hostile input | fuzzed, robust |
| deserialization | cached artifacts → `pickle.load` | code execution if a cache file is untrusted | flagged, low risk today |

**Important correction to a first-pass assumption.** The paper does **not** enter
the prompt as interpolated text. `extract_spec`/`referee_spec` pass it as a
*document block* (`LLM.paper_block(pdf)`), structurally separate from the
instruction text — the injection-resistant pattern, not a bug. And the `snippet`
in `llm.py`'s JSON-repair feedback interpolates the model's **own prior output**
plus a Python exception string, not external untrusted text. So there is no
raw-text paper sink to string-wrap here. The real residual LLM01 exposure is the
paper's *content* steering the model that reads it, and the correct mitigation is
in the app-controlled system prompt, not a string filter.

## What's been done

- **Data-boundary directive** (`replicator/spec/extract.py`): `EXTRACT_SYSTEM` and
  `REFEREE_SYSTEM` now tell the model the paper is untrusted data to extract from,
  never a source of instructions — the spotlighting defense at the instruction
  level, which preserves paper fidelity (no content mutation). A test
  (`test_paper_reading_prompts_state_the_data_boundary`) guards that both prompts
  keep it.
- **Prompt-injection harness** (`tests/security/test_prompt_injection.py`): the
  documented LLM01 corpus is replayed against a model of the paper-text sink.
  Undefended it is subverted; through `replicator.security.harden_paper_text` it
  resists 8/8. This demonstrates the measure→mitigate loop and guards any *future*
  raw-text sink.
- **Defensive wrapper** (`replicator/security.py`): `harden_paper_text` (fence +
  neutralize) and `scrub_model_output`, reusing the shared toolkit with a
  self-contained fallback. Kept for the day paper text is interpolated as a string
  (e.g. a future snippet of paper text in a repair prompt).
- **Fuzzing** (`tests/security/test_fuzz_parsers.py`): Hypothesis drives hostile
  input through `find_companions` and `_arxiv_version`; both degrade safely (no
  undeclared exceptions across 300 examples each).
- **SAST** (Semgrep, custom rules + `p/python`/`p/secrets`):
  - `papers/batch/resnet/work/assemble.py:17` — `pickle.load` (CWE-502). Low risk
    today (the file is agent-produced, not downloaded) but flagged; prefer a
    non-pickle format for anything that could originate off-machine.

## Root-cause fixes (2026-11-26)

Implemented from `../security-agents/DESIGN.md` (§3.3, §3.4, §3.5, §3.7, §3.8).
Each item is one line on *how*; the design holds the reasoning.

| RC | Status | How |
|---|---|---|
| RC-3 sandbox | **FIXED** | `replicator/build/sandbox.py` is the single execution boundary. `_load_pipeline` (in-process `exec_module`) is deleted; the finance `run`, `raw_data`, `build_features`, `pit_audit` and the smoke attribute check go through `Sandbox.call`, a file-based RPC (`_runner.py`: JSON + parquet, per-call files, nothing live crosses). `cs_eval.run_reproduce` runs `bash reproduce.sh` through `Sandbox.run` (not a second `Popen`), the Builder receives the orchestrator's Sandbox, and `run_python.path` is checked with `contained_path`; an AST test forbids `subprocess` anywhere in `replicator/` except `build/sandbox.py` (and the orchestrator's own `uv lock --check`, allow-listed by its literal argv). Limits: `RLIMIT_CPU/NPROC/FSIZE/AS` where supported, plus an RSS watchdog (psutil) that kills the process group over `mem_gb`. `RLIMIT_AS` is live on Linux (an over-limit allocation fails with `MemoryError` before the watchdog fires — CI confirmed); the RSS watchdog is the macOS fallback, where `RLIMIT_AS` is advisory. The test accepts either path and fails if the allocation succeeds. Backend `auto` runs `docker run --network none --read-only --cap-drop ALL …` when `replicator-sandbox:latest` exists (`make sandbox-image`), else `local` with the banner `sandbox backend=local: network is NOT restricted` written to `orchestrator.log` at `stage_setup` on every run (any track, prebuilt or not); `sandbox_log.jsonl` sits in the run root, outside the model-writable work dir. Container mode: each call is `--name sandbox-<id>` and a timeout runs `docker kill <name>` (killing only the client would leave the container running); memory there is bounded by `--memory`, the RSS watchdog is the local-mode equivalent. **The container path is untested end to end on this machine** (command shape and kill-by-name are unit-tested with a stand-in). `REPLICATOR_SANDBOX_BACKEND=local|container` is the rollback flag. `BLACKLIST` → `PROVENANCE_MONITOR` (alias kept). |
| RC-4 fetch | **FIXED** | Every outbound fetch in `replicator/` goes through `secprim.safe_fetch` (resolve once, connect to the validated address, hop-by-hop redirects, bounded body): `spec/intake.py` `fetch_arxiv` (`allow_hosts=["arxiv.org"]`, 64 MB) and `fetch_html_as_text` (2 MB); `data/adapters/french.py` `probe`/`_download` (`allow_hosts=["mba.tuck.dartmouth.edu"]`, 50 MB, and the zip is budgeted with `safe_zip_members` before any member is read); `data/adapters/hf_datasets.py` `probe`/`probe_model`/`fetch` (`allow_hosts=["huggingface.co"]`, 5 MB, `json.loads` on the capped body). `httpx` is no longer imported anywhere in `replicator/`. |
| RC-2 parsers | **FIXED** | `intake.html_to_text` strips tags in linear time (`<[^<>]+>` plus a single forward scan for script/style blocks; the previous `.*?</\1>` and `<[^>]+>` patterns were quadratic on hostile markup). `redos_probe` over every pattern and `size_sweep` over the whole function are in `tests/security/`; `html_to_text`, `french._parse_block` and `hf_datasets.parse_dataset_info` are registered with `assert_robust`. |
| RC-5 secrets | **FIXED** | `sandbox_env()` = `secprim.child_env` over an explicit allow-list (PATH, locale, cache dirs, thread/device knobs, the per-run values) + throwaway HOME; `scrub_secrets` and `assert_no_secrets` stay as the second check before every exec. |
| RC-7 supply chain | **FIXED** | `uv remove pypdf` (declared, never imported); `urllib3` 2.7.0→2.8.0, `setuptools` 84.0.0→84.2.1, `aiohttp`, `certifi`, `filelock`, `multidict`, `pillow` bumped; floors in `[tool.uv] constraint-dependencies`; `pip-audit --strict` on the exported lock reports no known vulnerabilities; `sca` job + Dependabot (pip, github-actions, weekly). |
| RC-8 reader/actor | **in place, test-guarded** | `Builder.__init__` takes `Spec` (pydantic), not a path or text; `tests/security/test_builder_signature.py` asserts it and that the builder prompt is built from `spec.model_dump_json`. All three paper-reading prompts (`EXTRACT_SYSTEM`, `REFEREE_SYSTEM`, `REREAD_SYSTEM`) carry the data-boundary directive, test-guarded. |

Tests: `tests/security/test_sandbox_call.py` (round-trip, DataFrame args, concurrency, allow-listed env,
banner, RSS watchdog, process-group timeout, backend resolution, container command shape),
`test_no_inprocess_exec.py` (AST walk: no `exec_module`/`spec_from_file_location`/`exec`/`eval` in
`replicator/`, no `sys.path` edits in the orchestrator), `test_builder_signature.py`, plus the
existing env-scrub, fuzz and LLM01 tests. CI: `.github/workflows/security.yml` (security-tests,
sast, sca — actions SHA-pinned; the toolkit repo is public, so the jobs run anonymously; an optional read-only `TOOLKIT_TOKEN` secret is honored if set and only needed if the repo is ever made private again).

**Accepted trade-off — `PYTHONPATH` is work dir + repo root, not work dir only.** Generated finance pipelines import the trusted data adapters (`replicator.data.adapters`), so the repo root is on the child's path in local mode. Consequence: generated code can drive adapter fetches (host-allow-listed, size-capped, cached under `data_cache/`) — it still cannot see credentials. In container mode (`--network none`) adapters cannot fetch at all: data must be pre-staged by the orchestrator (triage probes and the adapter cache under `data_cache/`, mounted read-only at `/app/data_cache`, where `CACHE_ROOT` resolves inside the image). Both modes are stated here so neither is a surprise.

## Remaining (tracked)

1. **Container backend in daily use** — the `container` path is implemented (command shape and
   kill-by-name unit-tested with a stand-in), but the image has not been built and exercised end
   to end on this machine. Until `make sandbox-image` is run, the sandbox is `local` (announced in
   every run log) and network egress from generated code is not restricted.
2. `papers/batch/resnet/work/assemble.py` (one-off, already-run, agent-produced) still uses
   `pickle.load` on the CIFAR-10 *python* batches, which are pickled dicts — `np.frombuffer` would
   need the *binary* distribution instead, i.e. a re-download, so it is left documented
   (DESIGN §3.3 component 7; `_runner.save_result/load_result` is the replacement format).
3. Apply `harden_paper_text` if paper text is ever interpolated into a prompt as a
   string (it is not today).

## Deep analysis (2026-11-25)

> Pre-fix state, kept as the record the fixes were designed against. For what holds now,
> read *Root-cause fixes (2026-11-26)* above.

A depth pass extending the baseline: full scanner suite plus a hand trace of the
untrusted-input→dangerous-sink flows, with the model-generated-code execution
surface as the headline. Each scanner finding is traced to a verdict rather than
listed.

### Scanner results

| Scanner | Scope | Result |
|---|---|---|
| **bandit** `-r replicator` | 2674 LoC | 9 LOW, **0 MEDIUM, 0 HIGH**. All LOW: `subprocess` import/use (B404/B603/B607) at the three exec sites, and one `try/except/pass` (B110) in triage env-parsing. No eval/pickle/yaml/weak-crypto in `replicator`. |
| **semgrep** custom `rules.yml` | 161 files | **1 finding** — `pickle.load` in `papers/batch/resnet/work/assemble.py:17` (same one the baseline flagged; agent-produced artifact, not downloaded). |
| **semgrep** `p/python` + `p/secrets` | tracked source (packs fetched, not skipped) | **0 findings**. No hardcoded secrets, no injection sinks in hand-written code. |
| **pip-audit** (venv) | installed deps | `pypdf 6.18.1` → **~20 CVEs**, `urllib3 2.7.0` → 3, `setuptools 84.0.0` → 1. See dependency note below. |

bandit's three subprocess sites are the real execution surface and are traced by
hand below, not dismissed as LOW.

### Dangerous-sink sweep (eval/exec/pickle/yaml/network), traced

- **`eval`/`exec`/`compile`/`yaml.load`** in hand-written `replicator/` code: **none**
  (the grep hits are all `re.compile`). `importlib.exec_module` is used — see the
  sandbox write-up; it is the finance-path code-exec, not a text `exec()`.
- **`pickle.load`** — one site, `papers/.../assemble.py` (agent-generated, local).
  Unchanged verdict: low risk today, prefer a non-pickle format for anything that
  could originate off-machine. Note it loads an agent/download artifact, so if a
  companion repo's cached tensor is ever fed here it is CWE-502 code-exec.
- **Network fetch** — `httpx.get` in `spec/intake.py` (`fetch_arxiv`,
  `fetch_html_as_text`) and the data adapters. `fetch_arxiv` builds a **constant
  `https://arxiv.org/pdf/{id}`** host from a CLI-validated arXiv id
  (`cli.py:_ARXIV`), so no SSRF. `fetch_html_as_text(url, …)` fetches an
  **operator-supplied URL** with `follow_redirects=True` and **no host/scheme
  guard** — a thin SSRF surface, but the URL is a CLI argument the operator types,
  not attacker-reachable, and the response is parsed as text, not acted on.
  **Verdict then: low** (operator-trust boundary). **Now fixed (RC-4):** both, and the data
  adapters (French library, HF Hub) that this pass had not listed, go through `secprim.safe_fetch`
  with host allow-lists and body caps.

### The headline: running model-generated code

The agent's core function is to **write and execute model-authored code**. There
are **three** distinct execution paths, and the `Sandbox` class covers only one:

1. **Finance track — in-process `importlib`, ZERO isolation.**
   `orchestrator._load_pipeline()` does `spec_.loader.exec_module(mod)` and then
   `mod.run(cfg, seed)` / `mod.raw_data()` (orchestrator.py:141–150, 166, 175).
   The model-written `pipeline.py`'s **top-level statements and `run()` execute
   inside the orchestrator's own Python process** — no subprocess, no timeout, no
   memory bound, no cwd confinement. `sys.path.insert(0, work)` is also never
   popped, so later imports resolve from the work dir. A hostile `pipeline.py`
   here has the orchestrator's full authority directly: its env, its open file
   handles, its network, and the ability to monkeypatch the rest of the run.

2. **CS track — `subprocess.Popen(["bash","reproduce.sh"])`** in
   `cs_eval.run_reproduce`. Good: own process group + wall-clock watchdog that
   `killpg`s the tree on timeout (robust against pipe-holding grandchildren).
   Missing: it passed the **full `os.environ`** through and set no rlimits and no
   network restriction. cwd is the work dir but the process is not confined to it.

3. **Builder tool loop — `Sandbox.run` / `.python_file`** for `run_python` tool
   calls. This is the only path with the `Sandbox` wrapper.

**What the `Sandbox` actually provides (and does not):**

- *cwd* = a dedicated work dir — but this is a convenience, not a jail. Absolute
  paths and `..` escape it freely; the process can read/write anywhere the user can.
- *wall-clock timeout* via `subprocess.run(timeout=)` — real, enforced.
- *memory rlimit* (`RLIMIT_AS`) — **skipped on macOS** (`preexec_fn=… if
  sys.platform != "darwin" else None`), i.e. **no memory cap on this dev machine**;
  it applies only on Linux.
- *env* — inherited the full parent environment (now scrubbed of the Anthropic
  key, see fix) ; HF/AWS/other tokens still pass through.
- *network* — **wide open**. No egress control of any kind.
- *the `BLACKLIST` regex is not a security control.* It only matches git/HF URLs in
  the command string and **appends to a log** (`blacklist_hits`); the command still
  runs. Its sole effect is to downgrade `kind_of_test` from `re_implementation` to
  `reproduction` (the author-code-isolation monitor), and it is trivially evaded
  (raw IP, DNS, a URL the generated *program* builds at runtime rather than on argv).
- `run_python` does **not** path-check its `path` arg (unlike `write_file`, which
  confines to the work dir) — but this adds no privilege, since the model already
  has `write_file` + arbitrary execution.

**Exposure class at the time of the analysis** (categorical): credential exposure
through an inherited environment with open egress; filesystem access outside the run dir; no
memory bound on macOS; and, on the finance path, execution inside the orchestrator process
itself. This was a trust-the-model posture, acceptable only for an operator running chosen papers
on their own machine.

**Resolved by the root-cause fixes above (2026-11-26):** the finance path now runs in a
subprocess through `Sandbox.call` and the CS path through `Sandbox.run` — there is one process
spawner in `replicator/`, test-enforced; every exec path gets an allow-listed environment and a
throwaway HOME; memory is bounded by an RSS watchdog on every platform, CPU/process/file size by
rlimits where supported; `run_python.path` is contained; the `container` backend removes network
egress and makes the work dir the only writable mount when the image is present. What is not yet
closed is listed under *Remaining*.

### LLM01 paper injection — still holds (not re-done)

Confirmed unchanged from the baseline: the paper enters `extract_spec` /
`referee_spec` as a **document block** (`LLM.paper_block`), not interpolated text,
and `EXTRACT_SYSTEM` / `REFEREE_SYSTEM` carry the data-boundary directive (guarded
by `test_paper_reading_prompts_state_the_data_boundary`). The builder prompts the
model from the frozen spec, not raw paper text. No new raw-paper string sink was
introduced. The real residual is paper *content* steering the model that writes the
code — the privilege-split / containment items above, not a string filter.

## Re-run

```bash
.venv/bin/python -m pytest tests/security/ -q          # or: make security-tests
.venv/bin/python -m pytest tests/ -q                   # whole suite; the smoke stage is the finance-path acceptance test
../security-agents/semgrep/scan.sh .
.venv/bin/python -m bandit -r replicator -q
uv export --frozen --no-dev --no-hashes --no-emit-project --no-emit-package security-agents > /tmp/req.txt \
  && .venv/bin/python -m pip_audit --strict --no-deps -r /tmp/req.txt
make sandbox-image                                     # enables the container backend (REPLICATOR_SANDBOX_BACKEND=auto)
```
