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
| build/run actions | paper content → builder prompts | coerced tool call (LLM01 / excessive agency) | residual — privilege split recommended |
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

## Remaining (tracked)

1. **Privilege split** — the model that reads paper content should not be the one
   that can invoke build/run tools without a gate. This is the durable fix for the
   excessive-agency row; the directive reduces but does not eliminate it.
2. Apply `harden_paper_text` if paper text is ever interpolated into a prompt as a
   string (it is not today).

## Deep analysis (2026-11-25)

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
  **Verdict: low** (operator-trust boundary). If paper URLs are ever taken from an
  untrusted queue, add `firstplay`'s `assert_public_url`-style guard here.

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

**What a malicious generated program could still do today** (any of the three paths):

- **Exfiltrate secrets.** Network egress is open and the process inherits the
  environment. Reading a token from `os.environ` and POSTing it out is unimpeded.
  *(The single most valuable secret, the Anthropic key, is now removed — see fix.)*
- **Read/write outside the run dir** — cwd is cosmetic; `open("/Users/...")` works.
- **Burn unbounded memory on macOS** (no `RLIMIT_AS`), i.e. a local DoS.
- **Finance only:** execute in the orchestrator process itself — strictly worse
  than the subprocess paths; it can corrupt the run's own verdict from inside.

**Verdict: this is arbitrary code execution by design, and containment today is
weak — effectively a trust-the-model posture.** It is acceptable *only* because
the operator runs it on their own machine against papers they chose, and because
the LLM01 directive (below) reduces paper-driven coercion of what gets written.
It is **not** safe against a paper engineered to make the model write an exfiltrating
or destructive `pipeline.py`/`reproduce.sh`, which is exactly the excessive-agency
residual the baseline's threat-model table already names.

**Hardening applied (clearly-safe, tested):**

- **Env scrubbing of the orchestrator's LLM credentials.**
  `sandbox.scrub_secrets()` drops `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN`
  from every subprocess environment — wired into `Sandbox.run` (the merged env,
  so a caller-passed `env` can't reintroduce it) and `cs_eval.run_reproduce`.
  Generated code provably never needs the Anthropic key, so this is free and
  removes the highest-value exfil target. Tests
  (`tests/security/test_sandbox_env_scrub.py`): a unit check of the denylist, an
  **empirical** check that a sentinel key set in the parent is `<absent>` inside a
  `Sandbox.python` run, and the same for a `reproduce.sh` subprocess. All 31 repo
  tests still pass.

**Hardening recommended (documented, not applied — needs design/containers/CI):**

1. **Give the finance path a subprocess boundary** like the CS path, instead of
   `exec_module` in-process. This is the biggest single reduction in blast radius
   and should come first. (Not applied: changes the finance execution model and
   needs its own test pass across the finance families.)
2. **Network egress off by default** for generated code — run under a deny-all
   egress namespace / proxy, allow-listing only the data-adapter hosts
   (arxiv, HF, Ken French). Removes the exfil channel the open network gives today.
3. **Full env minimization** — pass generated code an explicit allow-list
   (`PATH`, `HF_HOME`, `SEED`, `REPLICATOR_CONFIG`, …) rather than inheriting
   `os.environ` and subtracting two keys. (Scrubbing the Anthropic key is the
   high-value slice done now; the general case risks dropping a var a pipeline
   needs, so it wants a test matrix.)
4. **Containment**: a temp/overlay filesystem or a real sandbox (Docker/gVisor —
   the module docstring already anticipates swapping these in behind the same
   interface) so writes can't escape the run dir; and apply `RLIMIT_AS` on macOS
   via an alternative (ulimit in the bash wrapper) since `preexec_fn` rlimit is
   Linux-only here.
5. **Dependency bump — `pypdf`.** Installed `pypdf 6.18.1` carries ~20 CVEs and
   `pypdf` is a *PDF parser* in a tool whose input is untrusted PDFs. It is a
   declared dep (`pyproject.toml`) but is **not imported** in `replicator` (intake
   parses with `pymupdf`), so exposure is latent, not live. Recommended: raise the
   floor to a patched `pypdf` (>= 6.21.4) and `uv lock`; also `urllib3>=2.8.0`.
   Not applied here: re-locking wants a fresh resolve + test run.

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
.venv/bin/python -m pytest tests/security/ -q
../security-agents/semgrep/scan.sh .
.venv/bin/python -m bandit -r replicator -q
.venv/bin/python -m pip_audit
```
