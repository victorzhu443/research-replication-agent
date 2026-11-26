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

## Re-run

```bash
.venv/bin/python -m pytest tests/security/ -q
../security-agents/semgrep/scan.sh .
```
