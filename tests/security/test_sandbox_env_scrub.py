"""Containment test for the build sandbox: model-authored code runs under the orchestrator's
own identity, so the orchestrator's Anthropic credentials must not reach the subprocess env.
Added 2026-11-25 with the deep security pass (see SECURITY.md)."""
from __future__ import annotations

from replicator.build.sandbox import Sandbox, scrub_secrets
from replicator.templates import cs_eval


def test_scrub_secrets_drops_anthropic_credentials():
    env = {"ANTHROPIC_API_KEY": "sk-ant-secret", "ANTHROPIC_AUTH_TOKEN": "tok",
           "PATH": "/usr/bin", "HF_HOME": "/cache"}
    out = scrub_secrets(env)
    assert "ANTHROPIC_API_KEY" not in out
    assert "ANTHROPIC_AUTH_TOKEN" not in out
    assert out["PATH"] == "/usr/bin" and out["HF_HOME"] == "/cache"


def test_generated_code_cannot_read_anthropic_key(tmp_path, monkeypatch):
    """Empirically: set a sentinel key in the parent env, run generated code in the sandbox,
    and confirm it comes back empty — the key never crosses the process boundary."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SENTINEL-must-not-leak")
    sb = Sandbox(tmp_path / "work")
    r = sb.python("import os; print('KEY=' + os.environ.get('ANTHROPIC_API_KEY', '<absent>'))")
    assert r.returncode == 0, r.stderr
    assert "KEY=<absent>" in r.stdout
    assert "SENTINEL" not in r.stdout


def test_cs_reproduce_env_is_scrubbed(tmp_path, monkeypatch):
    """The CS-track reproduce.sh runner builds its own env dict; it must also be scrubbed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SENTINEL-must-not-leak")
    wd = tmp_path / "work"
    wd.mkdir()
    # reproduce.sh echoes the key (if present) into metrics so we can observe the subprocess env.
    (wd / "reproduce.sh").write_text(
        '#!/usr/bin/env bash\n'
        'printf \'{"seed":0,"split":"s","n_examples":1,"scale":1.0,'
        '"metrics":{"leaked":"%s"}}\' "${ANTHROPIC_API_KEY:-<absent>}" > "$METRICS_OUT"\n')
    out = cs_eval.run_reproduce(wd, {"_smoke": True}, 0, timeout_s=60)
    assert out["metrics"]["leaked"] == "<absent>"
