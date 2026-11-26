"""Model-authored code must never be imported or executed inside the orchestrator process.

An AST walk over `replicator/` asserts the in-process loader is gone for good: no
`exec_module` / `spec_from_file_location` / `load_module`, no `exec()` / `eval()`, and no
`sys.path.insert` in the orchestrator (the old loader left the work dir on the import path).
And there is exactly one place that spawns processes: `build/sandbox.py`. The only other
`subprocess` call allowed is the orchestrator's `uv lock --check` over the repo's own lockfile
(trusted argv, not generated code), allow-listed by its literal first argument."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "replicator"
FORBIDDEN_ATTRS = {"exec_module", "spec_from_file_location", "load_module"}
FORBIDDEN_NAMES = {"exec", "eval"}


def _calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


@pytest.mark.parametrize("path", sorted(PKG.rglob("*.py")), ids=lambda p: str(p.relative_to(PKG)))
def test_no_inprocess_exec(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for call in _calls(tree):
        f = call.func
        if isinstance(f, ast.Attribute):
            assert f.attr not in FORBIDDEN_ATTRS, f"{path.name}:{call.lineno} uses {f.attr}"
        if isinstance(f, ast.Name):
            assert f.id not in FORBIDDEN_NAMES, f"{path.name}:{call.lineno} uses {f.id}()"


def test_orchestrator_does_not_touch_sys_path():
    tree = ast.parse((PKG / "orchestrator.py").read_text())
    for call in _calls(tree):
        f = call.func
        if isinstance(f, ast.Attribute) and f.attr in ("insert", "append"):
            v = f.value
            assert not (isinstance(v, ast.Attribute) and v.attr == "path" and isinstance(v.value, ast.Name)
                        and v.value.id == "sys"), f"orchestrator.py:{call.lineno} modifies sys.path"
    src = (PKG / "orchestrator.py").read_text()
    assert "_load_pipeline" not in src and "importlib" not in src


SUBPROCESS_FREE_ZONE_EXCEPTIONS = {
    "build/sandbox.py": None,        # the boundary itself
    "orchestrator.py": {"uv"},       # `uv lock --check`: argv[0] must be this literal
}


def _is_subprocess_call(call: ast.Call) -> bool:
    f = call.func
    if isinstance(f, ast.Attribute):
        if isinstance(f.value, ast.Name) and f.value.id == "subprocess":
            return True
        return f.attr in {"Popen", "system", "popen", "spawn", "execv", "execvp", "execve"}
    return isinstance(f, ast.Name) and f.id in {"Popen", "check_output", "check_call", "system", "popen"}


@pytest.mark.parametrize("path", sorted(PKG.rglob("*.py")), ids=lambda p: str(p.relative_to(PKG)))
def test_only_the_sandbox_spawns_processes(path: Path):
    rel = str(path.relative_to(PKG))
    allowed_argv0 = SUBPROCESS_FREE_ZONE_EXCEPTIONS.get(rel, set())
    if allowed_argv0 is None:
        return
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
            assert "subprocess" not in names or rel in SUBPROCESS_FREE_ZONE_EXCEPTIONS, f"{rel}:{node.lineno} imports subprocess"
    for call in _calls(tree):
        if not _is_subprocess_call(call):
            continue
        argv0 = None
        if call.args and isinstance(call.args[0], ast.List) and call.args[0].elts and isinstance(call.args[0].elts[0], ast.Constant):
            argv0 = call.args[0].elts[0].value
        assert argv0 in allowed_argv0, f"{rel}:{call.lineno} spawns a process outside the Sandbox (argv[0]={argv0!r})"
