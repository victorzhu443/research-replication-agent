"""Model-authored code must never be imported or executed inside the orchestrator process.

An AST walk over `replicator/` asserts the in-process loader is gone for good: no
`exec_module` / `spec_from_file_location` / `load_module`, no `exec()` / `eval()`, and no
`sys.path.insert` in the orchestrator (the old loader left the work dir on the import path)."""
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
