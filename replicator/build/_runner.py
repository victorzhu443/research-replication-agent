"""In-sandbox runner: the only thing that imports model-authored code.

The orchestrator never imports `pipeline.py` into its own process. It writes
`_call.json` = {id, module, fn, args, kwargs}, runs this file under the Sandbox
(`python _runner.py`), and reads `_result.json` back. Only JSON and parquet cross
the boundary — never a live module object.

Encoding (shared by both sides, so this file is imported by the sandbox *and*
copied into the work dir): JSON for dicts/lists/scalars; `pandas.DataFrame` and
`Series` are written to `<prefix>_<n>.parquet` next to the JSON and referenced as
{"__parquet__": name}. numpy scalars become Python scalars, arrays become lists,
timestamps become ISO strings.

Generated code may `from _runner import save_result, load_result` to persist its
own intermediates in the same safe format instead of pickle.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

CALL_FILE = "_call.json"
RESULT_FILE = "_result.json"


# ---------------------------------------------------------------- encoding
def _encode(obj: Any, workdir: Path, prefix: str, counter: list[int]) -> Any:
    """JSON-ready copy of `obj`; DataFrames/Series go to parquet side files."""
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - pandas is always present in this project
        pd = None  # type: ignore
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    if pd is not None and isinstance(obj, (pd.DataFrame, pd.Series)):
        counter[0] += 1
        name = f"{prefix}_{counter[0]}.parquet"
        frame = obj.to_frame() if isinstance(obj, pd.Series) else obj
        frame.to_parquet(workdir / name)
        return {"__parquet__": name, "series": isinstance(obj, pd.Series)}
    if pd is not None and isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if np is not None and isinstance(obj, np.ndarray):
        return [_encode(x, workdir, prefix, counter) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _encode(v, workdir, prefix, counter) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_encode(v, workdir, prefix, counter) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def _decode(obj: Any, workdir: Path) -> Any:
    if isinstance(obj, dict):
        if "__parquet__" in obj and isinstance(obj["__parquet__"], str):
            import pandas as pd
            name = Path(obj["__parquet__"]).name  # a bare file name; never a path
            frame = pd.read_parquet(workdir / name)
            return frame.iloc[:, 0] if obj.get("series") else frame
        return {k: _decode(v, workdir) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v, workdir) for v in obj]
    return obj


def encode(obj: Any, workdir: Path, prefix: str) -> Any:
    return _encode(obj, Path(workdir), prefix, [0])


def decode(obj: Any, workdir: Path) -> Any:
    return _decode(obj, Path(workdir))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------- helpers for generated code
def save_result(obj: Any, path: str | os.PathLike = "result.json") -> Path:
    """Persist `obj` (dicts/lists/scalars/DataFrames) without pickle."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(p, json.dumps(encode(obj, p.parent, p.stem), allow_nan=True))
    return p


def load_result(path: str | os.PathLike = "result.json") -> Any:
    p = Path(path)
    return decode(json.loads(p.read_text()), p.parent)


def has_fns(module: str, fns: list[str]) -> dict[str, bool]:
    """Which of `fns` the module exposes as callables. Runs inside the sandbox."""
    mod = importlib.import_module(module)
    return {fn: callable(getattr(mod, fn, None)) for fn in fns}


# ---------------------------------------------------------------- entry point
def call_paths(workdir: Path, call_id: str) -> tuple[Path, Path]:
    """Per-call file names, so concurrent calls (the verify grid runs flips in a thread pool)
    never read each other's request or result."""
    return workdir / f"_call_{call_id}.json", workdir / f"_result_{call_id}.json"


def main(argv: list[str]) -> int:
    workdir = Path.cwd()
    call_file = workdir / Path(argv[1]).name if len(argv) > 1 else workdir / CALL_FILE
    call = json.loads(call_file.read_text())
    result_file = call_paths(workdir, call["id"])[1] if call.get("id") else workdir / RESULT_FILE
    out: dict[str, Any] = {"id": call.get("id")}
    try:
        mod = importlib.import_module(call["module"])
        fn = getattr(mod, call["fn"])
        args = decode(call.get("args") or [], workdir)
        kwargs = decode(call.get("kwargs") or {}, workdir)
        result = fn(*args, **kwargs)
        out.update(ok=True, result=encode(result, workdir, result_file.stem))
    except BaseException as e:  # noqa: BLE001 - every failure is reported, never swallowed
        out.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-6000:])
    _atomic_write(result_file, json.dumps(out, allow_nan=True))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
