"""The containment matrix is the test: for every execution mode that can run here, each cell
`SECURITY.md` claims as blocked must measure PASS. A FAIL is a finding about the sandbox, not
about the claim. Advisory cells (N-A) are reported in the matrix, not asserted. The committed
matrix (`security/containment-matrix.json`) and the block in `SECURITY.md` must be copies of a
measurement, so staleness is a failure too.

`CONTAINMENT_MATRIX_FRESH=<path.json>` makes the tests read a matrix this same CI run just
generated instead of measuring again.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from . import matrix

COMMITTED = matrix.DEFAULT_OUT / matrix.JSON_NAME
MODES_HERE, SKIPPED_HERE = matrix.available_modes()
REGENERATE = "regenerate with `python -m tests.security.containment.matrix` and commit security/ and SECURITY.md"


@pytest.fixture(scope="module")
def fresh() -> dict[str, dict]:
    """{mode: run} measured now (or by this CI run), for the modes available on this machine."""
    given = os.environ.get("CONTAINMENT_MATRIX_FRESH")
    if given:
        runs = json.loads(Path(given).read_text())["runs"]
        missing = [m for m in MODES_HERE if m not in runs]
        assert not missing, f"{given} has no run for {missing}, which can run here"
        return {m: runs[m] for m in MODES_HERE}
    return {m: matrix.asdict(run) for m, run in matrix.measure(MODES_HERE).items()}


@pytest.fixture(scope="module")
def committed() -> dict:
    assert COMMITTED.exists(), f"{COMMITTED} is missing; {REGENERATE}"
    return json.loads(COMMITTED.read_text())


# ---------------------------------------------------------------- the claims, measured

@pytest.mark.parametrize("mode", MODES_HERE)
@pytest.mark.parametrize("probe", list(matrix.PROBES))
def test_claimed_cell_is_blocked(fresh, record_property, probe, mode):
    cell = fresh[mode]["cells"][probe]
    claim, enforced_by = matrix.CLAIMS[probe][mode]
    record_property("status", cell["status"])
    record_property("measured", cell["detail"])
    assert cell["status"] != matrix.ERROR, f"{probe}/{mode} could not be measured: {cell['detail']}"
    if claim == matrix.BLOCKED:
        assert cell["status"] == matrix.PASS, (
            f"{probe} in {mode} mode: SECURITY.md claims '{enforced_by}' but the probe {cell['outcome']}: "
            f"{cell['detail']} (rc={cell['rc']}). Fix the sandbox; do not soften the claim.")
    else:
        assert cell["status"] == matrix.NA  # reported with its reason in the matrix, never asserted


def test_modes_not_measurable_here_are_named():
    """What did not run is stated with a reason, never silently absent (the brief: detect, never install)."""
    for mode in matrix.MODES:
        assert mode in MODES_HERE or SKIPPED_HERE.get(mode), mode


# ---------------------------------------------------------------- the committed copies

def test_committed_matrix_agrees_with_this_measurement(fresh, committed):
    """The artefact in the repo is a copy of a measurement: for every mode measured here, each committed
    status must equal the one just measured. A mode not yet in the committed file is reported, not failed
    (it is merged in after the first machine that can run it does)."""
    stale = []
    for mode, run in fresh.items():
        old = committed["runs"].get(mode)
        if old is None:
            continue
        for probe in matrix.PROBES:
            want, have = run["cells"][probe]["status"], old["cells"].get(probe, {}).get("status")
            if want != have:
                stale.append(f"{probe}/{mode}: committed {have!r}, measured {want!r}")
    assert not stale, "; ".join(stale) + f"; {REGENERATE}"
    assert set(committed["probes"]) == set(matrix.PROBES) and committed["claims"] == json.loads(json.dumps(
        {p: {m: {"claim": c, "enforced_by": e} for m, (c, e) in matrix.CLAIMS[p].items()} for p in matrix.PROBES})), REGENERATE


def test_committed_matrix_has_no_failing_cell(committed):
    bad = [(m, p, c["status"]) for m, run in committed["runs"].items() for p, c in run["cells"].items()
           if c["status"] in (matrix.FAIL, matrix.ERROR)]
    assert not bad, bad


def test_security_md_block_is_a_copy_of_the_committed_matrix(committed):
    text = matrix.SECURITY_MD.read_text()
    block = matrix.security_block(text)
    assert block is not None, f"SECURITY.md lost its {matrix.BEGIN} block"
    assert block == matrix.render_security_block(committed), f"the block in SECURITY.md is not the committed matrix; {REGENERATE}"


def test_security_md_types_no_status_word_outside_the_block():
    """Every PASS / FAIL / N-A in the containment section is inside the generated block."""
    text = matrix.SECURITY_MD.read_text()
    m = re.search(r"^## Containment.*?(?=^## )", text, flags=re.S | re.M)
    assert m, "SECURITY.md has no '## Containment' section"
    section = m.group(0).replace(matrix.security_block(text) or "", "")
    typed = re.findall(r"\b(?:PASS|FAIL|N-A|ERROR)\b", section)
    assert not typed, f"status words typed by hand in SECURITY.md's containment section: {typed}"


# ---------------------------------------------------------------- the table itself

def test_every_probe_has_a_program_and_a_claim_per_mode():
    for probe in matrix.PROBES:
        assert (matrix.PROBES_DIR / f"probe_{probe.replace('-', '_')}.py").exists(), probe
        assert set(matrix.CLAIMS[probe]) == set(matrix.MODES), probe
    assert set(matrix.PROBE_FNS) == set(matrix.PROBES)


def test_status_vocabulary():
    assert matrix.status_for(matrix.BLOCKED, "blocked") == matrix.PASS
    assert matrix.status_for(matrix.BLOCKED, "succeeded") == matrix.FAIL
    assert matrix.status_for(matrix.ADVISORY, "succeeded") == matrix.NA
    assert matrix.status_for(matrix.ADVISORY, "blocked") == matrix.NA
    assert matrix.status_for(matrix.BLOCKED, "error") == matrix.ERROR
