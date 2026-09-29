"""Finance sweep on open data: specs built from Open Source Asset Pricing's SignalDoc rows (the
expert-extracted ground truth) plus Kenneth French's pre-formed sorted portfolios (tier A), the
live builder writes each pipeline, verify compares to the OSAP-documented return and t-stat.

    uv run python -m evals.finance_batch            # all
    uv run python -m evals.finance_batch str size   # subset
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from replicator.kb import signaldoc_rows
from replicator.orchestrator import Orchestrator
from replicator.schema import Ambiguity, Claim, DataSource, DataSpec, Method, PaperMeta, Spec, Units
from evals.batch import _row, log as _log, write_summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "finance"

# slug -> OSAP acronym, French file, sign (+1 high-minus-low), notes on how the French file relates to the paper's sort
PAPERS = {
    "str":   dict(acronym="STreversal", french="10_str", sign=-1, title="Evidence of Predictable Behavior of Security Returns", authors=["Jegadeesh"], year=1990,
                  note="French 10 portfolios formed on prior 1-month return; the reversal strategy buys last month's losers (low-minus-high)."),
    "ltr":   dict(acronym="LRreversal", french="10_ltr", sign=-1, title="Does the Stock Market Overreact?", authors=["De Bondt", "Thaler"], year=1985,
                  note="French 10 portfolios formed on prior (60,13) return; long-term losers minus winners. OSAP reports a 3-year cumulative-style number; compare the monthly spread direction and t-stat."),
    "size":  dict(acronym="Size", french="10_size", sign=-1, title="The Relationship Between Return and Market Value of Common Stocks", authors=["Banz"], year=1981,
                  note="French portfolios formed on ME (deciles); small-minus-big. OSAP uses a median split (q=0.5); French deciles are used, a documented deviation."),
    "op":    dict(acronym="GP", french="10_op", sign=+1, title="The Other Side of Value: The Gross Profitability Premium", authors=["Novy-Marx"], year=2013,
                  note="French portfolios formed on operating profitability (OP, not exactly gross profitability); OSAP row uses VW quintiles: set split=quintile, weighting=VW."),
    "inv":   dict(acronym="AssetGrowth", french="10_inv", sign=-1, title="Asset Growth and the Cross-Section of Stock Returns", authors=["Cooper", "Gulen", "Schill"], year=2008,
                  note="French portfolios formed on investment (asset growth); low-minus-high investment. EW deciles."),
    "mom6":  dict(acronym="Mom6m", french="10_mom", sign=+1, title="Returns to Buying Winners and Selling Losers (J=6, K=3)", authors=["Jegadeesh", "Titman"], year=1993,
                  note="French forms on (12,2) prior return, not (6,1); documented deviation. EW, holding 3 months."),
}


def spec_from_signaldoc(slug: str) -> Spec:
    p = PAPERS[slug]
    row = next(r for r in signaldoc_rows() if r["Acronym"] == p["acronym"])
    y0, y1 = int(row["SampleStartYear"]), int(row["SampleEndYear"])
    n_months = (y1 - y0 + 1) * 12
    w = row["Stock Weight"] or "EW"
    q = float(row["LS Quantile"] or 0.1)
    split = "quintile" if abs(q - 0.2) < 1e-6 else ("tercile" if q > 0.3 else "decile")
    hold = int(float(row["Portfolio Period"] or 1))
    claims = []
    if row["Return"]:
        claims.append(Claim(id=f"{p['acronym']}.ret", where=f"OSAP SignalDoc {p['acronym']}: original paper {row['Key Table in OP']}, long-short {'monthly' if hold <= 12 else ''} return (percent)",
                            metric="mean_return_pct", units=Units(period="monthly", scale="percent"), value=float(row["Return"]), reported_precision=0.005,
                            n_periods=n_months, reported_t_stat=float(row["T-Stat"]) if row["T-Stat"] else None,
                            sample_start=f"{y0}-01", sample_end=f"{y1}-12", method_variant="default", priority="headline", relation="gt" if abs(float(row["Return"])) < 0.05 else "eq"))
    if row["T-Stat"]:
        claims.append(Claim(id=f"{p['acronym']}.t", where=f"OSAP SignalDoc {p['acronym']}: t-statistic", metric="t_stat", units=Units(period="monthly"),
                            value=float(row["T-Stat"]), reported_precision=0.005, n_periods=n_months, sample_start=f"{y0}-01", sample_end=f"{y1}-12",
                            priority="secondary"))
    if not claims:
        raise ValueError(f"{slug}: no numeric claims in SignalDoc")
    spec = Spec(
        paper=PaperMeta(title=p["title"], authors=p["authors"], year=p["year"], track="finance", family="cross_sectional_anomaly"),
        claims=claims,
        data=DataSpec(sources=[DataSource(canonical="french_library", description=f"Kenneth French pre-formed portfolios: {p['french']}")], frequency="monthly"),
        method=Method(signal=row["Detailed Definition"], evaluation=["mean_return_pct", "t_stat"],
                      variants={"default": {"french_name": p["french"], "weighting": w, "split": split, "sign": p["sign"], "holding_months": hold,
                                            "sample_start": f"{y0}-01", "sample_end": f"{y1}-12"}}),
        ambiguities=[
            Ambiguity(id="A1", question="Equal- or value-weighted?", config_key="weighting", options=["EW", "VW"], default=w, source="conventions_kb", reason="OSAP Stock Weight", sensitivity="high"),
            Ambiguity(id="A2", question="Holding period in months?", config_key="holding_months", options=["1", "3", "6", "12"], default=str(hold), source="conventions_kb", reason="OSAP Portfolio Period", sensitivity="medium"),
            Ambiguity(id="A3", question="Overlapping-holding inference: cohort or rolling SE?", config_key="overlap_inference", options=["cohort", "rolling"], default="cohort", source="guess", reason="pre-formed monthly portfolios", sensitivity="medium"),
            Ambiguity(id="A4", question="Newey-West lags?", config_key="nw_lags", options=["0", "3", "6"], default="0", source="guess", reason="OSAP reports plain t-stats", sensitivity="low"),
        ],
        descriptive_stats=[],
    )
    spec.plan.substitutions["french_portfolios"] = p["note"]
    spec.plan.seeds = [0]
    return spec


def run(slug: str) -> dict:
    d = OUT / slug
    d.mkdir(parents=True, exist_ok=True)
    orch = Orchestrator(d, batch=True)
    spec = spec_from_signaldoc(slug)
    spec = orch.stage_triage(spec)
    t0 = time.time()
    rep = orch.replicate(spec, do_grid=True)
    row = _row(json.loads(rep.model_dump_json()))
    row.update(slug=slug, minutes=round((time.time() - t0) / 60, 1), cost=round(orch.llm.cost_usd if orch.llm else 0, 2))
    return row


def main(slugs: list[str]) -> None:
    import evals.batch as B
    B.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    slugs = slugs or list(PAPERS)
    _log(f"finance batch start: {slugs}")
    rows = []
    for slug in slugs:
        _log(f"replicate {slug} ...")
        try:
            row = run(slug)
        except Exception as e:  # noqa: BLE001
            row = {"slug": slug, "grade": "-", "failure": f"{type(e).__name__}: {str(e)[:120]}"}
        rows.append(row)
        _log(f"{slug}: grade {row.get('grade')} claims [{row.get('claims')}] leakage [{row.get('leakage')}] {row.get('failure','')}")
        write_summary(rows)


if __name__ == "__main__":
    main(sys.argv[1:])
