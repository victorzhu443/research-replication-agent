"""Cross-sectional anomaly template.

Two entry points:
  * `portfolio_pipeline`  : the paper's sorted portfolios already exist as a data product
                            (French library). Nodes: load -> long_short -> evaluate.
  * `stock_pipeline`      : stock-level panel with a signal column. Nodes:
                            load -> signal -> sort -> portfolios -> long_short -> evaluate.
Both end in `evaluate`, which produces the metrics the claims are compared against:
mean_return (per period), t_stat (Newey-West optional), sharpe (per period), alpha_ff3, t_alpha,
decile_monotonic (fraction of adjacent decile pairs increasing), n_periods.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from replicator.data.adapters import french


# ----------------------------------------------------------------------------- shared nodes

def evaluate(ls: pd.DataFrame, factors: pd.DataFrame | None, config: dict[str, Any]) -> dict[str, float]:
    """ls: columns [date, ls] (+ optional decile columns d1..dK). Returns the metrics dict."""
    r = ls.dropna(subset=["ls"])
    x = r["ls"].to_numpy(dtype=float)
    T = len(x)
    if T < 12:
        return {"n_periods": T}
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    lags = int(config.get("nw_lags", 0))
    ols = sm.OLS(x, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": lags}) if lags else sm.OLS(x, np.ones(T)).fit()
    out = {
        "n_periods": T,
        "mean_return": mean,
        "t_stat": float(ols.tvalues[0]),
        "sharpe": mean / sd if sd else float("nan"),
        "sharpe_annualized": (mean / sd) * math.sqrt(12) if sd else float("nan"),
    }
    dec = [c for c in r.columns if c.startswith("d") and c[1:].isdigit()]
    if len(dec) >= 3:
        means = [r[c].mean() for c in sorted(dec, key=lambda c: int(c[1:]))]
        pairs = list(zip(means[:-1], means[1:]))
        out["decile_monotonic"] = float(np.mean([b > a for a, b in pairs]))
        out["decile_spread"] = float(means[-1] - means[0])
    if factors is not None:
        f = factors.merge(r[["date", "ls"]], on="date", how="inner")
        X = sm.add_constant(f[["mkt_rf", "smb", "hml"]].to_numpy(dtype=float))
        y = f["ls"].to_numpy(dtype=float)
        if lags:
            fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        else:
            fit = sm.OLS(y, X).fit()
        out["alpha_ff3"] = float(fit.params[0])
        out["t_alpha_ff3"] = float(fit.tvalues[0])
    return out


def _window(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if config.get("sample_start"):
        df = df[df["date"] >= pd.Timestamp(config["sample_start"]) + pd.offsets.MonthEnd(0)]
    if config.get("sample_end"):
        df = df[df["date"] <= pd.Timestamp(config["sample_end"]) + pd.offsets.MonthEnd(0)]
    return df


# ------------------------------------------------------------------- portfolio-level pipeline

def portfolio_pipeline(config: dict[str, Any], seed: int = 0) -> tuple[dict[str, float], dict[str, Any]]:
    """config keys: french_name ('10_mom' | '10_size' | '10_bm' | '10_op' | '10_inv' | '10_str' | '10_ltr' | '10_ind'),
    weighting ('VW'|'EW'), split ('decile'|'quintile'|'tercile' for the Formed_on_X files), sign (+1 high-minus-low,
    -1 low-minus-high), holding_months, return_def, nw_lags, overlap_inference, sample_start, sample_end, _shuffle_labels."""
    name = config.get("french_name", "10_mom")
    weighting = config.get("weighting", "VW")
    ports, key = french.fetch({"name": name, "weighting": weighting})
    ff3, key_f = french.fetch({"name": "ff3"})
    ports = _window(ports, config)
    cols = [c for c in ports.columns if c != "date"]
    # French "Portfolios_Formed_on_ME/BE-ME/OP/INV" files hold tercile, quintile and decile
    # splits side by side; pick one split. Prior-return and industry files have one split.
    split = config.get("split", "decile")
    pats = {"decile": ("Lo 10", "Dec ", "Hi 10"), "quintile": ("Lo 20", "Qnt ", "Hi 20"), "tercile": ("Lo 30", "Med 40", "Hi 30")}[split]
    sel = [c for c in cols if any(c.startswith(pt) for pt in pats)]
    if sel:
        cols = sel
    K = len(cols)
    df = ports.rename(columns={c: f"d{i+1}" for i, c in enumerate(cols)})
    if config.get("_shuffle_labels"):
        # shuffle portfolio identities within each period: destroys the sort/return link
        rng = np.random.default_rng(seed)
        vals = df[[f"d{i+1}" for i in range(K)]].to_numpy().copy()
        for row in range(len(vals)):
            rng.shuffle(vals[row])
        df[[f"d{i+1}" for i in range(K)]] = vals
    top, bot = f"d{K}", "d1"
    if config.get("return_def", "simple") == "log":
        cohort = np.log1p(df[top]) - np.log1p(df[bot])
    else:
        cohort = df[top] - df[bot]
    h = int(config.get("holding_months", 1) or 1)
    if int(config.get("sign", 1)) < 0:      # low-minus-high strategies (reversal, size, investment)
        cohort = -cohort
    df["ls"] = cohort.rolling(h, min_periods=h).mean() if h > 1 else cohort
    metrics = evaluate(df, ff3, config)
    metrics["_shuffled"] = bool(config.get("_shuffle_labels"))
    if h > 1 and config.get("overlap_inference", "cohort") == "cohort":
        # Overlapping K-month holding: the rolling mean of one monthly-rebalanced series smooths
        # away variance that K distinct cohorts would keep, inflating a plain t-stat. Use the
        # non-overlapping cohort series' SE for inference (the paper's K cohorts are the closer
        # analogue). 'rolling' keeps the smoothed-series SE; nw_lags adds HAC on top of either.
        xc = cohort.loc[df["ls"].notna()].to_numpy(dtype=float)
        xc = xc[~np.isnan(xc)]
        if len(xc) > 12:
            se_c = xc.std(ddof=1) / math.sqrt(len(xc))
            metrics["t_stat_rolling"] = metrics["t_stat"]
            metrics["t_stat"] = metrics["mean_return"] / se_c if se_c else float("nan")
    inter = {"data_hashes": {f"french:{name}:{weighting}": key, "french:ff3": key_f},
             "n_portfolios": K, "first_date": str(df["date"].min().date()), "last_date": str(df["date"].max().date())}
    return metrics, inter


# ------------------------------------------------------------------------ stock-level pipeline

def sort_portfolios(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """panel: [date, id, signal, ret_fwd, mktcap]. Quantile sort at each date on `signal`
    (available at date), returns held over the next period (`ret_fwd`)."""
    q = float(config.get("quantile", 0.1))
    K = int(round(1 / q))
    w = config.get("weighting", "EW")
    pf = float(config.get("price_filter", 0) or 0)
    p = panel.dropna(subset=["signal", "ret_fwd"]).copy()
    if pf and "prc" in p.columns:
        p = p[p["prc"] >= pf]
    if config.get("_shuffle_labels"):
        rng = np.random.default_rng(int(config.get("_seed", 0)))
        p["ret_fwd"] = p.groupby("date")["ret_fwd"].transform(lambda s: rng.permutation(s.to_numpy()))
    p["bucket"] = p.groupby("date")["signal"].transform(lambda s: pd.qcut(s.rank(method="first"), K, labels=False)) + 1
    if w == "VW" and "mktcap" in p.columns:
        p["w"] = p["mktcap"].clip(lower=0)
        g = p.groupby(["date", "bucket"]).apply(lambda d: np.average(d["ret_fwd"], weights=d["w"]) if d["w"].sum() > 0 else d["ret_fwd"].mean(),
                                                include_groups=False)
    else:
        g = p.groupby(["date", "bucket"])["ret_fwd"].mean()
    wide = g.unstack("bucket")
    wide.columns = [f"d{int(c)}" for c in wide.columns]
    wide = wide.reset_index()
    wide["ls"] = wide[f"d{K}"] - wide["d1"]
    return wide


def stock_pipeline(panel: pd.DataFrame, factors: pd.DataFrame | None, config: dict[str, Any], seed: int = 0) -> dict[str, float]:
    config = {**config, "_seed": seed}
    wide = sort_portfolios(_window(panel, config), config)
    m = evaluate(wide, factors, config)
    m["_shuffled"] = bool(config.get("_shuffle_labels"))
    return m


def momentum_signal(panel: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """Reference paper-specific node: prior (t-12, t-2) cumulative return, available at t."""
    p = panel.sort_values(["id", "date"]).copy()
    p["_lr"] = np.log1p(p["ret"])
    roll = p.groupby("id")["_lr"].transform(lambda s: s.rolling(lookback - skip, min_periods=lookback - skip).sum())
    p["signal"] = roll.groupby(p["id"]).shift(skip)  # skip the most recent `skip` months
    p["ret_fwd"] = p.groupby("id")["ret"].shift(-1)
    return p.drop(columns=["_lr"])
