"""Kenneth French data library adapter: factor files and pre-formed sorted portfolios.
Percent returns are converted to decimals here, once (a documented source quirk)."""
from __future__ import annotations

import io
import zipfile

import httpx
import pandas as pd

from ..cache import cached_fetch
from ..schemas import Factors, PortfolioReturns

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

FILES = {
    "ff3": "F-F_Research_Data_Factors_CSV.zip",
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom_factor": "F-F_Momentum_Factor_CSV.zip",
    "10_mom": "10_Portfolios_Prior_12_2_CSV.zip",
    "10_size": "Portfolios_Formed_on_ME_CSV.zip",
    "10_bm": "Portfolios_Formed_on_BE-ME_CSV.zip",
    "10_op": "Portfolios_Formed_on_OP_CSV.zip",
    "10_inv": "Portfolios_Formed_on_INV_CSV.zip",
    "25_size_bm": "25_Portfolios_5x5_CSV.zip",
    "10_str": "10_Portfolios_Prior_1_0_CSV.zip",
    "10_ltr": "10_Portfolios_Prior_60_13_CSV.zip",
    "10_ind": "10_Industry_Portfolios_CSV.zip",
    "10_beta": "Portfolios_Formed_on_BETA_CSV.zip",
    "10_var": "Portfolios_Formed_on_VAR_CSV.zip",
    "10_ac": "Portfolios_Formed_on_AC_CSV.zip",
    "10_ni": "Portfolios_Formed_on_NI_CSV.zip",
}


def probe() -> tuple[bool, str]:
    try:
        r = httpx.head(BASE + FILES["ff3"], timeout=15, follow_redirects=True)
        return r.status_code == 200, f"HTTP {r.status_code}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _download(name: str) -> str:
    r = httpx.get(BASE + FILES[name], timeout=60, follow_redirects=True)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return z.read(z.namelist()[0]).decode("latin-1")


def _parse_block(text: str, block_index: int, monthly: bool = True) -> pd.DataFrame:
    """French CSVs are several blocks separated by blank lines with a title line. Return block k."""
    lines = text.splitlines()
    blocks: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        if ln.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)
    # drop header/preamble blocks: a data block has a header line starting with ',' and rows starting with digits
    data_blocks = []
    for b in blocks:
        rows = [ln for ln in b if ln[:1].isdigit()]
        if rows:
            hdr = next((ln for ln in b if ln.startswith(",")), None)
            data_blocks.append((hdr, rows, b[0]))
    hdr, rows, title = data_blocks[block_index]
    cols = ["date"] + [c.strip() for c in hdr.split(",")[1:]]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=cols)
    df = df[df["date"].astype(str).str.len() == (6 if monthly else 4)]
    if monthly:
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y")
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([-99.99, -999], pd.NA).astype(float) / 100.0
    return df.reset_index(drop=True)


def fetch(params: dict) -> tuple[pd.DataFrame, str]:
    """params: {name: '10_mom', block: 0 (VW monthly) | 1 (EW monthly) | ..., weighting: 'VW'|'EW'}"""
    name = params["name"]
    if name in ("ff3", "ff5", "mom_factor"):
        def _f():
            df = _parse_block(_download(name), 0)
            ren = {"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RF": "rf", "RMW": "rmw", "CMA": "cma", "Mom": "mom"}
            df = df.rename(columns={c: ren.get(c, c.lower()) for c in df.columns})
            return df
        df, key = cached_fetch("french_library", params, _f)
        if name != "mom_factor":
            Factors.validate(df)
        return df, key
    block = params.get("block")
    if block is None:
        block = {"VW": 0, "EW": 1}[params.get("weighting", "VW")]
    df, key = cached_fetch("french_library", {**params, "block": block}, lambda: _parse_block(_download(name), block))
    PortfolioReturns.validate(df)
    return df, key


def fetch_firm_counts(name: str) -> pd.DataFrame:
    """The 'Number of Firms in Portfolios' block, for the Table 1 checkpoint."""
    text = _download(name)
    lines = text.splitlines()
    idx = next(i for i, ln in enumerate(lines) if "Number of Firms" in ln)
    rows = []
    for ln in lines[idx + 1:]:
        if ln.strip() == "" and rows:
            break
        if ln[:1].isdigit():
            rows.append(ln)
    hdr = lines[idx + 1]
    cols = ["date"] + [c.strip() for c in hdr.split(",")[1:]]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=cols)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    return df
