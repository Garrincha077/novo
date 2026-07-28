from __future__ import annotations

import base64
import gzip
import io
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay

import contrarian_backtest as cb
from cboe_pc_cache import CBOE_PC_CACHE_GZIP_B64

FRED_CACHE = Path("output/fred_hy_oas.csv")
OFFICIAL_FRED_OVERLAY = """date,value
2026-03-23,3.19
2026-03-24,3.19
2026-03-25,3.17
2026-03-26,3.21
2026-03-27,3.42
2026-03-30,3.46
2026-03-31,3.28
2026-04-01,3.16
2026-04-02,3.17
2026-04-03,3.13
2026-04-06,3.05
2026-04-07,3.12
2026-04-08,2.94
2026-04-09,2.90
2026-04-10,2.94
2026-04-13,2.95
2026-04-14,2.84
2026-04-15,2.85
2026-04-16,2.86
2026-04-17,2.83
2026-04-20,2.87
2026-04-21,2.85
2026-04-22,2.84
2026-04-23,2.86
2026-04-24,2.86
2026-04-27,2.84
2026-04-28,2.85
2026-04-29,2.82
2026-04-30,2.83
2026-05-01,2.77
2026-05-04,2.78
2026-05-05,2.77
2026-05-06,2.75
2026-05-07,2.79
2026-05-08,2.81
2026-05-11,2.79
2026-05-12,2.82
2026-05-13,2.82
2026-05-14,2.76
2026-05-15,2.80
2026-05-18,2.83
2026-05-19,2.86
2026-05-20,2.80
2026-05-21,2.78
2026-05-22,2.74
2026-05-25,2.74
2026-05-26,2.72
2026-05-27,2.71
2026-05-28,2.72
2026-05-29,2.72
2026-05-31,2.74
2026-06-01,2.72
2026-06-02,2.71
2026-06-03,2.75
2026-06-04,2.74
2026-06-05,2.76
2026-06-08,2.75
2026-06-09,2.78
2026-06-10,2.80
2026-06-11,2.78
2026-06-12,2.71
2026-06-15,2.66
2026-06-16,2.71
2026-06-17,2.63
2026-06-18,2.66
2026-06-19,2.66
2026-06-22,2.65
2026-06-23,2.71
2026-06-24,2.76
2026-06-25,2.78
2026-06-26,2.83
2026-06-29,2.80
2026-06-30,2.75
2026-07-01,2.74
2026-07-02,2.75
2026-07-03,2.74
2026-07-06,2.72
2026-07-07,2.67
2026-07-08,2.70
2026-07-09,2.70
2026-07-10,2.69
2026-07-13,2.69
2026-07-14,2.72
2026-07-15,2.71
2026-07-16,2.71
2026-07-17,2.73
2026-07-20,2.69
2026-07-21,2.69
2026-07-22,2.68
2026-07-23,2.77
2026-07-24,2.79
"""


def load_cboe_pc_from_audited_cache(trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    raw = gzip.decompress(base64.b64decode(CBOE_PC_CACHE_GZIP_B64))
    df = pd.read_csv(io.BytesIO(raw))
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    df["total_pc"] = pd.to_numeric(df["total_pc"], errors="coerce")
    df["equity_pc"] = pd.to_numeric(df["equity_pc"], errors="coerce")
    df = df.dropna(subset=["date", "total_pc", "equity_pc"]).drop_duplicates("date", keep="last")
    requested = pd.DatetimeIndex(trading_dates).normalize().astype("datetime64[ns]")
    out = df.set_index("date").reindex(requested)
    missing = out[["total_pc", "equity_pc"]].isna().any(axis=1)
    if missing.any():
        raise RuntimeError(f"CBOE cache missing {int(missing.sum())} requested sessions")
    df.to_csv(cb.OUT / "cboe_pc.csv", index=False)
    (cb.OUT / "cboe_source_audit.txt").write_text(
        f"Audited exact-date Cboe daily statistics cache: {len(df)} rows; "
        f"requested {len(requested)} rows; missing 0.\n"
    )
    return out


def load_hy_oas_from_audited_cache() -> pd.DataFrame:
    if not FRED_CACHE.exists():
        raise RuntimeError(f"HY OAS cache missing: {FRED_CACHE}")

    base = pd.read_csv(FRED_CACHE)
    date_col = base.columns[0]
    value_col = next((c for c in base.columns if c != date_col), None)
    if value_col is None:
        raise RuntimeError(f"HY OAS cache has no value column: {base.columns.tolist()}")

    base = pd.DataFrame({
        "observation_date": pd.to_datetime(base[date_col], errors="coerce").dt.normalize().astype("datetime64[ns]"),
        "hy_oas": pd.to_numeric(base[value_col], errors="coerce"),
    }).dropna()

    overlay = pd.read_csv(io.StringIO(OFFICIAL_FRED_OVERLAY))
    overlay = pd.DataFrame({
        "observation_date": pd.to_datetime(overlay["date"], errors="coerce").dt.normalize().astype("datetime64[ns]"),
        "hy_oas": pd.to_numeric(overlay["value"], errors="coerce"),
    }).dropna()

    combined = pd.concat([base, overlay], ignore_index=True)
    combined = combined.drop_duplicates("observation_date", keep="last").sort_values("observation_date")
    combined = combined[(combined["observation_date"] >= cb.WARMUP) & (combined["observation_date"] <= cb.END)]
    combined["effective_date"] = (combined["observation_date"] + BDay(1)).astype("datetime64[ns]")
    combined.to_csv(cb.OUT / "fred_hy_oas_audited.csv", index=False)
    (cb.OUT / "hy_oas_source_audit.txt").write_text(
        "Base transport: public mirror of FRED series through 2026-03-20.\n"
        "Official overlay: FRED table data, 2026-03-23 through 2026-07-24.\n"
        "Canonical series: BAMLH0A0HYM2. Observation effective one U.S. business day later.\n"
    )
    return combined


def robust_asof_value(base: pd.DataFrame, releases: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    left = base.reset_index()
    index_col = base.index.name if base.index.name in left.columns else left.columns[0]
    left = left.rename(columns={index_col: "date"})
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    left = left.drop(columns=[c for c in left.columns if c.startswith("effective_date")], errors="ignore")
    left = left.sort_values("date")

    right = releases[["effective_date"] + value_cols].copy()
    right["effective_date"] = pd.to_datetime(right["effective_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    right = right.dropna(subset=["effective_date"]).sort_values("effective_date")

    out = pd.merge_asof(left, right, left_on="date", right_on="effective_date", direction="backward")
    out = out.drop(columns=["effective_date"], errors="ignore")
    return out.set_index("date")


cb.load_cboe_pc = load_cboe_pc_from_audited_cache
cb.load_hy_oas = load_hy_oas_from_audited_cache
cb.asof_value = robust_asof_value
cb.main()
