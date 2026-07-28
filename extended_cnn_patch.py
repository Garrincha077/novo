from __future__ import annotations

from pathlib import Path

import pandas as pd

import contrarian_backtest as cb

OUT = Path("output_extended")
ARCHIVE = OUT / "cnn_fear_greed_archive.csv"
OFFICIAL_START = pd.Timestamp("2021-02-01")


def _archive_series() -> pd.Series:
    df = pd.read_csv(ARCHIVE)
    normalized = {str(c).strip().lower(): c for c in df.columns}
    date_col = normalized.get("date")
    score_col = normalized.get("fear greed") or normalized.get("fear_greed") or normalized.get("score")
    if date_col is None or score_col is None:
        raise RuntimeError(f"Unexpected CNN archive columns: {df.columns.tolist()}")
    idx = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    score = pd.to_numeric(df[score_col], errors="coerce")
    series = pd.Series(score.values, index=idx, name="cnn_fg").dropna()
    return series[~series.index.duplicated(keep="last")].sort_index()


def _official_series() -> pd.Series:
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2021-01-01"
    response = cb.get(url, timeout=90, retries=4)
    data = response.json()["fear_and_greed_historical"]["data"]
    df = pd.DataFrame(data)
    idx = pd.to_datetime(df["x"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    score = pd.to_numeric(df["y"], errors="coerce")
    series = pd.Series(score.values, index=idx, name="cnn_fg").dropna()
    return series[~series.index.duplicated(keep="last")].sort_index()


def load_cnn_extended() -> pd.Series:
    archive = _archive_series()
    official_status = "official_success"
    try:
        official = _official_series()
    except Exception as exc:
        official = pd.Series(dtype=float, name="cnn_fg")
        official_status = f"official_failed_archive_used: {exc}"

    pre = archive[archive.index < OFFICIAL_START]
    post = official[official.index >= OFFICIAL_START] if len(official) else archive[archive.index >= OFFICIAL_START]
    combined = pd.concat([pre, post]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    pd.DataFrame({"date": combined.index, "cnn_fg": combined.values}).to_csv(OUT / "cnn_history_audited.csv", index=False)
    (OUT / "cnn_source_audit.txt").write_text(
        "2020-01-02 through 2021-01-29: whit3rabbit/fear-greed-data frozen pre-2021 archive; secondary historical reconstruction.\n"
        "2021-02-01 onward: official CNN graphdata endpoint when available; official values override overlap.\n"
        f"Official endpoint status: {official_status}\n",
        encoding="utf-8",
    )
    return combined


cb.load_cnn = load_cnn_extended
