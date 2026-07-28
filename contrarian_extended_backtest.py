from __future__ import annotations

import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

import contrarian_backtest as cb

OUT = Path("output_extended")
OUT.mkdir(exist_ok=True)
START = pd.Timestamp("2020-01-02")
END = pd.Timestamp("2026-07-27")
WARMUP = pd.Timestamp("2018-01-01")
INITIAL = 10000.0

# Lock the canonical model inputs and extend only the research window.
cb.OUT = OUT
cb.START = START
cb.END = END
cb.WARMUP = WARMUP
cb.SOURCES["cnn"] = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2020-01-01"

_PRICE_CACHE: pd.DataFrame | None = None
_ORIGINAL_LOAD_PRICES = cb.load_prices


def cached_load_prices() -> pd.DataFrame:
    global _PRICE_CACHE
    if _PRICE_CACHE is None:
        _PRICE_CACHE = _ORIGINAL_LOAD_PRICES()
        _PRICE_CACHE.to_csv(OUT / "market_prices_warmup.csv")
    return _PRICE_CACHE.copy()


def robust_asof_value(base: pd.DataFrame, releases: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    left = base.reset_index()
    index_col = base.index.name if base.index.name in left.columns else left.columns[0]
    left = left.rename(columns={index_col: "date"})
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    left = left.drop(columns=[c for c in left.columns if str(c).startswith("effective_date")], errors="ignore")
    left = left.sort_values("date")

    right = releases[["effective_date"] + value_cols].copy()
    right["effective_date"] = pd.to_datetime(right["effective_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    right = right.dropna(subset=["effective_date"]).sort_values("effective_date")
    out = pd.merge_asof(left, right, left_on="date", right_on="effective_date", direction="backward")
    return out.drop(columns=["effective_date"], errors="ignore").set_index("date")


def load_hy_oas_audited() -> pd.DataFrame:
    mirror_path = Path("output_extended/fred_hy_oas.csv")
    if not mirror_path.exists():
        raise RuntimeError(f"HY OAS transport missing: {mirror_path}")
    base = pd.read_csv(mirror_path)
    date_col = base.columns[0]
    value_col = next(c for c in base.columns if c != date_col)
    base = pd.DataFrame({
        "observation_date": pd.to_datetime(base[date_col], errors="coerce").dt.normalize(),
        "hy_oas": pd.to_numeric(base[value_col], errors="coerce"),
    }).dropna()

    # Reuse the previously audited official FRED overlay stored in the research branch.
    wrapper_text = Path("run_backtest_with_patch.py").read_text(encoding="utf-8")
    match = re.search(r'OFFICIAL_FRED_OVERLAY\s*=\s*"""(.*?)"""', wrapper_text, re.S)
    if not match:
        raise RuntimeError("Could not extract audited official FRED overlay")
    overlay = pd.read_csv(io.StringIO(match.group(1)))
    overlay = pd.DataFrame({
        "observation_date": pd.to_datetime(overlay["date"], errors="coerce").dt.normalize(),
        "hy_oas": pd.to_numeric(overlay["value"], errors="coerce"),
    }).dropna()

    combined = pd.concat([base, overlay], ignore_index=True)
    combined = combined.drop_duplicates("observation_date", keep="last").sort_values("observation_date")
    combined = combined[(combined["observation_date"] >= WARMUP) & (combined["observation_date"] <= END)]
    combined["effective_date"] = combined["observation_date"] + BDay(1)
    combined.to_csv(OUT / "fred_hy_oas_audited.csv", index=False)
    return combined


def load_cboe_pc_extended(trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.DatetimeIndex(trading_dates).normalize().unique().sort_values()
    rows: list[tuple[pd.Timestamp, float | None, float | None, str]] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(cb.fetch_cboe_one, d): d for d in dates}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 100 == 0:
                print(f"CBOE first pass {i}/{len(futures)}")

    df = pd.DataFrame(rows, columns=["date", "total_pc", "equity_pc", "source_or_error"]).sort_values("date")
    missing_dates = df.loc[df[["total_pc", "equity_pc"]].isna().any(axis=1), "date"].tolist()
    if missing_dates:
        print(f"CBOE retrying {len(missing_dates)} missing sessions")
        retry_rows = []
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {ex.submit(cb.fetch_cboe_one, d): d for d in missing_dates}
            for fut in as_completed(futures):
                retry_rows.append(fut.result())
        retry = pd.DataFrame(retry_rows, columns=df.columns)
        df = pd.concat([df[~df["date"].isin(missing_dates)], retry], ignore_index=True).sort_values("date")

    df.to_csv(OUT / "cboe_pc.csv", index=False)
    success = float(df[["total_pc", "equity_pc"]].notna().all(axis=1).mean())
    print("CBOE exact-date success rate", success)
    return df.set_index("date")


cb.load_prices = cached_load_prices
cb.asof_value = robust_asof_value
cb.load_hy_oas = load_hy_oas_audited
cb.load_cboe_pc = load_cboe_pc_extended


@dataclass
class Position:
    side: int = 0
    qty: float = 0.0
    entry: float = np.nan
    original_qty: float = 0.0
    stop4_done: bool = False
    stop6_done: bool = False


def add_regime(d: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    weekly = prices["SPY_close"].resample("W-FRI").last().dropna()
    sma30 = weekly.rolling(30, min_periods=30).mean()
    slope = sma30.diff()
    weekly_frame = pd.DataFrame({"sma30w": sma30, "sma30w_slope": slope})
    mapped = weekly_frame.reindex(d.index, method="ffill")
    d = d.join(mapped)
    d["regime"] = np.select(
        [
            (d["SPY_close"] > d["sma30w"]) & (d["sma30w_slope"] > 0),
            (d["SPY_close"] < d["sma30w"]) & (d["sma30w_slope"] < 0),
        ],
        ["BULL", "BEAR"],
        default="TRANSITION",
    )
    d["bull_ema"] = (d["SPY_close"] > d["spy_ema10"]) & (d["spy_ema10"] > d["spy_ema20"])
    d["bear_ema"] = (d["SPY_close"] < d["spy_ema10"]) & (d["spy_ema10"] < d["spy_ema20"])
    d["bull_5d"] = d["spy_ret5"] >= 0.02
    d["bear_5d"] = d["spy_ret5"] <= -0.02
    return d


def qualification(row: pd.Series, mode: str) -> tuple[bool, bool]:
    if mode == "none":
        return True, True
    if mode == "canonical":
        return bool(row["bullish_confirmation"]), bool(row["bearish_confirmation"])
    if mode == "ema":
        return bool(row["bull_ema"]), bool(row["bear_ema"])
    if mode == "5d":
        return bool(row["bull_5d"]), bool(row["bear_5d"])
    raise ValueError(mode)


def run_strategy(
    d: pd.DataFrame,
    strategy: str,
    confirmation_mode: str = "canonical",
    long_threshold: float = 35,
    short_threshold: float = 65,
    initial: float = INITIAL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash = initial
    pos = Position()
    pending: str | None = None
    long_lock = False
    long_cleared = False
    short_lock = False
    short_cleared = False
    events: list[dict] = []
    daily: list[dict] = []

    def equity_at(px: float) -> float:
        return cash + pos.qty * px

    def close_qty(qty_to_close: float, px: float, date: pd.Timestamp, reason: str) -> None:
        nonlocal cash, pos
        signed_close = math.copysign(qty_to_close, pos.qty)
        cash += signed_close * px
        events.append({
            "strategy": strategy,
            "date": date,
            "event": "EXIT",
            "side": "LONG" if pos.side == 1 else "SHORT",
            "qty": qty_to_close,
            "price": px,
            "reason": reason,
            "cash_after": cash,
        })
        pos.qty -= signed_close
        if abs(pos.qty) < 1e-9:
            pos = Position()

    def open_side(side: int, px: float, date: pd.Timestamp, reason: str) -> None:
        nonlocal cash, pos
        qty_abs = cash / px
        signed = side * qty_abs
        cash -= signed * px
        pos = Position(side=side, qty=signed, entry=px, original_qty=qty_abs)
        events.append({
            "strategy": strategy,
            "date": date,
            "event": "ENTRY",
            "side": "LONG" if side == 1 else "SHORT",
            "qty": qty_abs,
            "price": px,
            "reason": reason,
            "cash_after": cash,
        })

    for date, row in d.iterrows():
        op, hi, lo, cl = (float(row[c]) for c in ["SPY_open", "SPY_high", "SPY_low", "SPY_close"])

        if pending == "FLAT":
            if pos.side != 0:
                close_qty(abs(pos.qty), op, date, "REGIME_OR_GREED_EXIT")
        elif pending in ("LONG", "SHORT"):
            desired = 1 if pending == "LONG" else -1
            blocked = (desired == 1 and long_lock and not long_cleared) or (desired == -1 and short_lock and not short_cleared)
            if not blocked and pos.side != desired:
                if pos.side != 0:
                    close_qty(abs(pos.qty), op, date, "OPPOSITE_SIGNAL")
                open_side(desired, op, date, "QUALIFIED_THRESHOLD_SIGNAL")
                if desired == 1:
                    long_lock = False
                    long_cleared = False
                else:
                    short_lock = False
                    short_cleared = False
        pending = None

        stopped_side = pos.side
        fully_stopped = False
        if pos.side == 1:
            levels = [(0.04, 0.25, "STOP_4"), (0.06, 0.25, "STOP_6"), (0.08, 0.50, "STOP_8")]
            for adverse, frac, reason in levels:
                already = pos.stop4_done if adverse == 0.04 else pos.stop6_done if adverse == 0.06 else False
                if already or pos.side == 0:
                    continue
                level = pos.entry * (1 - adverse)
                if lo <= level:
                    fill = op if op < level else level
                    qty = min(pos.original_qty * frac, abs(pos.qty))
                    if adverse == 0.04:
                        pos.stop4_done = True
                    if adverse == 0.06:
                        pos.stop6_done = True
                    close_qty(qty, fill, date, reason)
                    if adverse == 0.08 or pos.side == 0:
                        fully_stopped = True
        elif pos.side == -1:
            levels = [(0.04, 0.25, "STOP_4"), (0.06, 0.25, "STOP_6"), (0.08, 0.50, "STOP_8")]
            for adverse, frac, reason in levels:
                already = pos.stop4_done if adverse == 0.04 else pos.stop6_done if adverse == 0.06 else False
                if already or pos.side == 0:
                    continue
                level = pos.entry * (1 + adverse)
                if hi >= level:
                    fill = op if op > level else level
                    qty = min(pos.original_qty * frac, abs(pos.qty))
                    if adverse == 0.04:
                        pos.stop4_done = True
                    if adverse == 0.06:
                        pos.stop6_done = True
                    close_qty(qty, fill, date, reason)
                    if adverse == 0.08 or pos.side == 0:
                        fully_stopped = True
        if fully_stopped:
            if stopped_side == 1:
                long_lock = True
                long_cleared = False
            elif stopped_side == -1:
                short_lock = True
                short_cleared = False

        comp = row["composite"]
        if pd.notna(comp):
            if long_lock and comp > long_threshold:
                long_cleared = True
            if short_lock and comp < short_threshold:
                short_cleared = True

            bull_ok, bear_ok = qualification(row, confirmation_mode)
            long_signal = comp <= long_threshold and bull_ok and not (long_lock and not long_cleared)
            short_signal = comp >= short_threshold and bear_ok and not (short_lock and not short_cleared)
            regime = row["regime"]

            if strategy == "baseline":
                long_signal = comp <= long_threshold and not (long_lock and not long_cleared)
                short_signal = comp >= short_threshold and not (short_lock and not short_cleared)
                if long_signal:
                    pending = "LONG"
                elif short_signal:
                    pending = "SHORT"
            elif strategy == "A":
                if long_signal:
                    pending = "LONG"
                elif short_signal:
                    pending = "SHORT"
            elif strategy == "B_strict":
                if long_signal and regime == "BULL":
                    pending = "LONG"
                elif short_signal and regime == "BEAR":
                    pending = "SHORT"
                elif (pos.side == 1 and regime != "BULL") or (pos.side == -1 and regime != "BEAR"):
                    pending = "FLAT"
            elif strategy == "B_relaxed":
                if long_signal:
                    pending = "LONG"
                elif short_signal and regime in ("BEAR", "TRANSITION"):
                    pending = "SHORT"
                elif short_signal and regime == "BULL" and pos.side == 1:
                    pending = "FLAT"
                elif pos.side == -1 and regime == "BULL":
                    pending = "FLAT"
            else:
                raise ValueError(strategy)

        daily.append({
            "date": date,
            "equity": equity_at(cl),
            "side": "LONG" if pos.side == 1 else "SHORT" if pos.side == -1 else "FLAT",
            "qty": abs(pos.qty),
            "cash": cash,
        })

    return pd.DataFrame(daily).set_index("date"), pd.DataFrame(events)


def metrics(name: str, curve: pd.Series, sides: pd.Series, trades: pd.DataFrame) -> dict:
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1 / 365.25)
    total_return = float(curve.iloc[-1] / INITIAL - 1)
    cagr = float((curve.iloc[-1] / INITIAL) ** (1 / years) - 1)
    dd = curve / curve.cummax() - 1
    max_dd = float(dd.min())
    ret = curve.pct_change().dropna()
    sharpe = float(np.sqrt(252) * ret.mean() / ret.std()) if ret.std() > 0 else 0.0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else None
    return {
        "strategy": name,
        "final_equity_usd": float(curve.iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe_annualized": sharpe,
        "calmar": calmar,
        "entry_events": int((trades["event"] == "ENTRY").sum()) if len(trades) else 0,
        "trade_events": int(len(trades)),
        "exposure_pct": float((sides != "FLAT").mean()),
        "long_exposure_pct": float((sides == "LONG").mean()),
        "short_exposure_pct": float((sides == "SHORT").mean()),
        "final_side": str(sides.iloc[-1]),
    }


def annual_metrics(curves: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, idx in curves.groupby(curves.index.year).groups.items():
        dates = curves.index[curves.index.year == year]
        for col in curves.columns:
            series = curves.loc[dates, col]
            start_value = INITIAL if year == curves.index[0].year else curves.loc[curves.index[curves.index < dates[0]][-1], col]
            ret = float(series.iloc[-1] / start_value - 1)
            dd = float((series / series.cummax() - 1).min())
            side_col = col if col in sides.columns else None
            rows.append({
                "year": int(year),
                "strategy": col,
                "return": ret,
                "max_drawdown_within_year": dd,
                "sessions": len(series),
                "exposure_pct": float((sides.loc[dates, side_col] != "FLAT").mean()) if side_col else 1.0,
            })
    return pd.DataFrame(rows)


def regime_metrics(d: pd.DataFrame, curves: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    daily_returns = curves.pct_change().fillna(0)
    rows = []
    for regime in ["BULL", "TRANSITION", "BEAR"]:
        mask = d["regime"] == regime
        for col in curves.columns:
            r = daily_returns.loc[mask, col]
            compounded = float((1 + r).prod() - 1)
            rows.append({
                "regime": regime,
                "strategy": col,
                "sessions": int(mask.sum()),
                "compounded_return_on_regime_days": compounded,
                "mean_daily_return": float(r.mean()),
                "positive_day_rate": float((r > 0).mean()),
                "exposure_pct": float((sides.loc[mask, col] != "FLAT").mean()) if col in sides else 1.0,
            })
    return pd.DataFrame(rows)


def validate_two_year(d: pd.DataFrame) -> pd.DataFrame:
    subset = d.loc[pd.Timestamp("2024-07-29"):END]
    expected = {"baseline": -0.21976847424603563, "A": 0.20497929326775255, "B_relaxed": 0.15424827078272085}
    rows = []
    for name in ["baseline", "A", "B_relaxed"]:
        mode = "none" if name == "baseline" else "canonical"
        curve, trades = run_strategy(subset, name, confirmation_mode=mode)
        actual = float(curve["equity"].iloc[-1] / INITIAL - 1)
        rows.append({
            "strategy": name,
            "expected_2y_return": expected[name],
            "actual_2y_return": actual,
            "difference": actual - expected[name],
            "status": "PASS" if abs(actual - expected[name]) <= 0.003 else "REVIEW",
            "entries": int((trades["event"] == "ENTRY").sum()) if len(trades) else 0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    d = cb.build_daily()
    d = add_regime(d, cached_load_prices())
    d.to_csv(OUT / "extended_daily_inputs.csv")

    strategy_specs = [
        ("Baseline v1.0-paper", "baseline", "none"),
        ("Variant A — price confirmation", "A", "canonical"),
        ("Variant B strict — A + 30W gate", "B_strict", "canonical"),
        ("Variant B relaxed — asymmetric regime", "B_relaxed", "canonical"),
    ]

    curves = pd.DataFrame(index=d.index)
    sides = pd.DataFrame(index=d.index)
    all_events = []
    comparison = []
    for label, code, mode in strategy_specs:
        daily, events = run_strategy(d, code, confirmation_mode=mode)
        curves[label] = daily["equity"]
        sides[label] = daily["side"]
        if len(events):
            events["strategy"] = label
            all_events.append(events)
        comparison.append(metrics(label, daily["equity"], daily["side"], events))

    first_open = float(d["SPY_open"].iloc[0])
    bh_qty = INITIAL / first_open
    curves["SPY Buy & Hold"] = bh_qty * d["SPY_close"]
    sides["SPY Buy & Hold"] = "LONG"
    comparison.append(metrics("SPY Buy & Hold", curves["SPY Buy & Hold"], sides["SPY Buy & Hold"], pd.DataFrame([{"event": "ENTRY"}])))

    daily_out = d[[
        "SPY_open", "SPY_high", "SPY_low", "SPY_close", "composite", "coverage", "publishable",
        "regime", "sma30w", "sma30w_slope", "bullish_confirmation", "bearish_confirmation",
        "bull_ema", "bear_ema", "bull_5d", "bear_5d", "spy_ret5",
    ]].copy()
    for col in curves:
        daily_out[f"{col} equity"] = curves[col]
        daily_out[f"{col} side"] = sides[col]
    daily_out.reset_index(names="date").to_csv(OUT / "daily_curves.csv", index=False)

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(OUT / "strategy_comparison.csv", index=False)
    pd.concat(all_events, ignore_index=True).to_csv(OUT / "trade_events.csv", index=False) if all_events else pd.DataFrame().to_csv(OUT / "trade_events.csv", index=False)
    annual_metrics(curves, sides).to_csv(OUT / "annual_metrics.csv", index=False)
    regime_metrics(d, curves, sides).to_csv(OUT / "regime_metrics.csv", index=False)
    validate_two_year(d).to_csv(OUT / "validation_2y.csv", index=False)

    # Confirmation branch diagnostic on the full extended window.
    confirmation_rows = []
    for mode in ["canonical", "ema", "5d"]:
        daily, events = run_strategy(d, "A", confirmation_mode=mode)
        rec = metrics(f"A_{mode}", daily["equity"], daily["side"], events)
        rec["confirmation"] = mode
        confirmation_rows.append(rec)
    pd.DataFrame(confirmation_rows).to_csv(OUT / "confirmation_diagnostics.csv", index=False)

    # Same diagnostic grid as the 2Y test. It is not used to change canonical thresholds.
    sensitivity_rows = []
    for long_threshold in [30, 35, 40]:
        for short_threshold in [60, 65, 70]:
            daily, events = run_strategy(d, "A", confirmation_mode="canonical", long_threshold=long_threshold, short_threshold=short_threshold)
            rec = metrics(f"A_{long_threshold}_{short_threshold}", daily["equity"], daily["side"], events)
            rec["long_threshold"] = long_threshold
            rec["short_threshold"] = short_threshold
            sensitivity_rows.append(rec)
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "threshold_sensitivity.csv", index=False)

    # Source and signal audit.
    raw_cols = ["vix", "equity_pc", "total_pc_10d", "naaim", "aaii_spread", "rsp_spy_20d", "iwm_spy_20d", "cnn_fg", "hy_oas", "hyg_ief_20d", "spy_rsi14", "uup_20d"]
    audit = []
    for col in raw_cols:
        audit.append({
            "leaf": col,
            "valid_sessions": int(d[col].notna().sum()),
            "missing_sessions": int(d[col].isna().sum()),
            "coverage_pct": float(d[col].notna().mean()),
            "first_valid": str(d[col].first_valid_index().date()) if d[col].first_valid_index() is not None else None,
            "last_valid": str(d[col].last_valid_index().date()) if d[col].last_valid_index() is not None else None,
        })
    pd.DataFrame(audit).to_csv(OUT / "source_audit.csv", index=False)

    signal_rows = []
    for date, row in d.iterrows():
        if row["composite"] <= 35 or row["composite"] >= 65:
            signal_rows.append({
                "date": date,
                "composite": row["composite"],
                "threshold_side": "LONG" if row["composite"] <= 35 else "SHORT",
                "regime": row["regime"],
                "bullish_confirmation": row["bullish_confirmation"],
                "bearish_confirmation": row["bearish_confirmation"],
                "bull_ema": row["bull_ema"],
                "bear_ema": row["bear_ema"],
                "bull_5d": row["bull_5d"],
                "bear_5d": row["bear_5d"],
                "spy_ret5": row["spy_ret5"],
            })
    pd.DataFrame(signal_rows).to_csv(OUT / "signal_days.csv", index=False)

    summary = {
        "period": {"start": str(d.index[0].date()), "end": str(d.index[-1].date()), "sessions": len(d)},
        "publishable_sessions": int(d["publishable"].sum()),
        "average_coverage": float(d["coverage"].mean()),
        "minimum_coverage": float(d["coverage"].min()),
        "regime_sessions": d["regime"].value_counts().to_dict(),
        "strategy_comparison": comparison,
        "production_change": False,
        "model_version": "v1.0-PARETO-AI",
        "research_rules_locked": True,
        "notes": [
            "Composite weights, anchors and thresholds remain canonical v1.0.",
            "A and B rules are research-only action mappings.",
            "Signals execute next regular SPY open.",
            "Stops remain 25% at 4%, 25% at 6%, 50% at 8% adverse move.",
            "No fees, slippage, dividends, taxes or borrow costs.",
            "30W regime uses completed Friday-labelled weekly closes without forward information.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
