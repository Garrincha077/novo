from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request

DATASET = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
CODE = "13874A"
N = 156

params = {
    "$select": (
        "report_date_as_yyyy_mm_dd,open_interest_all,"
        "asset_mgr_positions_long,asset_mgr_positions_short"
    ),
    "$where": f"cftc_contract_market_code='{CODE}'",
    "$order": "report_date_as_yyyy_mm_dd DESC",
    "$limit": str(N),
}
url = DATASET + "?" + urllib.parse.urlencode(params)
request = urllib.request.Request(
    url,
    headers={"User-Agent": "Contrarian-Greed-audit/1.0"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    rows = json.load(response)

if len(rows) != N:
    raise RuntimeError(f"Expected {N} rows, received {len(rows)}")

observations: list[tuple[str, float, int, int, int]] = []
for row in rows:
    date = row["report_date_as_yyyy_mm_dd"][:10]
    oi = int(row["open_interest_all"])
    long_pos = int(row["asset_mgr_positions_long"])
    short_pos = int(row["asset_mgr_positions_short"])
    if oi <= 0:
        raise RuntimeError(f"Invalid open interest for {date}: {oi}")
    raw = (long_pos - short_pos) / oi
    if not math.isfinite(raw):
        raise RuntimeError(f"Invalid raw value for {date}")
    observations.append((date, raw, oi, long_pos, short_pos))

latest_date, current_raw, current_oi, current_long, current_short = observations[0]
values = [item[1] for item in observations]
count_lt = sum(value < current_raw for value in values)
count_eq = sum(math.isclose(value, current_raw, rel_tol=0.0, abs_tol=1e-15) for value in values)
count_le = sum(value <= current_raw for value in values)

# Canonical empirical percentile: fraction of the 156 observations at or below current.
percentile_ecdf = 100.0 * count_le / N
# Also show strict and average-rank conventions for audit transparency.
percentile_strict = 100.0 * count_lt / N
percentile_average_rank = 100.0 * (count_lt + 0.5 * count_eq) / N

# Excel-style PERCENTRANK.INC, computed by linear interpolation between sorted observations.
def percentrank_inc(data: list[float], x: float) -> float:
    ordered = sorted(data)
    if x <= ordered[0]:
        return 0.0
    if x >= ordered[-1]:
        return 1.0
    for i in range(len(ordered) - 1):
        lo, hi = ordered[i], ordered[i + 1]
        if lo <= x <= hi:
            if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-15):
                return i / (len(ordered) - 1)
            fraction = (x - lo) / (hi - lo)
            return (i + fraction) / (len(ordered) - 1)
    raise RuntimeError("PERCENTRANK interpolation failed")

excel_percentile = 100.0 * percentrank_inc(values, current_raw)

print(f"rows={len(observations)}")
print(f"window_latest={latest_date}")
print(f"window_oldest={observations[-1][0]}")
print(f"open_interest={current_oi}")
print(f"asset_manager_long={current_long}")
print(f"asset_manager_short={current_short}")
print(f"net={current_long-current_short}")
print(f"raw_ratio={current_raw:.12f}")
print(f"raw_percent={100.0*current_raw:.8f}")
print(f"count_lt={count_lt}")
print(f"count_eq={count_eq}")
print(f"count_le={count_le}")
print(f"percentile_ecdf_le={percentile_ecdf:.12f}")
print(f"percentile_strict_lt={percentile_strict:.12f}")
print(f"percentile_average_rank={percentile_average_rank:.12f}")
print(f"percentile_excel_inc={excel_percentile:.12f}")
