from __future__ import annotations

import io
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from pandas.tseries.offsets import BDay

OUT = Path('output')
OUT.mkdir(exist_ok=True)
START = pd.Timestamp('2024-07-29')
END = pd.Timestamp('2026-07-27')
WARMUP = pd.Timestamp('2023-06-01')
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36'
HEADERS = {'User-Agent': UA, 'Accept': 'text/html,application/xhtml+xml,application/json,*/*'}

SOURCES = {
    'prices': 'Yahoo Finance via yfinance (unadjusted daily OHLC)',
    'vix': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv',
    'cboe_pc': 'https://www.cboe.com/us/options/market_statistics/daily/?dt=YYYY-MM-DD',
    'naaim': 'https://naaim.org/wp-content/uploads/2026/07/USE_Data-since-Inception_2026-07-22.xlsx',
    'aaii': 'https://www.aaii.com/files/surveys/sentiment.xls',
    'cnn': 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2024-01-01',
    'hy_oas': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2',
}


def get(url: str, *, timeout: int = 45, retries: int = 4) -> requests.Response:
    err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f'GET failed: {url}: {err}')


def load_prices() -> pd.DataFrame:
    symbols = ['SPY', 'RSP', 'IWM', 'HYG', 'IEF', 'UUP']
    raw = yf.download(symbols, start=WARMUP.strftime('%Y-%m-%d'), end=(END + pd.Timedelta(days=2)).strftime('%Y-%m-%d'),
                      auto_adjust=False, progress=False, group_by='column', threads=True)
    if raw.empty:
        raise RuntimeError('yfinance returned no price data')
    frames = []
    for sym in symbols:
        d = pd.DataFrame(index=raw.index)
        if isinstance(raw.columns, pd.MultiIndex):
            for field in ['Open','High','Low','Close']:
                d[f'{sym}_{field.lower()}'] = raw[(field, sym)]
        else:
            for field in ['Open','High','Low','Close']:
                d[f'{sym}_{field.lower()}'] = raw[field]
        frames.append(d)
    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out = out.sort_index().dropna(subset=['SPY_close'])
    return out


def load_vix() -> pd.Series:
    r = get(SOURCES['vix'])
    df = pd.read_csv(io.BytesIO(r.content))
    date_col = next(c for c in df.columns if c.strip().upper() == 'DATE')
    close_col = next(c for c in df.columns if c.strip().upper() == 'CLOSE')
    s = pd.Series(pd.to_numeric(df[close_col], errors='coerce').values,
                  index=pd.to_datetime(df[date_col], errors='coerce').dt.normalize(), name='vix')
    return s[~s.index.duplicated(keep='last')].sort_index()


def parse_cboe_html(text: str) -> tuple[float | None, float | None]:
    flat = BeautifulSoup(text, 'lxml').get_text(' ', strip=True)
    def grab(label: str):
        pats = [
            rf'{re.escape(label)}\s*[:|]?\s*([0-9]+(?:\.[0-9]+)?)',
            rf'"name"\s*:\s*"{re.escape(label)}".*?"value"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)',
            rf'{re.escape(label)}.*?([0-9]+(?:\.[0-9]+)?)',
        ]
        for source in (flat, text):
            for pat in pats:
                m = re.search(pat, source, re.I | re.S)
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        pass
        return None
    return grab('TOTAL PUT/CALL RATIO'), grab('EQUITY PUT/CALL RATIO')


def fetch_cboe_one(date: pd.Timestamp) -> tuple[pd.Timestamp, float | None, float | None, str]:
    urls = [
        f'https://www.cboe.com/us/options/market_statistics/daily/?dt={date:%Y-%m-%d}',
        f'https://www.cboe.com/markets/us/options/market-statistics/daily?dt={date:%Y-%m-%d}',
        f'https://www.cboe.com/data/mktstat.aspx?date={date:%Y-%m-%d}',
    ]
    last = ''
    for url in urls:
        try:
            r = get(url, timeout=30, retries=3)
            total, equity = parse_cboe_html(r.text)
            if total is not None and equity is not None:
                return date, total, equity, url
            last = f'no ratios len={len(r.text)}'
        except Exception as e:
            last = str(e)
    return date, None, None, last


def load_cboe_pc(trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    cache = OUT / 'cboe_pc.csv'
    rows = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(fetch_cboe_one, d): d for d in trading_dates}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 50 == 0:
                print(f'CBOE {i}/{len(futs)}')
    df = pd.DataFrame(rows, columns=['date','total_pc','equity_pc','source_or_error']).sort_values('date')
    df.to_csv(cache, index=False)
    ok = df[['total_pc','equity_pc']].notna().all(axis=1).mean()
    print('CBOE success rate', ok)
    return df.set_index('date')


def extract_date_value_workbook(content: bytes, kind: str) -> pd.DataFrame:
    engine = 'openpyxl' if content[:2] == b'PK' else 'xlrd'
    xls = pd.ExcelFile(io.BytesIO(content), engine=engine)
    best = None
    best_n = -1
    for sh in xls.sheet_names:
        try:
            raw = pd.read_excel(io.BytesIO(content), sheet_name=sh, header=None, engine=engine)
        except Exception:
            continue
        for dc in range(min(4, raw.shape[1])):
            dates = pd.to_datetime(raw.iloc[:, dc], errors='coerce')
            for vc in range(min(8, raw.shape[1])):
                if vc == dc:
                    continue
                vals = pd.to_numeric(raw.iloc[:, vc], errors='coerce')
                valid = dates.notna() & vals.notna() & (dates.dt.year >= 1980) & (dates.dt.year <= 2030)
                n = int(valid.sum())
                if n > best_n:
                    best_n = n
                    best = pd.DataFrame({'date': dates[valid].dt.normalize(), 'value': vals[valid]}).drop_duplicates('date', keep='last')
    if best is None or best_n < 20:
        raise RuntimeError(f'Could not parse {kind} workbook; best_n={best_n}, sheets={xls.sheet_names}')
    print(kind, 'parsed rows', best_n, 'range', best.date.min(), best.date.max())
    return best.sort_values('date')


def load_naaim() -> pd.DataFrame:
    r = get(SOURCES['naaim'], timeout=90)
    (OUT / 'naaim_source.xlsx').write_bytes(r.content)
    df = extract_date_value_workbook(r.content, 'NAAIM')
    df = df.rename(columns={'date':'survey_date','value':'naaim'})
    # Survey is taken Wednesday at close and posted Thursday; value is usable for Thursday target session.
    df['effective_date'] = df['survey_date'] + pd.Timedelta(days=1)
    return df


def load_aaii() -> pd.DataFrame:
    r = get(SOURCES['aaii'], timeout=90)
    (OUT / 'aaii_source.xls').write_bytes(r.content)
    x = pd.read_excel(io.BytesIO(r.content), sheet_name='SENTIMENT', skiprows=3, engine='xlrd')
    x.columns = [str(c).strip() for c in x.columns]
    date_col = x.columns[0]
    bullish_col = next((c for c in x.columns if 'Bullish' in c and 'Average' not in c and 'Mov' not in c), x.columns[1])
    bearish_col = next((c for c in x.columns if 'Bearish' in c and 'Average' not in c), x.columns[3])
    df = pd.DataFrame({
        'effective_date': pd.to_datetime(x[date_col], errors='coerce').dt.normalize(),
        'bullish': pd.to_numeric(x[bullish_col], errors='coerce'),
        'bearish': pd.to_numeric(x[bearish_col], errors='coerce'),
    }).dropna()
    # Some historical files store proportions, some percentages.
    if df['bullish'].median() < 1.5:
        df[['bullish','bearish']] *= 100
    df['aaii_spread'] = df['bullish'] - df['bearish']
    df = df.drop_duplicates('effective_date', keep='last').sort_values('effective_date')
    print('AAII rows', len(df), 'range', df.effective_date.min(), df.effective_date.max())
    return df


def load_cnn() -> pd.Series:
    r = get(SOURCES['cnn'], timeout=90)
    js = r.json()
    arr = js['fear_and_greed_historical']['data']
    df = pd.DataFrame(arr)
    idx = pd.to_datetime(df['x'], unit='ms', utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    s = pd.Series(pd.to_numeric(df['y'], errors='coerce').values, index=idx, name='cnn_fg')
    s = s[~s.index.duplicated(keep='last')].sort_index()
    pd.DataFrame({'date':s.index,'cnn_fg':s.values}).to_csv(OUT/'cnn_history.csv', index=False)
    return s


def load_hy_oas() -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2&cosd={WARMUP:%Y-%m-%d}&coed={END:%Y-%m-%d}"
    r = get(url, timeout=90)
    df = pd.read_csv(io.BytesIO(r.content))
    df.columns = ['observation_date','hy_oas']
    df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce').dt.normalize()
    df['hy_oas'] = pd.to_numeric(df['hy_oas'], errors='coerce')
    df = df.dropna().sort_values('observation_date')
    # Conservative point-in-time assumption: observation becomes usable one U.S. business day later.
    df['effective_date'] = df['observation_date'] + BDay(1)
    return df


def asof_value(base: pd.DataFrame, releases: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    left = base.reset_index().rename(columns={'index':'date'}).sort_values('date')
    right = releases[['effective_date'] + value_cols].sort_values('effective_date')
    out = pd.merge_asof(left, right, left_on='date', right_on='effective_date', direction='backward')
    return out.set_index('date')


def piecewise(x: pd.Series, anchors: list[tuple[float,float]]) -> pd.Series:
    xp = np.array([a[0] for a in anchors], dtype=float)
    fp = np.array([a[1] for a in anchors], dtype=float)
    vals = pd.to_numeric(x, errors='coerce').to_numpy(dtype=float)
    res = np.interp(vals, xp, fp, left=fp[0], right=fp[-1])
    res[np.isnan(vals)] = np.nan
    return pd.Series(res, index=x.index)


def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100/(1+rs)
    return rsi.where(avg_loss != 0, 100)


def build_daily() -> pd.DataFrame:
    prices = load_prices()
    test_dates = prices.loc[START:END].index
    # Include 15 prior sessions so 10D P/C is available from first test date.
    pc_dates = prices.loc[:END].index
    pc_dates = pc_dates[pc_dates >= START - pd.Timedelta(days=35)]
    vix = load_vix()
    cboe = load_cboe_pc(pc_dates)
    naaim = load_naaim()
    aaii = load_aaii()
    cnn = load_cnn()
    hy = load_hy_oas()

    d = prices.copy()
    d['vix'] = vix.reindex(d.index)
    d['equity_pc'] = cboe['equity_pc'].reindex(d.index)
    d['total_pc'] = cboe['total_pc'].reindex(d.index)
    d['total_pc_10d'] = d['total_pc'].rolling(10, min_periods=10).mean()
    d['cnn_fg'] = cnn.reindex(d.index)
    d = asof_value(d, naaim, ['naaim'])
    d = asof_value(d, aaii, ['aaii_spread','bullish','bearish'])
    d = asof_value(d, hy, ['hy_oas'])

    d['rsp_spy_20d'] = d['RSP_close'].pct_change(20) - d['SPY_close'].pct_change(20)
    d['iwm_spy_20d'] = d['IWM_close'].pct_change(20) - d['SPY_close'].pct_change(20)
    d['hyg_ief_20d'] = d['HYG_close'].pct_change(20) - d['IEF_close'].pct_change(20)
    d['spy_rsi14'] = wilder_rsi(d['SPY_close'], 14)
    d['uup_20d'] = d['UUP_close'].pct_change(20)
    d['spy_ema10'] = d['SPY_close'].ewm(span=10, adjust=False).mean()
    d['spy_ema20'] = d['SPY_close'].ewm(span=20, adjust=False).mean()
    d['spy_ret5'] = d['SPY_close'].pct_change(5)

    scores = {
        'vix_score': piecewise(d['vix'], [(12,100),(15,80),(20,50),(30,20),(40,0)]),
        'equity_pc_score': piecewise(d['equity_pc'], [(0.45,100),(0.55,80),(0.70,50),(0.90,20),(1.10,0)]),
        'total_pc_10d_score': piecewise(d['total_pc_10d'], [(0.70,100),(0.80,80),(0.95,50),(1.10,20),(1.30,0)]),
        'naaim_score': piecewise(d['naaim'], [(-20,0),(0,20),(50,50),(100,80),(150,100)]),
        'aaii_score': piecewise(d['aaii_spread'], [(-40,0),(-20,20),(0,50),(20,80),(40,100)]),
        'rsp_spy_score': piecewise(d['rsp_spy_20d'], [(-.08,0),(-.04,20),(0,50),(.04,80),(.08,100)]),
        'iwm_spy_score': piecewise(d['iwm_spy_20d'], [(-.08,0),(-.04,20),(0,50),(.04,80),(.08,100)]),
        'cnn_score': d['cnn_fg'],
        'hy_oas_score': piecewise(d['hy_oas'], [(2.5,100),(3.0,80),(4.0,50),(5.5,20),(8.0,0)]),
        'hyg_ief_score': piecewise(d['hyg_ief_20d'], [(-.04,0),(-.02,20),(0,50),(.02,80),(.04,100)]),
        'rsi_score': d['spy_rsi14'],
        'uup_score': piecewise(d['uup_20d'], [(-.03,0),(-.015,20),(0,50),(.015,80),(.03,100)]),
    }
    for k,v in scores.items(): d[k] = v
    weights = {
        'vix_score':.15,'equity_pc_score':.09,'total_pc_10d_score':.06,'naaim_score':.12,
        'aaii_score':.08,'rsp_spy_score':.075,'iwm_spy_score':.075,'cnn_score':.10,
        'hy_oas_score':.06,'hyg_ief_score':.04,'rsi_score':.10,'uup_score':.05,
    }
    valid_weight = pd.Series(0.0, index=d.index)
    composite = pd.Series(0.0, index=d.index)
    for col,w in weights.items():
        valid = d[col].notna()
        valid_weight += valid.astype(float)*w
        composite += d[col].fillna(50)*w
        d[col.replace('_score','_contribution')] = d[col].fillna(50)*w
    d['coverage'] = valid_weight*100
    d['mandatory_ok'] = d['vix_score'].notna() & d['rsi_score'].notna() & (d[['equity_pc_score','total_pc_10d_score']].notna().any(axis=1)) & (d[['rsp_spy_score','iwm_spy_score']].notna().any(axis=1))
    d['publishable'] = (d['coverage'] >= 80) & d['mandatory_ok']
    d['composite'] = composite.where(d['publishable'])
    d['zone'] = pd.cut(d['composite'], [-np.inf,20,35,44,55,64,79,89,np.inf], labels=['Extreme Fear','Fear','Mild Fear','Neutral','Mild Greed','Greed','Extreme Greed','Euphoria'])
    d['bullish_confirmation'] = ((d['SPY_close'] > d['spy_ema10']) & (d['spy_ema10'] > d['spy_ema20'])) | (d['spy_ret5'] >= .02)
    d['bearish_confirmation'] = ((d['SPY_close'] < d['spy_ema10']) & (d['spy_ema10'] < d['spy_ema20'])) | (d['spy_ret5'] <= -.02)
    d['instruction'] = np.select([d['composite'] <= 35, d['composite'] >= 65], ['LONG','SHORT'], default='HOLD')
    return d.loc[START:END].copy()


@dataclass
class Position:
    side: int = 0  # +1 long, -1 short
    qty: float = 0.0
    entry: float = np.nan
    original_qty: float = 0.0
    stop4_done: bool = False
    stop6_done: bool = False


def paper_backtest(d: pd.DataFrame, initial: float = 10000.0):
    cash = initial
    pos = Position()
    pending = None
    long_lock = False; long_cleared = False
    short_lock = False; short_cleared = False
    trades = []
    daily = []

    def equity_at(px):
        return cash + pos.qty * px

    def close_qty(qty_to_close, px, date, reason):
        nonlocal cash, pos
        signed_close = math.copysign(qty_to_close, pos.qty)
        # Sell long (cash +), cover short (cash -).
        cash += signed_close * px
        trades.append({'date':date,'event':'EXIT','side':'LONG' if pos.side==1 else 'SHORT','qty':qty_to_close,'price':px,'reason':reason,'cash_after':cash})
        pos.qty -= signed_close
        if abs(pos.qty) < 1e-9:
            pos = Position()

    def open_side(side, px, date, reason):
        nonlocal cash, pos
        eq = cash
        qty_abs = eq / px
        signed = side * qty_abs
        cash -= signed * px
        pos = Position(side=side, qty=signed, entry=px, original_qty=qty_abs)
        trades.append({'date':date,'event':'ENTRY','side':'LONG' if side==1 else 'SHORT','qty':qty_abs,'price':px,'reason':reason,'cash_after':cash})

    for i,(date,row) in enumerate(d.iterrows()):
        op, hi, lo, cl = row['SPY_open'], row['SPY_high'], row['SPY_low'], row['SPY_close']
        # Execute prior close instruction at today's open.
        if pending in ('LONG','SHORT'):
            desired = 1 if pending=='LONG' else -1
            blocked = (desired==1 and long_lock and not long_cleared) or (desired==-1 and short_lock and not short_cleared)
            if not blocked and pos.side != desired:
                if pos.side != 0:
                    close_qty(abs(pos.qty), op, date, 'OPPOSITE_SIGNAL')
                open_side(desired, op, date, 'THRESHOLD_SIGNAL')
                if desired==1:
                    long_lock=False; long_cleared=False
                else:
                    short_lock=False; short_cleared=False
        pending = None

        # Intraday stop ladder, based on original entry and quantity.
        fully_stopped = False
        if pos.side == 1:
            levels = [(0.04,0.25,'STOP_4'),(0.06,0.25,'STOP_6'),(0.08,0.50,'STOP_8')]
            for adverse, frac, reason in levels:
                already = pos.stop4_done if adverse==.04 else pos.stop6_done if adverse==.06 else False
                if already or pos.side==0: continue
                level = pos.entry*(1-adverse)
                if lo <= level:
                    fill = op if op < level else level
                    q = min(pos.original_qty*frac, abs(pos.qty))
                    if adverse==.04: pos.stop4_done=True
                    if adverse==.06: pos.stop6_done=True
                    close_qty(q, fill, date, reason)
                    if adverse==.08 or pos.side==0: fully_stopped=True
        elif pos.side == -1:
            levels = [(0.04,0.25,'STOP_4'),(0.06,0.25,'STOP_6'),(0.08,0.50,'STOP_8')]
            for adverse, frac, reason in levels:
                already = pos.stop4_done if adverse==.04 else pos.stop6_done if adverse==.06 else False
                if already or pos.side==0: continue
                level = pos.entry*(1+adverse)
                if hi >= level:
                    fill = op if op > level else level
                    q = min(pos.original_qty*frac, abs(pos.qty))
                    if adverse==.04: pos.stop4_done=True
                    if adverse==.06: pos.stop6_done=True
                    close_qty(q, fill, date, reason)
                    if adverse==.08 or pos.side==0: fully_stopped=True
        if fully_stopped:
            # Determine direction from most recent entry event.
            last_side = trades[-1]['side'] if trades else ''
            if last_side=='LONG': long_lock=True; long_cleared=False
            elif last_side=='SHORT': short_lock=True; short_cleared=False

        comp = row['composite']
        if pd.notna(comp):
            if long_lock and comp > 35: long_cleared=True
            if short_lock and comp < 65: short_cleared=True
            if comp <= 35 and not (long_lock and not long_cleared): pending='LONG'
            elif comp >= 65 and not (short_lock and not short_cleared): pending='SHORT'

        eq = equity_at(cl)
        daily.append({'date':date,'paper_equity':eq,'paper_side':'LONG' if pos.side==1 else 'SHORT' if pos.side==-1 else 'FLAT','paper_qty':abs(pos.qty),'cash':cash})

    # Mark-to-market final; no forced exit.
    return pd.DataFrame(daily).set_index('date'), pd.DataFrame(trades)


def forward_stats(d: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for h in [1,3,5,10,20,60]:
        fwd=d['SPY_close'].shift(-h)/d['SPY_close']-1
        groups={
            '<=35':d['composite']<=35,
            '36-44':d['composite'].between(36,44,inclusive='both'),
            '45-55':d['composite'].between(45,55,inclusive='both'),
            '56-64':d['composite'].between(56,64,inclusive='both'),
            '>=65':d['composite']>=65,
            'ALL':d['composite'].notna(),
        }
        for g,m in groups.items():
            x=fwd[m].dropna()
            rows.append({'horizon_sessions':h,'composite_bucket':g,'n':len(x),'mean_return':x.mean(),'median_return':x.median(),'positive_rate':(x>0).mean(),'min_return':x.min(),'max_return':x.max()})
    return pd.DataFrame(rows)


def crossing_events(d: pd.DataFrame) -> pd.DataFrame:
    prev=d['composite'].shift(1)
    long_cross=(d['composite']<=35)&(prev>35)
    short_cross=(d['composite']>=65)&(prev<65)
    rows=[]
    for label,mask in [('LONG_CROSS',long_cross),('SHORT_CROSS',short_cross)]:
        for date in d.index[mask.fillna(False)]:
            i=d.index.get_loc(date)
            rec={'date':date,'event':label,'composite':d.at[date,'composite'],'spy_close':d.at[date,'SPY_close']}
            for h in [1,3,5,10,20,60]:
                rec[f'ret_{h}d']=d['SPY_close'].iloc[i+h]/d['SPY_close'].iloc[i]-1 if i+h<len(d) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    d=build_daily()
    paper,trades=paper_backtest(d)
    d=d.join(paper)
    first_open=d['SPY_open'].iloc[0]
    bh_qty=10000/first_open
    d['buyhold_equity']=bh_qty*d['SPY_close']
    d['spy_norm']=100*d['SPY_close']/d['SPY_close'].iloc[0]
    d['paper_norm']=100*d['paper_equity']/10000
    stats=forward_stats(d)
    crosses=crossing_events(d)

    paper_ret=d['paper_equity'].iloc[-1]/10000-1
    bh_ret=d['buyhold_equity'].iloc[-1]/10000-1
    paper_dd=(d['paper_equity']/d['paper_equity'].cummax()-1).min()
    bh_dd=(d['buyhold_equity']/d['buyhold_equity'].cummax()-1).min()
    daily_ret=d['paper_equity'].pct_change().dropna()
    bh_daily=d['buyhold_equity'].pct_change().dropna()
    summary={
        'start':str(d.index.min().date()),'end':str(d.index.max().date()),'sessions':len(d),
        'publishable_sessions':int(d['publishable'].sum()),'avg_coverage':float(d['coverage'].mean()),
        'min_coverage':float(d['coverage'].min()),'long_days':int((d['composite']<=35).sum()),
        'short_days':int((d['composite']>=65).sum()),'long_crossings':int((crosses.event=='LONG_CROSS').sum()) if len(crosses) else 0,
        'short_crossings':int((crosses.event=='SHORT_CROSS').sum()) if len(crosses) else 0,
        'paper_final_equity':float(d['paper_equity'].iloc[-1]),'paper_return':float(paper_ret),'paper_max_drawdown':float(paper_dd),
        'paper_sharpe_annualized':float(np.sqrt(252)*daily_ret.mean()/daily_ret.std()) if daily_ret.std()>0 else None,
        'buyhold_final_equity':float(d['buyhold_equity'].iloc[-1]),'buyhold_return':float(bh_ret),'buyhold_max_drawdown':float(bh_dd),
        'buyhold_sharpe_annualized':float(np.sqrt(252)*bh_daily.mean()/bh_daily.std()) if bh_daily.std()>0 else None,
        'trade_events':int(len(trades)),'entry_events':int((trades.event=='ENTRY').sum()) if len(trades) else 0,
        'methodology_notes':[
            'NAAIM survey Wednesday effective Thursday (official posting convention).',
            'AAII workbook date used as effective date.',
            'HY OAS observation shifted one U.S. business day to avoid look-ahead.',
            'CNN values use CNN historical endpoint exact session date.',
            'Signals execute next regular SPY open; no fees or slippage.',
            'Fractional shares; 100% notional; no leverage.',
        ],
        'sources':SOURCES,
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    d.reset_index(names='date').to_csv(OUT/'backtest_daily.csv',index=False)
    stats.to_csv(OUT/'forward_return_stats.csv',index=False)
    crosses.to_csv(OUT/'threshold_crossings.csv',index=False)
    trades.to_csv(OUT/'paper_trade_events.csv',index=False)

    # Source audit
    audit=[]
    raw_cols=['vix','equity_pc','total_pc_10d','naaim','aaii_spread','rsp_spy_20d','iwm_spy_20d','cnn_fg','hy_oas','hyg_ief_20d','spy_rsi14','uup_20d']
    for c in raw_cols:
        audit.append({'leaf':c,'valid_sessions':int(d[c].notna().sum()),'missing_sessions':int(d[c].isna().sum()),'coverage_pct':float(d[c].notna().mean()*100),'first_valid':str(d[c].first_valid_index().date()) if d[c].first_valid_index() else None,'last_valid':str(d[c].last_valid_index().date()) if d[c].last_valid_index() else None})
    pd.DataFrame(audit).to_csv(OUT/'source_audit.csv',index=False)

    fig,ax1=plt.subplots(figsize=(14,7))
    ax1.plot(d.index,d['SPY_close'],label='SPY close')
    ax1.set_ylabel('SPY price')
    ax2=ax1.twinx()
    ax2.plot(d.index,d['composite'],label='Composite',alpha=.8)
    ax2.axhline(35,linestyle='--'); ax2.axhline(65,linestyle='--')
    ax2.set_ylabel('Composite 0-100')
    ax1.set_title('Contrarian Greed Composite vs SPY')
    fig.tight_layout(); fig.savefig(OUT/'composite_vs_spy.png',dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(14,7))
    ax.plot(d.index,d['paper_equity'],label='v1.0-paper')
    ax.plot(d.index,d['buyhold_equity'],label='SPY buy & hold')
    ax.legend(); ax.set_title('Equity curve: v1.0-paper vs SPY buy & hold'); ax.set_ylabel('Account value ($)')
    fig.tight_layout(); fig.savefig(OUT/'equity_curves.png',dpi=180); plt.close(fig)

    print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__':
    main()
