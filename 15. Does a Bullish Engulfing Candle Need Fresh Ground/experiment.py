"""I reproduce the numerical study for this article."""
import io,json,zipfile
import numpy as np
import pandas as pd
import requests
FEE=0.001
STOCKS = ['NKE', 'ADBE', 'PYPL']

def load_daily(symbol):
    """I fetch the adjusted daily prices for my fixed study period."""
    response = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}', params={'period1': 1262304000, 'period2': 1794614400, 'interval': '1d', 'events': 'div,splits'}, headers={'User-Agent': 'Mozilla/5.0'}, timeout=90)
    response.raise_for_status()
    raw = response.json()['chart']['result'][0]
    dates = pd.to_datetime(raw['timestamp'], unit='s', utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
    frame = pd.DataFrame(raw['indicators']['quote'][0], index=dates)
    factor = pd.Series(raw['indicators']['adjclose'][0]['adjclose'], index=dates) / frame.close
    for c in ['open', 'high', 'low', 'close']:
        frame[c] *= factor
    frame = frame.loc[:'2026-11-13'].dropna(subset=['open', 'high', 'low', 'close'])
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert (frame[['open', 'high', 'low', 'close']] > 0).all().all()
    assert (frame.high >= frame[['open', 'close', 'low']].max(axis=1) - 1e-07).all()
    assert (frame.low <= frame[['open', 'close', 'high']].min(axis=1) + 1e-07).all()
    return frame

def net_long(entry, exit):
    """I calculate this part of the experiment from the information available."""
    return exit / entry * (1 - FEE) / (1 + FEE) - 1

def merge_intervals(intervals):
    """I merge overlapping prior candle bodies without counting their overlap twice."""
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(high, merged[-1][1])
        else:
            merged.append([low, high])
    return merged

def features(d):
    """I calculate this part of the experiment from the information available."""
    lo = d[['open', 'close']].min(axis=1)
    hi = d[['open', 'close']].max(axis=1)
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
    fresh = pd.Series(np.nan, index=d.index)
    for i in range(5, len(d)):
        size = hi.iloc[i] - lo.iloc[i]
        if size <= 0:
            continue
        used = sum((max(0, min(b, hi.iloc[i]) - max(a, lo.iloc[i])) for a, b in merge_intervals(list(zip(lo.iloc[i - 5:i], hi.iloc[i - 5:i])))))
        fresh.iloc[i] = max(0, min(1, 1 - used / size))
    engulf = (d.close > d.open) & (d.close.shift() < d.open.shift()) & (d.open <= d.close.shift()) & (d.close >= d.open.shift())
    return pd.DataFrame({'fresh': fresh, 'body_atr': (hi - lo) / tr, 'previous_up': d.close.pct_change(5) > 0, 'engulf': engulf})

def event_frame(d, symbol):
    """I calculate this part of the experiment from the information available."""
    f = features(d)
    rows = []
    for i in np.flatnonzero(f.engulf.to_numpy()):
        if i + 11 >= len(d) or d.index[i] < pd.Timestamp('2016-01-01') or pd.isna(f.body_atr.iloc[i]):
            continue
        row = {'stock': symbol, 'date': str(d.index[i].date()), 'position': int(i), 'fresh': float(f.fresh.iloc[i]), 'body_atr': float(f.body_atr.iloc[i]), 'previous_up': bool(f.previous_up.iloc[i]), 'entry': float(d.open.iloc[i + 1]), 'first_execution': str(d.index[i + 1].date()), 'last_outcome': str(d.index[i + 11].date())}
        for h in [1, 3, 5, 10]:
            row[f'net_{h}'] = float(net_long(d.open.iloc[i + 1], d.open.iloc[i + 1 + h]))
        rows.append(row)
    return pd.DataFrame(rows)

def match_events(events):
    """I calculate this part of the experiment from the information available."""
    test = events.query('date >= "2020-01-01"').copy()
    test['bucket'] = pd.cut(test.body_atr, [0, 0.5, 1, 2, 100], labels=False, include_lowest=True)
    high = test.query('fresh >= 0.5').sort_values(['stock', 'date'])
    low = test.query('fresh <= 0.1')
    used = set()
    rows = []
    for i, r in high.iterrows():
        candidates = low.loc[(low.stock == r.stock) & (low.bucket == r.bucket) & (low.previous_up == r.previous_up) & ((low.position - r.position).abs() > 10) & ~low.index.isin(used)].copy()
        if candidates.empty:
            continue
        candidates['distance'] = (candidates.body_atr - r.body_atr).abs()
        candidates = candidates.loc[candidates.distance <= 0.25]
        if candidates.empty:
            continue
        j = candidates.sort_values(['distance', 'date']).index[0]
        q = low.loc[j]
        used.add(j)
        rows.append({'stock': r.stock, 'fresh_date': r.date, 'crowded_date': q.date, 'fresh_net_5': r.net_5, 'crowded_net_5': q.net_5, 'difference': r.net_5 - q.net_5, 'body_distance': abs(r.body_atr - q.body_atr)})
    return pd.DataFrame(rows)

def main():
    """I print my numerical comparison, including the stated fees."""
    data = {s: load_daily(s) for s in STOCKS}
    events = pd.concat([event_frame(d, s) for s, d in data.items()], ignore_index=True)
    test = events.query('date >= "2020-01-01"').copy()
    pairs = match_events(events)
    rows = []
    for stock in ['Pooled'] + STOCKS:
        d = test if stock == 'Pooled' else test.loc[test.stock == stock]
        for name, g in [('All', d), ('Fresh', d.query('fresh >= 0.5')), ('Crowded', d.query('fresh <= 0.1'))]:
            if len(g):
                rows.append({'stock': stock, 'group': name, 'events': len(g), 'mean_net_5_pct': 100 * g.net_5.mean(), 'median_net_5_pct': 100 * g.net_5.median(), 'win_pct': 100 * g.net_5.gt(0).mean(), 'worst_net_5_pct': 100 * g.net_5.min()})
    summary = pd.DataFrame(rows)
    key_results = {'pairs': len(pairs), 'paired_mean_difference_points': 100 * pairs.difference.mean() if len(pairs) else 0.0, 'paired_median_difference_points': 100 * pairs.difference.median() if len(pairs) else 0.0, 'fresh_pair_win_pct': 100 * pairs.difference.gt(0).mean() if len(pairs) else 0.0, 'events': len(test), 'fee_per_side': FEE, 'period_start': test.date.min(), 'period_end': test.date.max()}
    result = summary
    print(result.to_string(index=False))
    print(json.dumps(key_results, indent=2))
    return {'summary': result, 'key_results': key_results}
if __name__=='__main__':main()
