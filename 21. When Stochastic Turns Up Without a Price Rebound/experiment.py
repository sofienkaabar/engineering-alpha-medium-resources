"""I reproduce the numerical experiment for my article."""
import io,json,zipfile
import numpy as np
import pandas as pd
import requests
FEE = 0.001
SEED = 20261121

def load_daily(symbol):
    """I fetch adjusted OHLC for my fixed research period."""
    response = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}', params={'period1': 1104537600, 'period2': 1795132800, 'interval': '1d', 'events': 'div,splits'}, headers={'User-Agent': 'Mozilla/5.0'}, timeout=90)
    response.raise_for_status()
    raw = response.json()['chart']['result'][0]
    dates = pd.to_datetime(raw['timestamp'], unit='s', utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
    frame = pd.DataFrame(raw['indicators']['quote'][0], index=dates)
    ratio = pd.Series(raw['indicators']['adjclose'][0]['adjclose'], index=dates) / frame.close
    for c in ['open', 'high', 'low', 'close']:
        frame[c] *= ratio
    frame = frame.loc[:'2026-11-19'].dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert (frame[['open', 'high', 'low', 'close']] > 0).all().all()
    assert (frame.high >= frame[['open', 'close', 'low']].max(axis=1) - 1e-06).all()
    assert (frame.low <= frame[['open', 'close', 'high']].min(axis=1) + 1e-06).all()
    return frame

def net_long(entry, exit):
    """I pay the purchase and sale charges before calling a return profitable."""
    return exit / entry * (1 - FEE) / (1 + FEE) - 1

def weekly_positions(d):
    """I decide after the last session of complete Friday-ending weeks."""
    p = pd.Series(np.arange(len(d)), index=d.index).groupby(d.index.to_period('W-FRI')).last()
    return [int(i) for period, i in p.items() if period.end_time.normalize() <= d.index[-1]]
STOCKS = ['DIS', 'INTC', 'PFE']

def features(d):
    """I attribute the change symmetrically to price and rolling boundaries."""
    low = d.low.rolling(14).min()
    high = d.high.rolling(14).max()

    def k(c, l, h):
        return 100 * (c - l) / (h - l).replace(0, np.nan)
    before = k(d.close.shift(), low.shift(), high.shift())
    after = k(d.close, low, high)
    price = 0.5 * (k(d.close, low.shift(), high.shift()) - before + (after - k(d.close.shift(), low, high)))
    boundary = after - before - price
    return pd.DataFrame({'k': after, 'price': price, 'boundary': boundary, 'cross': (before < 20) & (after >= 20)})

def event_frame(d, stock):
    """I use only complete next-open outcome windows."""
    f = features(d)
    rows = []
    for i in range(20, len(d) - 6):
        if d.index[i] < pd.Timestamp('2021-01-01'):
            continue
        entry = d.open.iloc[i + 1]
        exit = d.open.iloc[i + 6]
        adverse = min(0, net_long(entry, d.low.iloc[i + 1:i + 6].min()), net_long(entry, exit))
        rows.append({'stock': stock, 'date': str(d.index[i].date()), 'position': i, 'entry_date': str(d.index[i + 1].date()), 'exit_date': str(d.index[i + 6].date()), 'cross': bool(f.cross.iloc[i]), 'k': float(f.k.iloc[i]), 'price': float(f.price.iloc[i]), 'boundary': float(f.boundary.iloc[i]), 'price_up': bool(d.close.iloc[i] > d.close.iloc[i - 1]), 'group': 'Price-led' if f.price.iloc[i] > f.boundary.iloc[i] else 'Range-led', 'net': float(net_long(entry, exit)), 'adverse': float(adverse)})
    return pd.DataFrame(rows)

def compute():
    """I compare conditional outcomes rather than invent a compounded event portfolio."""
    frames = {s: load_daily(s) for s in STOCKS}
    all_days = pd.concat([event_frame(d, s) for s, d in frames.items()], ignore_index=True)
    events = all_days.loc[all_days.cross].copy()
    rows = []
    for stock in STOCKS + ['Pooled']:
        base = all_days if stock == 'Pooled' else all_days.loc[all_days.stock.eq(stock)]
        ev = base.loc[base.cross]
        for label, g in [('All days', base), ('All crossings', ev), ('Price-led', ev.loc[ev.group.eq('Price-led')]), ('Range-led', ev.loc[ev.group.eq('Range-led')])]:
            rows.append({'stock': stock, 'group': label, 'count': len(g), 'mean_pct': g.net.mean() * 100, 'median_pct': g.net.median() * 100, 'win_pct': g.net.gt(0).mean() * 100, 'mean_adverse_pct': g.adverse.mean() * 100})
    quarters = pd.to_datetime(events.date).dt.to_period('Q')
    events['quarter'] = quarters.astype(str)
    q = events.groupby(['quarter', 'group']).net.agg(['sum', 'count']).unstack(fill_value=0)
    sums = q['sum'].reindex(columns=['Price-led', 'Range-led'], fill_value=0).to_numpy()
    counts = q['count'].reindex(columns=['Price-led', 'Range-led'], fill_value=0).to_numpy()
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(1000):
        ix = rng.integers(0, len(q), len(q))
        ss = sums[ix].sum(axis=0)
        nn = counts[ix].sum(axis=0)
        if (nn > 0).all():
            boots.append((ss[0] / nn[0] - ss[1] / nn[1]) * 100)
    groups = events.groupby('group').net.mean()
    candidates = events.loc[events.group.eq('Range-led') & ~events.price_up].sort_values(['date', 'stock'])
    if candidates.empty:
        candidates = events.loc[events.group.eq('Range-led')].sort_values(['date', 'stock'])
    case = candidates.iloc[0]
    keys = {'crossings': len(events), 'range_led': int(events.group.eq('Range-led').sum()), 'no_price_rebound': int((~events.price_up).sum()), 'mean_difference_pp': float((groups['Price-led'] - groups['Range-led']) * 100), 'bootstrap_low_pp': float(np.quantile(boots, 0.025)), 'bootstrap_high_pp': float(np.quantile(boots, 0.975)), 'case_stock': case.stock, 'case_date': case.date, 'case_price_contribution': float(case.price), 'case_boundary_contribution': float(case.boundary), 'case_net_pct': float(case.net * 100), 'first_signal': events.date.min(), 'last_signal': events.date.max(), 'fee_per_side': FEE}
    return {'tables': {'summary': pd.DataFrame(rows), 'events': events, 'all_days': all_days}, 'keys': keys, 'frames': frames}

if __name__=='__main__':
    result=compute()
    print(result['tables']['summary'].to_string(index=False))
    print(json.dumps(result['keys'],indent=2,allow_nan=False))
