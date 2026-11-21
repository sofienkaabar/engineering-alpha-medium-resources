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
FUNDS = ['SLV', 'USO', 'UNG']

def features(d):
    """I measure wick rejection and define the hammer before inspecting returns."""
    span = (d.high - d.low).replace(0, np.nan)
    body = (d.close - d.open).abs()
    lower = d[['open', 'close']].min(axis=1) - d.low
    upper = d.high - d[['open', 'close']].max(axis=1)
    imbalance = (lower - upper) / span
    pressure = imbalance.rolling(5).mean()
    return pd.DataFrame({'imbalance': imbalance, 'pressure': pressure, 'Pressure': (pressure >= 0.25) & (pressure.shift() < 0.25), 'Single wick': (imbalance >= 0.4) & (imbalance.shift() < 0.4), 'Hammer': (lower >= 2 * body) & (upper <= body) & (body <= 0.3 * span) & (body > 0), 'lower': lower, 'upper': upper})

def portfolio(d, f, method, stock):
    """I trade at the next opening, ignore overlapping entries and liquidate after five intervals."""
    eligible = np.flatnonzero(d.index >= pd.Timestamp('2019-01-01'))
    start = int(eligible[0])
    end = len(d) - 1
    wealth = 1.0
    rows = []
    trades = []
    busy_until = -1
    wealth_series = pd.Series(1.0, index=d.index[start:end + 1])
    trade_no = 0
    fees_paid = 0.0
    exposed = 0
    for i in range(start, end - 5):
        if i + 1 < busy_until or not bool(f[method].iloc[i]):
            continue
        entry_i = i + 1
        exit_i = i + 6
        if exit_i > end:
            continue
        before = wealth
        quantity = before / ((1 + FEE) * d.open.iloc[entry_i])
        fees_paid += quantity * d.open.iloc[entry_i] * FEE
        for j in range(entry_i, exit_i):
            wealth_series.loc[d.index[j]] = quantity * d.open.iloc[j]
        wealth = quantity * d.open.iloc[exit_i] * (1 - FEE)
        fees_paid += quantity * d.open.iloc[exit_i] * FEE
        wealth_series.loc[d.index[exit_i]:] = wealth
        adverse = min(0, net_long(d.open.iloc[entry_i], d.low.iloc[entry_i:exit_i].min()), net_long(d.open.iloc[entry_i], d.open.iloc[exit_i]))
        favorable = max(0, net_long(d.open.iloc[entry_i], d.high.iloc[entry_i:exit_i].max()), net_long(d.open.iloc[entry_i], d.open.iloc[exit_i]))
        trades.append({'stock': stock, 'method': method, 'signal_date': str(d.index[i].date()), 'entry_date': str(d.index[entry_i].date()), 'exit_date': str(d.index[exit_i].date()), 'entry_position': entry_i, 'exit_position': exit_i, 'net': float(wealth / before - 1), 'adverse': float(adverse), 'favorable': float(favorable)})
        busy_until = exit_i
        trade_no += 1
        exposed += 5
    values = wealth_series.to_numpy()
    dd = values / np.maximum.accumulate(np.r_[1.0, values])[1:] - 1
    t = pd.DataFrame(trades)
    row = {'stock': stock, 'method': method, 'trades': trade_no, 'net_return_pct': (wealth - 1) * 100, 'mean_trade_pct': t.net.mean() * 100 if len(t) else 0.0, 'median_trade_pct': t.net.median() * 100 if len(t) else 0.0, 'win_pct': t.net.gt(0).mean() * 100 if len(t) else 0.0, 'max_drawdown_pct': dd.min() * 100, 'exposure_pct': exposed / (end - start) * 100, 'fee_initial_capital_pct': fees_paid * 100, 'start': str(d.index[start].date()), 'end': str(d.index[end].date())}
    return (row, t, pd.DataFrame({'stock': stock, 'method': method, 'date': wealth_series.index.astype(str), 'wealth': values}))

def compute():
    """I compare repeated rejection with single-candle definitions on the same funds."""
    summary = []
    trades = []
    curves = []
    frames = {}
    features_by = {}
    for stock in FUNDS:
        d = load_daily(stock)
        f = features(d)
        frames[stock] = d
        features_by[stock] = f
        for method in ['Pressure', 'Hammer', 'Single wick']:
            row, t, c = portfolio(d, f, method, stock)
            summary.append(row)
            trades.append(t)
            curves.append(c)
        start = d.index.get_indexer([d.index[d.index >= pd.Timestamp('2019-01-01')][0]])[0]
        quantity = 1 / ((1 + FEE) * d.open.iloc[start])
        values = quantity * d.open.iloc[start:].to_numpy()
        values[-1] *= 1 - FEE
        dd = values / np.maximum.accumulate(np.r_[1.0, values])[1:] - 1
        summary.append({'stock': stock, 'method': 'Buy and hold', 'trades': 1, 'net_return_pct': float((values[-1] - 1) * 100), 'mean_trade_pct': float((values[-1] - 1) * 100), 'median_trade_pct': float((values[-1] - 1) * 100), 'win_pct': float(values[-1] > 1) * 100, 'max_drawdown_pct': float(dd.min() * 100), 'exposure_pct': 100.0, 'fee_initial_capital_pct': float(FEE * quantity * (d.open.iloc[start] + d.open.iloc[-1]) * 100), 'start': str(d.index[start].date()), 'end': str(d.index[-1].date())})
        curves.append(pd.DataFrame({'stock': stock, 'method': 'Buy and hold', 'date': d.index[start:].astype(str), 'wealth': values}))
    t = pd.concat(trades, ignore_index=True)
    s = pd.DataFrame(summary)
    candidates = t.loc[t.method.eq('Pressure')].sort_values(['signal_date', 'stock'])
    case = candidates.iloc[0]
    keys = {'pressure_trades': int(t.method.eq('Pressure').sum()), 'hammer_trades': int(t.method.eq('Hammer').sum()), 'single_wick_trades': int(t.method.eq('Single wick').sum()), 'case_stock': case.stock, 'case_date': case.signal_date, 'pressure_mean_trade_pct': float(t.loc[t.method.eq('Pressure'), 'net'].mean() * 100), 'hammer_mean_trade_pct': float(t.loc[t.method.eq('Hammer'), 'net'].mean() * 100), 'fee_per_side': FEE}
    return {'tables': {'summary': s, 'trades': t, 'portfolio': pd.concat(curves, ignore_index=True)}, 'keys': keys, 'frames': frames, 'features': features_by}

if __name__=='__main__':
    result=compute()
    print(result['tables']['summary'].to_string(index=False))
    print(json.dumps(result['keys'],indent=2,allow_nan=False))
