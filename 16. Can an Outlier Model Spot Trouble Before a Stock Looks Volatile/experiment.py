"""I reproduce the numerical study for this article."""
import io,json,zipfile
import numpy as np
import pandas as pd
import requests
FEE=0.001
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits
STOCKS = ['BA', 'CAT', 'DE', 'HON', 'UNP', 'UPS']
COLS = ['relative_day', 'relative_week', 'volatility_ratio', 'volume_activity', 'close_location']

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

def features(d, sector):
    """I calculate this part of the experiment from the information available."""
    sector = sector.reindex(d.index)
    r = d.close.pct_change()
    s = sector.close.pct_change()
    scale = r.rolling(60).std()
    residual = r - s
    return pd.DataFrame({'relative_day': residual / scale, 'relative_week': residual.rolling(5).sum() / (scale * np.sqrt(5)), 'volatility_ratio': r.rolling(20).std() / s.rolling(20).std(), 'volume_activity': d.volume / d.volume.shift().rolling(20).median(), 'close_location': (d.close - d.low) / (d.high - d.low).replace(0, np.nan) - 0.5, 'volatility': r.rolling(20).std()})

def fit_model(panel):
    """I calculate this part of the experiment from the information available."""
    train = panel.loc[panel.date < '2018-01-01']
    cal = panel.loc[(panel.date >= '2018-01-01') & (panel.date < '2019-01-01')]
    scaler = RobustScaler().fit(train[COLS])
    model = IsolationForest(n_estimators=200, max_samples=512, random_state=20261114, n_jobs=1)
    with threadpool_limits(limits=1):
        model.fit(scaler.transform(train[COLS]))
    score = model.score_samples(scaler.transform(cal[COLS]))
    return (model, scaler, float(np.quantile(score, 0.05)), float(cal.relative_day.abs().quantile(0.95)))

def main():
    """I print my numerical comparison, including the stated fees."""
    sector = load_daily('XLI')
    stocks = {s: load_daily(s) for s in STOCKS}
    parts = []
    for s, d in stocks.items():
        f = features(d, sector).dropna().reset_index(names='date')
        f['stock'] = s
        parts.append(f)
    panel = pd.concat(parts, ignore_index=True)
    model, scaler, cut, basecut = fit_model(panel)
    panel['score'] = model.score_samples(scaler.transform(panel[COLS]))
    panel['warning'] = panel.score < cut
    panel['simple_warning'] = panel.relative_day.abs() > basecut
    rows = []
    for s, d in stocks.items():
        f = panel.loc[(panel.stock == s) & (panel.date >= '2019-01-01')].set_index('date')
        volcut = float(panel.loc[(panel.stock == s) & panel.date.between('2018-01-01', '2018-12-31'), 'volatility'].median())
        for day, r in f.iloc[::5].iterrows():
            i = d.index.get_loc(day)
            if i + 6 >= len(d):
                continue
            outcomes = net_long(d.open.iloc[i + 1], d.open.iloc[i + 2:i + 7])
            row = {**r.to_dict(), 'stock': s, 'date': str(day.date()), 'entry_date': str(d.index[i + 1].date()), 'last_label': str(d.index[i + 6].date()), 'worst_net': float(outcomes.min()), 'terminal_net': float(outcomes.iloc[-1]), 'calm': bool(r.volatility <= volcut)}
            rows.append(row)
    events = pd.DataFrame(rows)
    summary = []
    for scope, q in [('All', events), ('Calm', events.loc[events.calm])]:
        for name, g in [('Ordinary', q.loc[~q.warning & ~q.simple_warning]), ('Isolation warning', q.loc[q.warning]), ('Large relative move', q.loc[q.simple_warning])]:
            summary.append({'scope': scope, 'group': name, 'events': len(g), 'median_worst_net_pct': 100 * g.worst_net.median(), 'mean_worst_net_pct': 100 * g.worst_net.mean(), 'loss_beyond_5_pct': 100 * g.worst_net.lt(-0.05).mean()})
    summary = pd.DataFrame(summary)
    key_results = {'training_rows': int((panel.date < '2018-01-01').sum()), 'calibration_rows': int(panel.date.between('2018-01-01', '2018-12-31').sum()), 'test_origins': len(events), 'isolation_cut': cut, 'simple_cut': basecut, 'warning_pct': 100 * events.warning.mean(), 'simple_warning_pct': 100 * events.simple_warning.mean(), 'fee_per_side': FEE}
    result = summary
    print(result.to_string(index=False))
    print(json.dumps(key_results, indent=2))
    return {'summary': result, 'key_results': key_results}
if __name__=='__main__':main()
