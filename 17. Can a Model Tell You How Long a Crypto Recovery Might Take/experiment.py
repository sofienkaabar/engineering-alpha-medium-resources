"""I reproduce the numerical study for this article."""
import io,json,zipfile
import numpy as np
import pandas as pd
import requests
FEE=0.001
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
COINS = ['ETHUSDT', 'SOLUSDT']
COLS = ['depth', 'return_6', 'volatility', 'red_share', 'volume_activity']

def archive(name, url):
    """I read the public archive in memory without saving a dataset or report."""
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(response.content))
    return (z, pd.read_csv(z.open(z.namelist()[0]), header=None))

def load_hourly(symbol):
    """I respect the archive's change from millisecond to microsecond timestamps."""
    parts = []
    for year in range(2022, 2027):
        for month in range(1, 13):
            if year == 2026 and month > 10:
                continue
            name = f'{symbol}-1h-{year}-{month:02d}.zip'
            z, d = archive(name, f'https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/{name}')
            d = d.iloc[:, :6]
            d.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            unit = 'us' if d.time.median() > 100000000000000.0 else 'ms'
            d.index = pd.to_datetime(d.pop('time'), unit=unit, utc=True)
            z.close()
            parts.append(d)
    data = pd.concat(parts).sort_index()
    assert data.index.is_unique and data.index.is_monotonic_increasing
    assert (data[['open', 'high', 'low', 'close']] > 0).all().all()
    assert (data.high >= data[['open', 'close', 'low']].max(axis=1) - 1e-08).all()
    assert (data.low <= data[['open', 'close', 'high']].min(axis=1) + 1e-08).all()
    assert (data.index.to_series().diff().dropna() >= pd.Timedelta(hours=1)).all()
    return data

def features(d):
    """I calculate this part of the experiment from the information available."""
    peak = d.close.shift().rolling(72).max()
    r = d.close.pct_change()
    result = pd.DataFrame({'peak': peak, 'depth': 1 - d.close / peak, 'return_6': d.close.pct_change(6), 'volatility': r.rolling(24).std(), 'red_share': d.close.diff().lt(0).rolling(12).mean(), 'volume_activity': d.volume / d.volume.shift().rolling(24).median()})
    complete = d.index.to_series().diff(72).eq(pd.Timedelta(hours=72))
    return result.where(complete, axis=0)

def episodes(d, symbol):
    """I calculate this part of the experiment from the information available."""
    f = features(d)
    trigger = f.depth.ge(0.04) & f.depth.shift().lt(0.04)
    rows = []
    blocked = -1
    for i in np.flatnonzero(trigger.to_numpy()):
        if i <= blocked or i + 49 >= len(d) or f.iloc[i][COLS].isna().any():
            continue
        if d.index[i + 49] - d.index[i] != pd.Timedelta(hours=49):
            continue
        barrier = float(f.peak.iloc[i] * (1 + FEE) / (1 - FEE))
        future = d.open.iloc[i + 2:i + 50].to_numpy()
        hit = np.flatnonzero(future >= barrier)
        time = int(hit[0] + 1) if len(hit) else 48
        observed = bool(len(hit))
        blocked = i + time + 1
        rows.append({'coin': symbol, 'origin': str(d.index[i] + pd.Timedelta(hours=1)), 'complete_by': str(d.index[i + 49]), 'time': time, 'observed': observed, 'barrier': barrier, **{c: float(f.iloc[i][c]) for c in COLS}})
    return pd.DataFrame(rows)

def person_period(events):
    """I calculate this part of the experiment from the information available."""
    xs = []
    ys = []
    for row in events.itertuples():
        for age in range(1, row.time + 1):
            xs.append([age] + [getattr(row, c) for c in COLS])
            ys.append(int(row.observed and age == row.time))
    return (np.asarray(xs), np.asarray(ys))

def km_curve(events):
    """I calculate this part of the experiment from the information available."""
    survival = 1.0
    curve = []
    for age in range(1, 49):
        risk = int(events.time.ge(age).sum())
        hits = int((events.observed & events.time.eq(age)).sum())
        survival *= 1 - hits / risk if risk else 1
        curve.append(1 - survival)
    return np.asarray(curve)

def predict(model, events):
    """I calculate this part of the experiment from the information available."""
    x = np.vstack([np.column_stack([np.arange(1, 49), np.tile(row[COLS].to_numpy(float), (48, 1))]) for _, row in events.iterrows()])
    with threadpool_limits(limits=1):
        hazard = model.predict_proba(x)[:, 1].reshape(len(events), 48)
    return 1 - np.cumprod(1 - hazard, axis=1)

def main():
    """I print my numerical comparison, including the stated fees."""
    events = pd.concat([episodes(load_hourly(s), s) for s in COINS], ignore_index=True).sort_values(['origin', 'coin']).reset_index(drop=True)
    train = events.loc[events.complete_by < '2024-01-01'].copy()
    x, y = person_period(train)
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=100, learning_rate=0.08, l2_regularization=1, random_state=20261114, early_stopping=False)
    with threadpool_limits(limits=1):
        model.fit(x, y)
    baseline = km_curve(train)
    pred = predict(model, events)
    for h in [6, 12, 24, 48]:
        events[f'p_{h}'] = pred[:, h - 1]
        events[f'km_{h}'] = baseline[h - 1]
        events[f'y_{h}'] = (events.observed & events.time.le(h)).astype(int)
    rows = []
    for period, mask in [('diagnostic', events.origin.between('2024-01-01', '2024-12-31 23:59:59')), ('test', events.origin >= '2025-01-01')]:
        for coin in ['Pooled'] + COINS:
            q = events.loc[mask & (True if coin == 'Pooled' else events.coin.eq(coin))]
            model_score = np.mean([((q[f'p_{h}'] - q[f'y_{h}']) ** 2).mean() for h in [6, 12, 24, 48]])
            base_score = np.mean([((q[f'km_{h}'] - q[f'y_{h}']) ** 2).mean() for h in [6, 12, 24, 48]])
            rows.append({'period': period, 'coin': coin, 'episodes': len(q), 'recovered_48_pct': 100 * q.observed.mean(), 'model_mean_brier': model_score, 'baseline_mean_brier': base_score, 'median_observed_recovery_hours': float(q.loc[q.observed, 'time'].median())})
    key_results = {'training_episodes': len(train), 'person_period_rows': len(x), 'training_recovered_pct': 100 * train.observed.mean(), 'fee_per_side': FEE, 'barrier_multiplier': (1 + FEE) / (1 - FEE), 'latest_hour': str(load_hourly('ETHUSDT').index[-1])}
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    print(json.dumps(key_results, indent=2))
    return {'summary': result, 'key_results': key_results}
if __name__=='__main__':main()
