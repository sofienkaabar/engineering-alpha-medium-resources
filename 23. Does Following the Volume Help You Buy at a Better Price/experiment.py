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
COINS = ['ADAUSDT', 'DOGEUSDT', 'XRPUSDT']
DAYS = ['2026-11-02', '2026-11-05', '2026-11-09', '2026-11-12', '2026-11-16', '2026-11-19']

def minute_data(coin, day):
    """I fetch completed minute bars and keep the download in memory."""
    name = f'{coin}-1m-{day}.zip'
    response = requests.get(f'https://data.binance.vision/data/spot/daily/klines/{coin}/1m/{name}', timeout=90)
    response.raise_for_status()
    payload = response.content
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        d = pd.read_csv(z.open(z.namelist()[0]), header=None)
    d = d.iloc[:, :6].apply(pd.to_numeric, errors='coerce').dropna()
    d.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    unit = 'us' if d.time.median() > 100000000000000.0 else 'ms'
    d.index = pd.to_datetime(d.pop('time'), unit=unit, utc=True)
    assert d.index.is_unique and d.index.is_monotonic_increasing
    assert (d.index.to_series().diff().dropna() >= pd.Timedelta(minutes=1)).all()
    assert (d[['open', 'high', 'low', 'close']] > 0).all().all() and (d.volume >= 0).all()
    assert (d.high >= d[['open', 'close', 'low']].max(axis=1) - 1e-09).all()
    assert (d.low <= d[['open', 'close', 'high']].min(axis=1) + 1e-09).all()
    return d

def schedule(d, start):
    """I choose every next-minute slice from information already completed."""
    q = 0.001 * d.volume.iloc[start - 60:start].sum()
    remaining = q
    adaptive = []
    for k in range(60):
        i = start + k
        past = d.volume.iloc[i - 60:i]
        last = past.iloc[-1]
        median = past.median()
        factor = 0.5 if last < median else 1.5 if last > median else 1.0
        amount = remaining if k == 59 else min(remaining, q / 60 * factor)
        adaptive.append(amount)
        remaining = max(0, remaining - amount)
    assert abs(sum(adaptive) - q) < max(1e-08, q * 1e-10)
    return (q, np.array(adaptive), np.full(60, q / 60))

def compute():
    """I compare equal-size acquisitions, with the same charged execution assumptions."""
    windows = []
    fills = []
    for coin in COINS:
        for day in DAYS:
            d = minute_data(coin, day)
            for hour in range(1, 24):
                stamp = pd.Timestamp(day + 'T00:00:00Z') + pd.Timedelta(hours=hour)
                if stamp not in d.index:
                    continue
                start = d.index.get_loc(stamp)
                part = d.iloc[start - 60:start + 60]
                if len(part) != 120 or not (part.index.to_series().diff().dropna() == pd.Timedelta(minutes=1)).all():
                    continue
                q, adaptive, clock = schedule(d, start)
                if q <= 0:
                    continue
                price = d.open.iloc[start:start + 60].to_numpy()
                charged = price * (1 + 0.0003) * (1 + 0.0005)
                ordinary = float(clock @ charged / q)
                aware = float(adaptive @ charged / q)
                arrival = price[0]
                actual_volume = d.volume.iloc[start:start + 60].to_numpy()
                participation = np.divide(adaptive, actual_volume, out=np.full(60, np.nan), where=actual_volume > 0)
                windows.append({'coin': coin, 'day': day, 'hour': hour, 'date': stamp.isoformat(), 'period': 'Diagnostic' if day in DAYS[:3] else 'Test', 'quantity': float(q), 'clock_price': ordinary, 'adaptive_price': aware, 'arrival': float(arrival), 'difference_bp': (aware / ordinary - 1) * 10000, 'clock_shortfall_bp': (ordinary / arrival - 1) * 10000, 'adaptive_shortfall_bp': (aware / arrival - 1) * 10000, 'last_slice_pct': float(adaptive[-1] / q * 100), 'complete_minute': int(np.flatnonzero(np.cumsum(adaptive) >= q * (1 - 1e-10))[0]), 'max_participation_pct': float(np.nanmax(participation) * 100), 'zero_volume_fills': int(((actual_volume == 0) & (adaptive > 0)).sum()), 'clock_zero_volume_fills': int((actual_volume == 0).sum()), 'slices_exceed_recorded_volume': int((adaptive > actual_volume).sum()), 'clock_slices_exceed_recorded_volume': int((clock > actual_volume).sum())})
                for k in range(60):
                    fills.append({'coin': coin, 'day': day, 'hour': hour, 'minute': k, 'price': float(price[k]), 'adaptive_fraction': float(adaptive[k] / q), 'clock_fraction': 1 / 60, 'adaptive_done': float(adaptive[:k + 1].sum() / q), 'period': 'Diagnostic' if day in DAYS[:3] else 'Test'})
    windows = pd.DataFrame(windows)
    fills = pd.DataFrame(fills)
    rows = []
    for period in ['Diagnostic', 'Test']:
        for coin in COINS + ['Pooled']:
            g = windows.loc[windows.period.eq(period)]
            if coin != 'Pooled':
                g = g.loc[g.coin.eq(coin)]
            rows.append({'period': period, 'coin': coin, 'count': len(g), 'mean_difference_bp': g.difference_bp.mean(), 'median_difference_bp': g.difference_bp.median(), 'better_pct': g.difference_bp.lt(0).mean() * 100, 'mean_last_slice_pct': g.last_slice_pct.mean(), 'p90_last_slice_pct': g.last_slice_pct.quantile(0.9), 'worst_difference_bp': g.difference_bp.max(), 'best_difference_bp': g.difference_bp.min(), 'mean_clock_shortfall_bp': g.clock_shortfall_bp.mean(), 'mean_adaptive_shortfall_bp': g.adaptive_shortfall_bp.mean()})
    test = windows.query('period == "Test"')
    daily = test.groupby(['coin', 'day']).agg(count=('difference_bp', 'size'), mean_difference_bp=('difference_bp', 'mean'), median_difference_bp=('difference_bp', 'median')).reset_index()
    keys = {'test_windows': len(test), 'all_windows': len(windows), 'fee_bp': 5, 'modeled_spread_slippage_bp': 3, 'mean_difference_bp': float(test.difference_bp.mean()), 'median_difference_bp': float(test.difference_bp.median()), 'better_pct': float(test.difference_bp.lt(0).mean() * 100), 'p90_last_slice_pct': float(test.last_slice_pct.quantile(0.9)), 'max_last_slice_pct': float(test.last_slice_pct.max()), 'max_participation_pct': float(test.max_participation_pct.max()), 'zero_volume_fills': int(test.zero_volume_fills.sum()), 'clock_zero_volume_fills': int(test.clock_zero_volume_fills.sum()), 'slices_exceed_recorded_volume': int(test.slices_exceed_recorded_volume.sum()), 'clock_slices_exceed_recorded_volume': int(test.clock_slices_exceed_recorded_volume.sum()), 'zero_volume_windows': int(test.clock_zero_volume_fills.gt(0).sum()), 'case_coin': 'XRPUSDT', 'case_day': '2026-11-12', 'case_hour': 12}
    return {'tables': {'summary': pd.DataFrame(rows), 'windows': windows, 'fills': fills, 'daily': daily}, 'keys': keys}

if __name__=='__main__':
    result=compute()
    print(result['tables']['summary'].to_string(index=False))
    print(json.dumps(result['keys'],indent=2,allow_nan=False))
