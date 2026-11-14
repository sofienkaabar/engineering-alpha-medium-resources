"""I reproduce the numerical study for this article."""
import io,json,zipfile
import numpy as np
import pandas as pd
import requests
FEE=0.001
COINS = ['LTCUSDT', 'LINKUSDT']
DAYS = ['2026-11-09', '2026-11-10', '2026-11-11']

def archive(name, url):
    """I read the public archive in memory without saving a dataset or report."""
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(response.content))
    return (z, pd.read_csv(z.open(z.namelist()[0]), header=None))

def load_tape(symbol, day):
    """I calculate this part of the experiment from the information available."""
    name = f'{symbol}-aggTrades-{day}.zip'
    z, d = archive(name, f'https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{name}')
    d.columns = ['id', 'price', 'quantity', 'first_id', 'last_id', 'timestamp', 'buyer_maker', 'best_match']
    z.close()
    if d.buyer_maker.dtype != bool:
        d['buyer_maker'] = d.buyer_maker.astype(str).str.lower().eq('true')
    unit = 'us' if d.timestamp.median() > 100000000000000.0 else 'ms'
    d['time'] = pd.to_datetime(d.timestamp, unit=unit, utc=True)
    assert d.id.is_unique and d.id.is_monotonic_increasing and d.time.is_monotonic_increasing
    assert (d.price > 0).all() and (d.quantity > 0).all()
    return d

def markouts(d, symbol, day):
    """I calculate this part of the experiment from the information available."""
    notional = d.price * d.quantity
    sign = np.where(d.buyer_maker, -1.0, 1.0)
    flow = (pd.Series(sign, index=d.index) * notional).rolling(50).sum() / notional.rolling(50).sum()
    flow = flow.shift(1)
    rows = []
    for i in range(50, len(d) - 500, 500):
        buy = bool(d.buyer_maker.iloc[i])
        f = float(flow.iloc[i])
        adverse = -f if buy else f
        state = 'Adverse flow' if adverse > 0.6 else 'Favorable flow' if adverse < -0.6 else 'Balanced' if abs(f) < 0.2 else 'Mixed'
        row = {'coin': symbol, 'day': day, 'position': i, 'id': int(d.id.iloc[i]), 'time': str(d.time.iloc[i]), 'maker_buy': buy, 'flow': f, 'adverse_flow': adverse, 'state': state}
        for h in [20, 100, 500]:
            now = float(d.price.iloc[i])
            later = float(d.price.iloc[i + h])
            net = later / now * (1 - 0.0004) / (1 + 0.0002) - 1 if buy else now / later * (1 - 0.0002) / (1 + 0.0004) - 1
            row[f'net_bps_{h}'] = 10000 * net
            row[f'seconds_{h}'] = (d.time.iloc[i + h] - d.time.iloc[i]).total_seconds()
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    """I print my numerical comparison, including the stated fees."""
    parts = []
    counts = []
    for coin in COINS:
        for day in DAYS:
            d = load_tape(coin, day)
            parts.append(markouts(d, coin, day))
            counts.append({'coin': coin, 'day': day, 'aggregates': len(d)})
    events = pd.concat(parts, ignore_index=True)
    summary = events.groupby(['coin', 'state']).agg(executions=('id', 'size'), mean_net_bps_100=('net_bps_100', 'mean'), median_net_bps_100=('net_bps_100', 'median'), median_seconds_100=('seconds_100', 'median')).reset_index()
    daily = events.groupby(['coin', 'day', 'state']).agg(executions=('id', 'size'), mean_net_bps_100=('net_bps_100', 'mean')).reset_index()
    key_results = {'aggregated_trades': sum((x['aggregates'] for x in counts)), 'sampled_executions': len(events), 'maker_fee_bps': 2, 'unwind_fee_bps': 4, 'first_day': DAYS[0], 'last_day': DAYS[-1]}
    result = summary
    print(result.to_string(index=False))
    print(json.dumps(key_results, indent=2))
    return {'summary': result, 'key_results': key_results}
if __name__=='__main__':main()
