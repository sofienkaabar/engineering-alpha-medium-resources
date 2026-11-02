"""I compare drawdown-based sizing with volatility sizing. I label the matched-exposure controls as hindsight."""
import numpy as np
import pandas as pd
import requests
COST = 0.001
START = '2017-01-01'
END = '2026-10-30'
PERIODS = [('development', '2011-01-01', '2016-12-30'), ('test', START, END), ('late', '2022-01-01', END)]

def load(symbol):
    """I fetch adjusted daily prices when you run the experiment."""
    response = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}', params={'period1': 1262304000, 'period2': 1793404800, 'interval': '1d', 'events': 'div,splits'}, headers={'User-Agent': 'Mozilla/5.0'}, timeout=60)
    response.raise_for_status()
    result = response.json()['chart']['result'][0]
    dates = pd.to_datetime(result['timestamp'], unit='s', utc=True)
    dates = dates.tz_convert('America/New_York').tz_localize(None).normalize()
    frame = pd.DataFrame(result['indicators']['quote'][0], index=dates)
    adjusted = pd.Series(result['indicators']['adjclose'][0]['adjclose'], index=dates)
    factor = adjusted / frame['close']
    for column in ['open', 'high', 'low', 'close']:
        frame[column] *= factor
    frame = frame.loc[:END].dropna(subset=['open', 'high', 'low', 'close'])
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert (frame[['open', 'high', 'low', 'close']] > 0).all().all()
    assert (frame.high >= frame[['open', 'close', 'low']].max(axis=1) - 1e-08).all()
    assert (frame.low <= frame[['open', 'close', 'high']].min(axis=1) + 1e-08).all()
    return frame

def record(rows, market, label, equity, log, period='test'):
    """I keep each net result in memory instead of writing report files."""
    rows.append({'market': market, 'variant': label, 'period': period, **stats(equity, log)})

def finish(rows):
    """I print my comparison after charging entries, adjustments and exits."""
    frame = pd.DataFrame(rows)
    print('I use a 0.1% charge on each purchase and sale; these results are net of costs.')
    print(frame[['market', 'variant', 'period', 'annual_return_pct', 'max_drawdown_pct']].round(3).to_string(index=False))
    return frame

def simulate(opens, targets, start=START, end=END, cost=COST):
    """I rebalance at the open, pay fees from account capital, and liquidate at the final open."""
    if isinstance(opens, pd.Series):
        opens = opens.to_frame('asset')
    prices = opens.loc[start:end].copy()
    orders = targets.reindex(index=prices.index, columns=opens.columns)
    assert len(prices) > 2 and (not prices.isna().any().any())
    actual = np.zeros(len(prices.columns))
    values = [1.0]
    logs = []
    for i in range(len(prices) - 1):
        order = orders.iloc[i].to_numpy(float)
        before = actual.copy()
        ratio = 1.0
        turnover = 0.0
        if not np.isnan(order).all():
            assert np.isfinite(order).all() and (order >= 0).all()
            assert order.sum() <= 1.0 + 1e-10
            for _ in range(30):
                ratio = 1.0 - cost * np.abs(ratio * order - actual).sum()
            turnover = np.abs(ratio * order - actual).sum()
            actual = order
        asset_returns = prices.iloc[i + 1].to_numpy() / prices.iloc[i].to_numpy() - 1
        gross = float(np.dot(actual, asset_returns))
        after = actual * (1 + asset_returns) / (1 + gross)
        factor = ratio * (1 + gross)
        if i == len(prices) - 2:
            turnover += ratio * (1 + gross) * after.sum()
            factor *= 1 - cost * after.sum()
        logs.append({'date': prices.index[i], 'next_date': prices.index[i + 1], 'net_return': factor - 1, 'exposure': actual.sum(), 'turnover': turnover, 'changed': bool(np.max(np.abs(actual - before)) > 1e-07)})
        values.append(values[-1] * factor)
        actual = after
    equity = pd.Series(values, index=prices.index, name='equity')
    return (equity, pd.DataFrame(logs).set_index('date'))

def orders_from_signal(signal):
    """I act on a completed closing signal at the following session's open."""
    return signal.astype(float).shift(1).to_frame('asset')

def buy_hold(opens, start=START, end=END):
    """I charge the passive comparison for its initial purchase and final sale."""
    if isinstance(opens, pd.Series):
        opens = opens.to_frame('asset')
    order = pd.DataFrame(np.nan, index=opens.index, columns=opens.columns)
    first = opens.loc[start:end].index[0]
    order.loc[first] = 1 / len(opens.columns)
    return simulate(opens, order, start, end)

def stats(equity, log):
    """I measure the net return, opening-valuation drawdown and time invested."""
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    drawdown = equity / equity.cummax() - 1
    return {'total_return_pct': 100 * (equity.iloc[-1] - 1), 'annual_return_pct': 100 * (equity.iloc[-1] ** (1 / years) - 1), 'max_drawdown_pct': 100 * drawdown.min(), 'mean_exposure_pct': 100 * log.exposure.mean(), 'turnover_times': float(log.turnover.sum()), 'order_days': int(log.changed.sum()), 'first_open': str(equity.index[0].date()), 'last_open': str(equity.index[-1].date()), 'intervals': len(log)}

def sizing(data):
    """I compare rolling drawdown depth with recent daily-return volatility."""
    c = data.close
    daily_drawdown = c / c.rolling(63).max() - 1
    ulcer = daily_drawdown.pow(2).rolling(63).mean().pow(0.5)
    volatility = c.pct_change().rolling(20).std() * np.sqrt(252)
    weights = pd.DataFrame({'Ulcer sizing': (0.03 / ulcer.clip(lower=0.0001)).clip(upper=1), 'Volatility sizing': (0.1 / volatility).clip(upper=1)})
    return (weights, ulcer)

def main():
    """I run my comparison with the same trading dates and explicit costs."""
    rows = []
    for symbol in ['SPY', 'TLT', 'GLD']:
        d = load(symbol)
        weights, _ = sizing(d)
        for label in weights:
            order = orders_from_signal(weights[label])
            for period, start, stop in PERIODS:
                eq, log = simulate(d.open, order, start, stop)
                record(rows, symbol, label, eq, log, period)
        for label in weights:
            exposure = weights[label].shift(1).loc[START:END].iloc[:-1].mean()
            order = pd.DataFrame(exposure, index=d.index, columns=['asset'])
            eq, log = simulate(d.open, order)
            record(rows, symbol, 'Hindsight constant: ' + label, eq, log)
        for period, start, stop in PERIODS:
            eq, log = buy_hold(d.open, start, stop)
            record(rows, symbol, 'Buy and hold', eq, log, period)
    return {'summary': finish(rows)}
if __name__ == '__main__':
    main()
