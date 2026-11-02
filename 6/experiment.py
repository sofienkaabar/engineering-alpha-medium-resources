"""I test whether adding NR7 improves a strictly defined inside-bar entry."""
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

def pattern_signals(d):
    """I require strict nesting and separately check the seven-session minimum range."""
    inside = d.high.lt(d.high.shift(1)) & d.low.gt(d.low.shift(1))
    width = d.high - d.low
    nr7 = width.le(width.rolling(7).min())
    return {'Inside bar': inside, 'Inside plus NR7': inside & nr7}

def event_orders(signal, start=START, end=END, holding=3):
    """I hold one trade at a time and leave a flat interval after each scheduled exit."""
    dates = signal.index
    values = pd.Series(0.0, index=dates)
    eligible = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    first = dates.get_loc(eligible[0])
    last = dates.get_loc(eligible[-1])
    exit_index = -1
    for j in range(first, last + 1):
        if j < exit_index:
            values.iloc[j] = 1
            continue
        if j == exit_index:
            continue
        if j > 0 and bool(signal.iloc[j - 1]):
            values.iloc[j] = 1
            exit_index = j + holding
    return values.to_frame('asset')

def block_diagnostic(ledgers, baseline, treatment):
    """I resample three-month blocks to describe uncertainty in daily net differences."""
    output = []
    for market in ['SPY', 'QQQ', 'IWM']:
        difference = ledgers[market, treatment].net_return - ledgers[market, baseline].net_return
        groups = difference.groupby(difference.index.to_period('M'))
        monthly_sum = groups.sum().to_numpy()
        monthly_count = groups.count().to_numpy()
        n = len(monthly_sum)
        rng = np.random.default_rng(20261102)
        samples = []
        for _ in range(2000):
            starts = rng.integers(0, n, size=(n + 2) // 3)
            positions = ((starts[:, None] + np.arange(3)) % n).ravel()[:n]
            samples.append(25200 * monthly_sum[positions].sum() / monthly_count[positions].sum())
        lower, upper = np.percentile(samples, [2.5, 97.5])
        output.append({'market': market, 'baseline': baseline, 'treatment': treatment, 'annualized_mean_difference_points': 25200 * difference.mean(), 'bootstrap_lower_points': lower, 'bootstrap_upper_points': upper, 'replicates': 2000, 'block_months': 3})
    return pd.DataFrame(output)

def main():
    """I run my comparison with the same trading dates and explicit costs."""
    rows = []
    ledgers = {}
    counts = []
    for symbol in ['SPY', 'QQQ', 'IWM']:
        d = load(symbol)
        for label, signal in pattern_signals(d).items():
            for period, start, stop in PERIODS:
                order = event_orders(signal, start, stop)
                eq, log = simulate(d.open, order, start, stop)
                record(rows, symbol, label, eq, log, period)
                if period == 'test':
                    ledgers[symbol, label] = log
                counts.append({'market': symbol, 'variant': label, 'period': period, 'raw_signals': int(signal.loc[start:stop].sum()), 'entries': int((log.exposure.gt(0) & log.exposure.shift(1, fill_value=0).eq(0)).sum())})
        for period, start, stop in PERIODS:
            eq, log = buy_hold(d.open, start, stop)
            record(rows, symbol, 'Buy and hold', eq, log, period)
    summary = finish(rows)
    diagnostic = block_diagnostic(ledgers, 'Inside bar', 'Inside plus NR7')
    print('\nI check the return difference with a descriptive block bootstrap:')
    print(diagnostic.round(3).to_string(index=False))
    return {'summary': summary, 'block_bootstrap': diagnostic, 'signal_counts': pd.DataFrame(counts)}
if __name__ == '__main__':
    main()
