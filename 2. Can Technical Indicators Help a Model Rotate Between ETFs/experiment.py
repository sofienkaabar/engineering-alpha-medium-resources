"""I compare a price-based allocation model with relative strength and passive portfolios."""
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
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

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

def features(data):
    """I build comparable technical features from information available at each close."""
    c = data.close
    return pd.DataFrame({'return_5': c.pct_change(5), 'return_20': c.pct_change(20), 'return_60': c.pct_change(60), 'volatility_20': c.pct_change().rolling(20).std(), 'distance_200': c / c.rolling(200).mean() - 1, 'range_20': ((data.high - data.low) / c).rolling(20).mean()})

def rotation_orders(data):
    """I refit monthly and leave unfinished forward returns out of training."""
    names = list(data)
    dates = data[names[0]].index
    xs = {s: features(d) for s, d in data.items()}
    ys = {s: d.open.shift(-22) / d.open.shift(-1) * (1 - COST) / (1 + COST) - 1 for s, d in data.items()}
    orders = {k: pd.DataFrame(np.nan, index=dates, columns=names) for k in ['Gradient boosting', 'Past 60 sessions', 'Equal weight monthly']}
    schedule = pd.Series(dates, index=dates).groupby(dates.to_period('M')).last()
    for day in schedule:
        i = dates.get_loc(day)
        if day < pd.Timestamp('2016-12-01') or i + 1 >= len(dates):
            continue
        training_dates = dates[:i - 22]
        train_x = pd.concat([xs[s].reindex(training_dates) for s in names], ignore_index=True)
        train_y = pd.concat([ys[s].reindex(training_dates) for s in names], ignore_index=True)
        valid = train_x.notna().all(axis=1) & train_y.notna()
        model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7, min_samples_leaf=40, learning_rate=0.05, l2_regularization=5, early_stopping=False, random_state=42)
        with threadpool_limits(limits=1):
            model.fit(train_x.loc[valid], train_y.loc[valid])
            forecasts = pd.Series(model.predict(pd.DataFrame([xs[s].loc[day] for s in names])), index=names)
        chosen = forecasts.idxmax()
        momentum = pd.Series({s: xs[s].loc[day, 'return_60'] for s in names}).idxmax()
        execution = dates[i + 1]
        orders['Gradient boosting'].loc[execution] = [float(s == chosen) for s in names]
        orders['Past 60 sessions'].loc[execution] = [float(s == momentum) for s in names]
        orders['Equal weight monthly'].loc[execution] = 1 / len(names)
    return orders

def main():
    """I run my comparison with the same trading dates and explicit costs."""
    data = {s: load(s) for s in ['SPY', 'EFA', 'TLT', 'GLD']}
    opens = pd.DataFrame({s: d.open for s, d in data.items()})
    orders = rotation_orders(data)
    rows = []
    for period, start in [('test', START), ('late', '2022-01-01')]:
        for label, order in orders.items():
            current = order.copy()
            first = opens.loc[start:].index[0]
            current.loc[first] = current.loc[:first].ffill().iloc[-1]
            eq, log = simulate(opens, current, start)
            record(rows, 'Four ETFs', label, eq, log, period)
        for market, price in [('Four ETFs', opens), ('SPY', data['SPY'].open)]:
            eq, log = buy_hold(price, start)
            label = 'Equal weight buy and hold' if market == 'Four ETFs' else 'Buy and hold'
            record(rows, market, label, eq, log, period)
    return {'summary': finish(rows)}
if __name__ == '__main__':
    main()
