"""I reproduce my I Replaced RSI's Fixed Oversold Level With a Rolling Floor experiment."""
import io
import numpy as np
import pandas as pd
import requests
COST=0.0002
HOLDING_DRAG=0.03
FX=True
START='2017-01-01'
END='2026-10-30'
ARTICLE=8
MARKETS=['EURUSD=X', 'GBPUSD=X', 'JPY=X']
PERIODS=[['earlier', '2011-01-01', '2016-12-30'], ['test', '2017-01-01', '2026-10-30'], ['late', '2022-01-01', '2026-10-30']]
FX_SERIES={'EURUSD=X':'DEXUSEU','GBPUSD=X':'DEXUSUK','JPY=X':'DEXJPUS','AUDUSD=X':'DEXUSAL','CAD=X':'DEXCAUS','CHF=X':'DEXSZUS'}
def simulate(opens, targets, start=START, end=END, cost=COST):
    """I rebalance at the open, pay actual trading fees, and liquidate at the final opening."""
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
        asset_returns = prices.iloc[i + 1].to_numpy() / prices.iloc[i].to_numpy() * np.exp(-HOLDING_DRAG * (prices.index[i + 1] - prices.index[i]).days / 365.25) - 1
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
    """I act on each completed close at the following open."""
    return signal.astype(float).shift(1).to_frame('asset')

def buy_hold(opens, start=START, end=END):
    """I charge my passive account for the initial purchase and final sale."""
    if isinstance(opens, pd.Series):
        opens = opens.to_frame('asset')
    order = pd.DataFrame(np.nan, index=opens.index, columns=opens.columns)
    first = opens.loc[start:end].index[0]
    order.loc[first] = 1 / len(opens.columns)
    return simulate(opens, order, start, end)

def stats(equity, log):
    """I measure net returns, opening account drawdowns and invested time."""
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    drawdown = equity / equity.cummax() - 1
    return {'total_return_pct': 100 * (equity.iloc[-1] - 1), 'annual_return_pct': 100 * (equity.iloc[-1] ** (1 / years) - 1), 'max_drawdown_pct': 100 * drawdown.min(), 'mean_exposure_pct': 100 * log.exposure.mean(), 'turnover_times': float(log.turnover.sum()), 'order_days': int(log.changed.sum()), 'first_open': str(equity.index[0].date()), 'last_open': str(equity.index[-1].date()), 'intervals': len(log)}

def atr(data):
    """I calculate this part of my experiment using only the available history."""
    if FX:
        return data.open.diff().abs().rolling(14).mean()
    return pd.concat([data.high - data.low, (data.high - data.close.shift()).abs(), (data.low - data.close.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()

def event_orders(signal, index, start, end, hold):
    """I ignore overlapping events and count holding intervals from the entry open."""
    result = pd.Series(0.0, index=index)
    state = 0
    age = 0
    for i in range(len(index) - 1):
        if not start <= str(index[i + 1].date()) <= end:
            continue
        if state:
            age += 1
            if age >= hold:
                state = 0
                age = 0
        elif bool(signal.iloc[i]):
            state = 1
            age = 0
        result.iloc[i + 1] = state
    return result.to_frame('asset')

def channel_orders(entry, exit_signal, index, start, end, cap):
    """I combine my fresh channel entry with the tested exit and deadline."""
    result = pd.Series(0.0, index=index)
    state = 0
    age = 0
    for i in range(len(index) - 1):
        if not start <= str(index[i + 1].date()) <= end:
            continue
        if state:
            age += 1
            if bool(exit_signal.iloc[i]) or (cap is not None and age >= cap):
                state = 0
                age = 0
        elif bool(entry.iloc[i]):
            state = 1
            age = 0
        result.iloc[i + 1] = state
    return result.to_frame('asset')

def rsi(close, length=14):
    """I calculate this part of my experiment using only the available history."""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = np.full(len(close), np.nan)
    loss = gain.copy()
    gain[length] = up.iloc[1:length + 1].mean()
    loss[length] = down.iloc[1:length + 1].mean()
    for i in range(length + 1, len(close)):
        gain[i] = (gain[i - 1] * (length - 1) + up.iloc[i]) / length
        loss[i] = (loss[i - 1] * (length - 1) + down.iloc[i]) / length
    total = gain + loss
    values = np.divide(100 * gain, total, out=np.full(len(close), 50.0), where=total > 0)
    values[:length] = np.nan
    return pd.Series(values, index=close.index)

def indicators(data):
    """I calculate this part of my experiment using only the available history."""
    oscillator = rsi(data.open if FX else data.close)
    result = pd.DataFrame({'RSI': oscillator, 'Fixed 30': 30.0})
    for p in [0.05, 0.1, 0.2]:
        result[f'Floor {int(p * 100)}'] = oscillator.shift(1).rolling(252).quantile(p)
    return result

def signals(data):
    """I calculate this part of my experiment using only the available history."""
    lines = indicators(data)
    osc = lines.RSI
    return {label: (osc > lines[label]) & (osc.shift(1) <= lines[label].shift(1)) for label in ['Fixed 30', 'Floor 5', 'Floor 10', 'Floor 20']}

def orders(data, start, end):
    """I calculate this part of my experiment using only the available history."""
    return {label: event_orders(signal, data.index, start, end, 10) for label, signal in signals(data).items()}
def load(symbol):
    """I fetch adjusted daily prices when you run my experiment."""
    if FX:
        response=requests.get('https://fred.stlouisfed.org/graph/fredgraph.csv',
            params={'id':FX_SERIES[symbol],'cosd':'2010-01-01','coed':END},timeout=60)
        response.raise_for_status()
        raw=pd.read_csv(io.StringIO(response.text),parse_dates=[0],index_col=0)
        price=pd.to_numeric(raw[FX_SERIES[symbol]],errors='coerce').dropna().loc[:END]
        assert (price>0).all() and price.index.is_unique and price.index.is_monotonic_increasing
        return price.to_frame('open')
    response=requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}',
        params={'period1':1262304000,'period2':1794009600,'interval':'1d','events':'div,splits'},
        headers={'User-Agent':'Mozilla/5.0'},timeout=60)
    response.raise_for_status()
    result=response.json()['chart']['result'][0]
    dates=pd.to_datetime(result['timestamp'],unit='s',utc=True).tz_convert(result['meta'].get('exchangeTimezoneName','America/New_York')).tz_localize(None).normalize()
    frame=pd.DataFrame(result['indicators']['quote'][0],index=dates)
    adjusted=pd.Series(result['indicators']['adjclose'][0]['adjclose'],index=dates)
    factor=adjusted/frame.close
    for column in ['open','high','low','close']:frame[column]*=factor
    frame=frame.loc[:END].dropna(subset=['open','high','low','close'])
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    if FX:
        frame=frame[['open']].dropna()
        assert (frame.open>0).all()
        return frame
    assert (frame[['open','high','low','close']]>0).all().all()
    assert (frame.high>=frame[['open','close','low']].max(axis=1)-1e-8).all()
    assert (frame.low<=frame[['open','close','high']].min(axis=1)+1e-8).all()
    return frame

def main():
    """I print my net account comparisons over the same trading dates."""
    rows=[]
    for symbol in MARKETS:
        data=load(symbol)
        pred=predictions(data) if ARTICLE==14 else None
        for period,start,end in PERIODS:
            if ARTICLE==14 and period=='earlier':continue
            plans=model_orders(data,pred,start,end) if ARTICLE==14 else orders(data,start,end)
            for label,plan in plans.items():
                equity,log=simulate(data.open,plan,start,end)
                rows.append({'market':symbol,'variant':label,'period':period,**stats(equity,log)})
            equity,log=buy_hold(data.open,start,end)
            rows.append({'market':symbol,'variant':'Buy and hold','period':period,**stats(equity,log)})
    frame=pd.DataFrame(rows)
    print(f'I charge {100*COST:g}% per purchase or sale and {100*HOLDING_DRAG:g}% annual holding cost on invested capital, including my passive comparison.')
    print(frame[['market','variant','period','annual_return_pct','max_drawdown_pct']].round(3).to_string(index=False))
    return frame

if __name__=='__main__':main()
