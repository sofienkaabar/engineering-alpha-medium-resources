"""I reproduce the numerical experiment for my article."""
import io,json,zipfile
import numpy as np
import pandas as pd
import requests
from xgboost import XGBRanker
from threadpoolctl import threadpool_limits
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
STOCKS = ['WMT', 'COST', 'KR', 'DG', 'DLTR', 'BBY']
FEATURES = ['relative20', 'reversal5', 'vol20', 'range', 'relative_volume', 'momentum60']

def panel_data():
    """I build weekly groups with features at the close and returns from later openings."""
    frames = {s: load_daily(s) for s in STOCKS}
    common = frames[STOCKS[0]].index
    for d in frames.values():
        common = common.intersection(d.index)
    common = common[common >= pd.Timestamp('2011-01-01')]
    opens = pd.DataFrame({s: d.open.reindex(common) for s, d in frames.items()})
    close = pd.DataFrame({s: d.close.reindex(common) for s, d in frames.items()})
    raw = {}
    for s, d in frames.items():
        r = d.close.pct_change()
        raw[s] = pd.DataFrame({'momentum20': d.close.pct_change(20), 'reversal5': -d.close.pct_change(5), 'vol20': r.rolling(20).std(), 'range': (d.high - d.low) / d.close, 'relative_volume': d.volume / d.volume.rolling(20).mean(), 'momentum60': d.close.pct_change(60)})
    positions = weekly_positions(opens)
    rows = []
    for wi in range(len(positions) - 1):
        i, j = (positions[wi], positions[wi + 1])
        if i + 1 >= len(opens) or j + 1 >= len(opens):
            continue
        x = pd.DataFrame({s: raw[s].loc[common[i]] for s in STOCKS}).T
        x['relative20'] = x.momentum20 - x.momentum20.mean()
        if not np.isfinite(x[FEATURES].to_numpy()).all():
            continue
        outcome = net_long(opens.iloc[i + 1], opens.iloc[j + 1])
        rank = outcome.rank(method='first').astype(int) - 1
        for s in STOCKS:
            row = {'stock': s, 'date': str(common[i].date()), 'entry_date': str(common[i + 1].date()), 'exit_date': str(common[j + 1].date()), 'entry_index': i + 1, 'exit_index': j + 1, 'label': int(rank[s]), 'future_net': float(outcome[s]), 'momentum20': float(x.loc[s, 'momentum20'])}
            row.update({f: float(x.loc[s, f]) for f in FEATURES})
            rows.append(row)
    return (opens, pd.DataFrame(rows))

def choose_weights(g, model):
    """I keep the same two-stock allocation rule for the model and momentum."""
    if model in ['Basket', 'Buy and hold']:
        return np.full(len(STOCKS), 1 / len(STOCKS))
    column = 'score' if model == 'Ranker' else 'momentum20'
    selected = g.sort_values([column, 'stock'], ascending=[False, True]).stock.iloc[:2]
    return np.array([0.5 if s in set(selected) else 0 for s in STOCKS])

def account(opens, groups, model):
    """I solve target weights after fees and charge actual changes in holdings."""
    quantity = np.zeros(len(STOCKS))
    cash = 1.0
    curve = []
    attribution = []
    allocations = []
    turnover = 0.0
    previous_value = np.zeros(len(STOCKS))
    fees_total = 0.0
    trades = 0
    for date, g in groups:
        ei = int(g.entry_index.iloc[0])
        xi = int(g.exit_index.iloc[0])
        price = opens.iloc[ei].to_numpy()
        current = quantity * price
        pnl = current - previous_value
        pre = float(current.sum() + cash)
        weights = current / pre if model == 'Buy and hold' and quantity.sum() > 0 else choose_weights(g, model)
        lower, upper = (pre * (1 - 2 * FEE), pre)
        for _ in range(50):
            after = (lower + upper) / 2
            if after + FEE * np.abs(weights * after - current).sum() > pre:
                upper = after
            else:
                lower = after
        after = (lower + upper) / 2
        changes = weights * after - current
        fees = FEE * np.abs(changes)
        assert abs(after + fees.sum() - pre) < 1e-10
        turnover += float(np.abs(changes).sum() / pre)
        trades += int((np.abs(changes) > 1e-12).sum())
        fees_total += float(fees.sum())
        for k, s in enumerate(STOCKS):
            attribution.append({'strategy': model, 'date': str(opens.index[ei].date()), 'stock': s, 'contribution_pct': float((pnl[k] - fees[k]) * 100), 'price_pnl': float(pnl[k]), 'fee': float(fees[k])})
            allocations.append({'strategy': model, 'date': date, 'stock': s, 'weight': float(weights[k]), 'trade_amount': float(changes[k])})
        quantity = weights * after / price
        cash = 0.0
        previous_value = weights * after
        if curve and curve[-1]['date'] == str(opens.index[ei].date()):
            curve[-1]['wealth'] = after
        else:
            curve.append({'strategy': model, 'date': str(opens.index[ei].date()), 'wealth': after})
        for k in range(ei + 1, xi + 1):
            curve.append({'strategy': model, 'date': str(opens.index[k].date()), 'wealth': float((quantity * opens.iloc[k].to_numpy()).sum())})
    final = quantity * opens.iloc[xi].to_numpy()
    pnl = final - previous_value
    fees = FEE * final
    for k, s in enumerate(STOCKS):
        attribution.append({'strategy': model, 'date': str(opens.index[xi].date()), 'stock': s, 'contribution_pct': float((pnl[k] - fees[k]) * 100), 'price_pnl': float(pnl[k]), 'fee': float(fees[k])})
    fees_total += float(fees.sum())
    turnover += 1
    trades += int((final > 1e-12).sum())
    curve[-1]['wealth'] = float((final - fees).sum())
    c = pd.DataFrame(curve)
    c['peak'] = np.maximum.accumulate(np.r_[1.0, c.wealth.to_numpy()])[1:]
    c['drawdown'] = c.wealth / c.peak - 1
    trough = int(c.drawdown.argmin())
    peak_value = float(c.peak.iloc[trough])
    prior = c.iloc[:trough + 1]
    peak_rows = prior.loc[np.isclose(prior.wealth, peak_value, rtol=1e-10, atol=1e-12)]
    peak_date = peak_rows.date.iloc[-1] if len(peak_rows) else c.date.iloc[0]
    later = c.iloc[trough + 1:]
    recovery = later.loc[later.wealth >= peak_value]
    recovery_date = recovery.date.iloc[0] if len(recovery) else 'Not recovered'
    years = (pd.Timestamp(c.date.iloc[-1]) - pd.Timestamp(c.date.iloc[0])).days / 365.25
    row = {'strategy': model, 'net_return_pct': float((c.wealth.iloc[-1] - 1) * 100), 'cagr_pct': float((c.wealth.iloc[-1] ** (1 / years) - 1) * 100), 'max_drawdown_pct': float(c.drawdown.min() * 100), 'annual_turnover': turnover / years, 'fee_initial_capital_pct': fees_total * 100, 'trades': trades, 'start': c.date.iloc[0], 'end': c.date.iloc[-1], 'peak_date': peak_date, 'trough_date': c.date.iloc[trough], 'recovery_date': recovery_date}
    att = pd.DataFrame(attribution)
    assert abs(att.contribution_pct.sum() - row['net_return_pct']) < 1e-08
    return (row, c, att, pd.DataFrame(allocations))

def compute():
    """I learn relative preferences without allowing incomplete future labels into training."""
    opens, panel = panel_data()
    predictions = []
    for year in range(2017, 2027):
        train = panel.loc[panel.exit_date.lt(f'{year}-01-01')]
        test = panel.loc[panel.date.str.startswith(str(year))].copy()
        if test.empty:
            continue
        sizes = train.groupby('date', sort=False).size().to_numpy()
        assert (sizes == 6).all()
        model = XGBRanker(objective='rank:pairwise', n_estimators=80, max_depth=2, learning_rate=0.05, min_child_weight=10, reg_lambda=5, subsample=1.0, colsample_bytree=1.0, n_jobs=1, random_state=SEED)
        with threadpool_limits(limits=1):
            model.fit(train[FEATURES].to_numpy(), train.label.to_numpy(), group=sizes, verbose=False)
            test['score'] = model.predict(test[FEATURES].to_numpy())
        test['period'] = 'Test' if year >= 2020 else 'Diagnostic'
        predictions.append(test)
    pred = pd.concat(predictions, ignore_index=True)
    test = pred.loc[pred.period.eq('Test')]
    groups = list(test.groupby('date', sort=True))
    summary = []
    curves = []
    atts = []
    weights = []
    for strategy in ['Ranker', 'Momentum', 'Basket', 'Buy and hold']:
        row, c, a, w = account(opens, groups, strategy)
        summary.append(row)
        curves.append(c)
        atts.append(a)
        weights.append(w)
    quality = []
    for date, g in groups:
        correct = 0
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                correct += int((g.score.iloc[i] - g.score.iloc[j]) * (g.label.iloc[i] - g.label.iloc[j]) > 0)
        quality.append({'date': date, 'pair_accuracy_pct': correct / 15 * 100, 'rank_correlation': float(g.score.rank().corr(g.label))})
    s = pd.DataFrame(summary)
    keys = {'test_weeks': len(groups), 'first_test_signal': test.date.min(), 'last_test_signal': test.date.max(), 'mean_pair_accuracy_pct': float(pd.DataFrame(quality).pair_accuracy_pct.mean()), 'mean_rank_correlation': float(pd.DataFrame(quality).rank_correlation.mean()), 'fee_per_side': FEE}
    for row in summary:
        for field in ['net_return_pct', 'cagr_pct', 'max_drawdown_pct', 'annual_turnover']:
            keys[row['strategy'].lower() + '_' + field] = row[field]
    return {'tables': {'summary': s, 'predictions': pred, 'portfolio': pd.concat(curves, ignore_index=True), 'attribution': pd.concat(atts, ignore_index=True), 'weights': pd.concat(weights, ignore_index=True), 'quality': pd.DataFrame(quality)}, 'keys': keys}

if __name__=='__main__':
    result=compute()
    print(result['tables']['summary'].to_string(index=False))
    print(json.dumps(result['keys'],indent=2,allow_nan=False))
