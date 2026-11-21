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
STOCKS = ['JPM', 'BAC', 'GS', 'C']

def candidates(d, stock):
    """I describe each sell-off using information already visible at its weekly close."""
    returns = d.close.pct_change()
    vol = returns.rolling(60).std()
    rows = []
    vectors = []
    for i in weekly_positions(d):
        if i < 60 or i + 6 >= len(d) or d.close.iloc[i] / d.close.iloc[i - 20] - 1 >= -0.05:
            continue
        vector = returns.iloc[i - 19:i + 1].to_numpy() / vol.iloc[i]
        if not np.isfinite(vector).all():
            continue
        entry = d.open.iloc[i + 1]
        exit = d.open.iloc[i + 6]
        rows.append({'stock': stock, 'date': str(d.index[i].date()), 'exit_date': str(d.index[i + 6].date()), 'position': i, 'net': float(net_long(entry, exit)), 'adverse': float(min(0, net_long(entry, d.low.iloc[i + 1:i + 6].min()), net_long(entry, exit))), 'drop20': float(d.close.iloc[i] / d.close.iloc[i - 20] - 1)})
        vectors.append(vector)
    return (pd.DataFrame(rows), np.array(vectors))

def independent_neighbors(distances, dates, count=20):
    """I avoid filling the analog list with overlapping versions of the same episode."""
    keep = []
    for j in np.argsort(distances, kind='stable'):
        if all((abs((dates[j] - dates[k]).days) >= 35 for k in keep)):
            keep.append(int(j))
        if len(keep) == count:
            break
    return keep

def compute():
    """I compare the ordered path with the same return distribution without its order."""
    pieces = [candidates(load_daily(s), s) for s in STOCKS]
    library = pd.concat([p[0] for p in pieces], ignore_index=True)
    x = np.concatenate([p[1] for p in pieces])
    order = np.lexsort((library.stock.to_numpy(), library.date.to_numpy()))
    library = library.iloc[order].reset_index(drop=True)
    x = x[order]
    dates = pd.to_datetime(library.date).to_numpy()
    date_values = [pd.Timestamp(t) for t in dates]
    rows = []
    links = []
    examples = []
    for year in range(2017, 2027):
        train = np.flatnonzero(library.exit_date.lt(f'{year}-01-01').to_numpy())
        query = np.flatnonzero(library.date.str.startswith(str(year)).to_numpy())
        train_dates = [date_values[j] for j in train]
        for qi in query:
            row = library.iloc[qi].to_dict()
            row['period'] = 'Test' if year >= 2021 else 'Diagnostic'
            for label, values in [('Ordered', x), ('Sorted', np.sort(x, axis=1))]:
                dist = np.linalg.norm(values[train] - values[qi], axis=1)
                near = independent_neighbors(dist, train_dates)
                assert len(near) == 20
                selected = train[near]
                row[label.lower()] = float(library.net.iloc[selected].mean())
                if year >= 2021 and (not examples):
                    for rank, j in enumerate(selected):
                        links.append({'model': label, 'query': int(qi), 'neighbor': int(j), 'rank': rank + 1, 'distance': float(np.linalg.norm(values[j] - values[qi]))})
            row['baseline'] = float(library.net.iloc[train].mean())
            rows.append(row)
            if year >= 2021 and (not examples):
                examples.append(int(qi))
    predictions = pd.DataFrame(rows)
    summary = []
    for period in ['Diagnostic', 'Test']:
        for stock in STOCKS + ['Pooled']:
            g = predictions.loc[predictions.period.eq(period)]
            if stock != 'Pooled':
                g = g.loc[g.stock.eq(stock)]
            for model in ['ordered', 'sorted', 'baseline']:
                accepted = g.loc[g[model] > 0]
                summary.append({'period': period, 'stock': stock, 'model': model, 'count': len(g), 'mae_pp': float((g[model] - g.net).abs().mean() * 100), 'accepted': len(accepted), 'coverage_pct': len(accepted) / len(g) * 100, 'mean_accepted_pct': accepted.net.mean() * 100, 'median_accepted_pct': accepted.net.median() * 100, 'all_selloffs_mean_pct': g.net.mean() * 100, 'mean_adverse_pct': accepted.adverse.mean() * 100})
    test = predictions.loc[predictions.period.eq('Test')]
    yearly = []
    for year, g in test.groupby(test.date.str[:4]):
        yearly.append({'year': int(year), 'count': len(g), 'ordered_mae_pp': float((g.ordered - g.net).abs().mean() * 100), 'sorted_mae_pp': float((g.sorted - g.net).abs().mean() * 100), 'mae_difference_pp': float(((g.ordered - g.net).abs() - (g.sorted - g.net).abs()).mean() * 100)})
    keys = {'test_queries': len(test), 'diagnostic_queries': int(predictions.period.eq('Diagnostic').sum()), 'neighbors': 20, 'case_date': library.iloc[examples[0]].date, 'case_stock': library.iloc[examples[0]].stock, 'ordered_mae_pp': float((test.ordered - test.net).abs().mean() * 100), 'sorted_mae_pp': float((test.sorted - test.net).abs().mean() * 100), 'mae_difference_pp': float(((test.ordered - test.net).abs() - (test.sorted - test.net).abs()).mean() * 100), 'library_count': len(library), 'first_test_signal': test.date.min(), 'last_test_signal': test.date.max(), 'fee_per_side': FEE}
    for model in ['ordered', 'sorted']:
        kept = test.loc[test[model] > 0]
        keys[model + '_accepted'] = len(kept)
        keys[model + '_mean_pct'] = float(kept.net.mean() * 100)
    vec = pd.DataFrame(x, columns=[f'lag_{i}' for i in range(20)])
    vec.insert(0, 'candidate_id', np.arange(len(x)))
    return {'tables': {'summary': pd.DataFrame(summary), 'predictions': predictions, 'library': library, 'vectors': vec, 'neighbors': pd.DataFrame(links), 'yearly': pd.DataFrame(yearly)}, 'keys': keys}

if __name__=='__main__':
    result=compute()
    print(result['tables']['summary'].to_string(index=False))
    print(json.dumps(result['keys'],indent=2,allow_nan=False))
