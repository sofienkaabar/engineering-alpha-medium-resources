"""Shared source copied into every independent article script by build_scripts.py."""
from pathlib import Path
import hashlib
import json
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
COST = 0.001  # 0.1% per side, on actual traded notional.
START = "2017-01-01"
END = "2026-10-30"
plt.rcParams.update({"font.size": 17, "axes.titlesize": 19,
                     "axes.labelsize": 17, "legend.fontsize": 14,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def load(symbol):
    """Use the saved Yahoo response; download only if it is absent.

    Scale OHLC by adjusted close / raw close. This is a dividend-adjusted
    return proxy, not an assertion that historical adjusted prices traded.
    """
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{symbol}_yahoo.json"
    if not path.exists():
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        response = requests.get(url, params={"period1": 1262304000,
            "period2": 1793404800, "interval": "1d", "events": "div,splits"},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    result = json.loads(path.read_bytes())["chart"]["result"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    dates = dates.tz_convert("America/New_York").tz_localize(None).normalize()
    frame = pd.DataFrame(result["indicators"]["quote"][0], index=dates)
    adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    factor = pd.Series(adjusted, index=dates) / frame["close"]
    for col in ["open", "high", "low", "close"]:
        frame[col] = frame[col] * factor
    frame = frame.loc[:END].dropna(subset=["open", "high", "low", "close"])
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert (frame[["open", "high", "low", "close"]] > 0).all().all()
    assert (frame.high >= frame[["open", "close", "low"]].max(axis=1) - 1e-8).all()
    assert (frame.low <= frame[["open", "close", "high"]].min(axis=1) + 1e-8).all()
    return frame


def simulate(opens, targets, start=START, end=END, cost=COST):
    """Orders are target weights at each execution open. NaN means hold.

    An interval runs from this open to the next open. At rebalance, solve
    for post-fee wealth, so fully invested trades never borrow their fee.
    Between orders, weights drift with asset prices. Liquidate at the last
    open and charge that exit. Cash earns zero. No shorting or leverage.
    """
    if isinstance(opens, pd.Series):
        opens = opens.to_frame("asset")
    prices = opens.loc[start:end].copy()
    orders = targets.reindex(index=prices.index, columns=opens.columns)
    assert len(prices) > 2 and not prices.isna().any().any()
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
        logs.append({"date": prices.index[i], "next_date": prices.index[i + 1],
                     "net_return": factor - 1, "exposure": actual.sum(),
                     "turnover": turnover, "changed": bool(np.max(np.abs(actual-before)) > 1e-7)})
        values.append(values[-1] * factor)
        actual = after
    equity = pd.Series(values, index=prices.index, name="equity")
    return equity, pd.DataFrame(logs).set_index("date")




def buy_hold(opens, start=START, end=END):
    if isinstance(opens, pd.Series):
        opens = opens.to_frame("asset")
    order = pd.DataFrame(np.nan, index=opens.index, columns=opens.columns)
    first = opens.loc[start:end].index[0]
    order.loc[first] = 1 / len(opens.columns)
    return simulate(opens, order, start, end)


def stats(equity, log):
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    drawdown = equity / equity.cummax() - 1
    return {"total_return_pct": 100 * (equity.iloc[-1] - 1),
            "annual_return_pct": 100 * (equity.iloc[-1] ** (1 / years) - 1),
            "max_drawdown_pct": 100 * drawdown.min(),
            "mean_exposure_pct": 100 * log.exposure.mean(),
            "turnover_times": float(log.turnover.sum()),
            "order_days": int(log.changed.sum()),
            "first_open": str(equity.index[0].date()),
            "last_open": str(equity.index[-1].date()),
            "intervals": len(log)}


def record(store, market, label, equity, log, period="test"):
    equity = equity.rename("equity")
    store.append({"market": market, "variant": label, "period": period, **stats(equity, log)})
    pd.concat([equity, (equity / equity.cummax() - 1).rename("drawdown")], axis=1).to_csv(
        OUT / f"{market}_{label}_{period}_equity.csv")
    log.to_csv(OUT / f"{market}_{label}_{period}_ledger.csv")


def curve_chart(curves, filename, title, drawdown=False):
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, equity in curves.items():
        y = equity / equity.cummax() - 1 if drawdown else equity
        ax.plot(y.index, y.values, label=label, linewidth=2)
    if drawdown:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title=title, xlabel="Date", ylabel="Drawdown" if drawdown else "Growth of 1 dollar")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(HERE / filename, dpi=180)
    plt.close(fig)








def finish(rows, extras=None):
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob("*.json")}
    report = {"python": platform.python_version(), "pandas": pd.__version__,
              "numpy": np.__version__, "matplotlib": matplotlib.__version__,
              "cost_per_side": COST, "data_sha256": hashes, "extras": extras or {}}
    (OUT / "run_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(frame[["market", "variant", "period", "annual_return_pct", "max_drawdown_pct"]].round(3).to_string(index=False))


from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import sklearn


def features(data):
    c = data.close
    return pd.DataFrame({"return_5": c.pct_change(5), "return_20": c.pct_change(20),
        "return_60": c.pct_change(60), "volatility_20": c.pct_change().rolling(20).std(),
        "distance_200": c / c.rolling(200).mean() - 1,
        "range_20": ((data.high-data.low)/c).rolling(20).mean()})


def rotation_orders(data):
    names = list(data)
    dates = data[names[0]].index
    xs = {s:features(d) for s,d in data.items()}
    # At close t, target measures next open through the open 21 sessions later.
    ys = {s:d.open.shift(-22)/d.open.shift(-1)*(1-COST)/(1+COST)-1 for s,d in data.items()}
    orders = {k:pd.DataFrame(np.nan,index=dates,columns=names)
              for k in ["Gradient boosting","Past 60 sessions","Equal weight monthly"]}
    schedule = pd.Series(dates,index=dates).groupby(dates.to_period("M")).last()
    predictions = []
    for day in schedule:
        i = dates.get_loc(day)
        if day < pd.Timestamp("2016-12-01") or i+1 >= len(dates):
            continue
        # Purge 23 recent feature rows: every training target ends before today.
        training_dates = dates[:i-22]
        train_x = pd.concat([xs[s].reindex(training_dates) for s in names],ignore_index=True)
        train_y = pd.concat([ys[s].reindex(training_dates) for s in names],ignore_index=True)
        valid = train_x.notna().all(axis=1) & train_y.notna()
        model = HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,
            min_samples_leaf=40,learning_rate=0.05,l2_regularization=5,
            early_stopping=False,random_state=42)
        with threadpool_limits(limits=1):
            model.fit(train_x.loc[valid],train_y.loc[valid])
            forecasts = pd.Series(model.predict(pd.DataFrame([xs[s].loc[day] for s in names])),index=names)
        chosen = forecasts.idxmax()
        momentum = pd.Series({s:xs[s].loc[day,"return_60"] for s in names}).idxmax()
        execution = dates[i+1]
        orders["Gradient boosting"].loc[execution] = [float(s==chosen) for s in names]
        orders["Past 60 sessions"].loc[execution] = [float(s==momentum) for s in names]
        orders["Equal weight monthly"].loc[execution] = 1/len(names)
        for s in names:
            predictions.append({"signal_date":day,"execution_date":execution,"market":s,
                "forecast":forecasts[s],"selected":s==chosen,"momentum_selected":s==momentum,
                "realized_21_session_net":ys[s].loc[day],"training_rows":int(valid.sum()),
                "latest_training_feature":training_dates[-1],
                "latest_training_label_open":dates[i-1]})
    return orders,pd.DataFrame(predictions)


def main():
    data = {s:load(s) for s in ["SPY","EFA","TLT","GLD"]}
    opens = pd.DataFrame({s:d.open for s,d in data.items()})
    orders,predictions = rotation_orders(data)
    predictions.to_csv(OUT/"monthly_predictions.csv",index=False)
    rows=[];curves={}
    for period,start in [("test",START),("late","2022-01-01")]:
        for label,order in orders.items():
            # On a split boundary, enter the most recently selected allocation.
            current = order.copy()
            first = opens.loc[start:].index[0]
            current.loc[first] = current.loc[:first].ffill().iloc[-1]
            eq,log=simulate(opens,current,start)
            record(rows,"Four ETFs",label,eq,log,period)
            if period=="test":curves[label]=eq
        for market,price in [("Four ETFs",opens),("SPY",data["SPY"].open)]:
            eq,log=buy_hold(price,start)
            label="Equal weight buy and hold" if market=="Four ETFs" else "Buy and hold"
            record(rows,market,label,eq,log,period)
            if period=="test":curves["SPY buy and hold" if market=="SPY" else label]=eq
    curve_chart({k:v for k,v in curves.items() if k!="Equal weight monthly"},"equity_curve.png","Monthly ETF rotation: the out-of-sample test")
    curve_chart({k:v for k,v in curves.items() if k in ["Gradient boosting","Past 60 sessions","Equal weight buy and hold"]},"drawdown.png","Rotation: opening-to-opening drawdowns",True)
    p=predictions.query("selected").copy();p["signal_date"]=pd.to_datetime(p.signal_date)
    p=p.loc[p.signal_date>="2017-01-01"]
    fig,ax=plt.subplots(figsize=(9,6));mapping={s:i for i,s in enumerate(data)}
    ax.scatter(p.signal_date,p.market.map(mapping),s=30,color="#24476c")
    ax.set_yticks(list(mapping.values()),list(mapping));ax.set(title="The ETF selected each month",xlabel="Signal date",ylabel="Selected market")
    ax.grid(alpha=0.2);fig.autofmt_xdate();fig.tight_layout();fig.savefig(HERE/"monthly_selections.png",dpi=180);plt.close(fig)
    finish(rows,{"sklearn":sklearn.__version__,"training":"expanding, monthly, 23-row purge; no shuffled validation",
                 "target":"21-session next-open return; actual holdings last until next monthly rebalance"})

if __name__ == "__main__":
    main()
