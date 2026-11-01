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


def orders_from_signal(signal):
    # The close signal becomes an order at the following session's open.
    return signal.astype(float).shift(1).to_frame("asset")


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




def comparison_chart(rows, filename, title, labels):
    data = pd.DataFrame(rows)
    data = data.loc[(data.period == "test") & data.variant.isin(labels)]
    pivot = data.pivot(index="market", columns="variant", values="annual_return_pct").reindex(columns=labels)
    fig, ax = plt.subplots(figsize=(9, 6))
    pivot.plot.bar(ax=ax, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title=title, xlabel="Market", ylabel="Annualized return after costs (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False, fontsize=13)
    ax.grid(axis="y", alpha=0.2)
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


def sizing(data):
    c=data.close
    # Each historical close is compared with its own trailing 63-session peak.
    daily_drawdown=c/c.rolling(63).max()-1
    ulcer=daily_drawdown.pow(2).rolling(63).mean().pow(0.5)
    volatility=c.pct_change().rolling(20).std()*np.sqrt(252)
    weights=pd.DataFrame({"Ulcer sizing":(0.03/ulcer.clip(lower=0.0001)).clip(upper=1),
        "Volatility sizing":(0.10/volatility).clip(upper=1)})
    return weights,ulcer


def main():
    rows=[];curves={};weights_all=[]
    for symbol in ["SPY","TLT","GLD"]:
        d=load(symbol);weights,ulcer=sizing(d)
        weights.assign(market=symbol,ulcer_index=ulcer).to_csv(OUT/f"{symbol}_daily_weights.csv")
        for label in weights:
            order=orders_from_signal(weights[label])
            for period,start,stop in [("development","2011-01-01","2016-12-30"),("test",START,END),("late","2022-01-01",END)]:
                eq,log=simulate(d.open,order,start,stop);record(rows,symbol,label,eq,log,period)
                if period=="test" and symbol=="SPY":curves[label]=eq
        # Descriptive control: match average test exposure. It is deliberately
        # labelled hindsight and is not presented as an implementable rule.
        for label in weights:
            exposure=weights[label].shift(1).loc[START:END].iloc[:-1].mean()
            order=pd.DataFrame(exposure,index=d.index,columns=["asset"])
            eq,log=simulate(d.open,order)
            record(rows,symbol,"Hindsight constant: "+label,eq,log)
        for period,start,stop in [("development","2011-01-01","2016-12-30"),("test",START,END),("late","2022-01-01",END)]:
            eq,log=buy_hold(d.open,start,stop);record(rows,symbol,"Buy and hold",eq,log,period)
            if period=="test" and symbol=="SPY":curves["Buy and hold"]=eq
        if symbol=="SPY":
            fig,ax=plt.subplots(figsize=(9,6));sample=weights.loc["2022-01-01":"2023-12-31"]
            for label in sample:ax.plot(sample.index,sample[label],label=label)
            ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set(title="SPY: allocations decided at each close",xlabel="Date",ylabel="Target ETF allocation")
            ax.legend(frameon=False);ax.grid(alpha=0.2);fig.autofmt_xdate();fig.tight_layout();fig.savefig(HERE/"position_weights.png",dpi=180);plt.close(fig)
    curve_chart(curves,"equity_curve.png","SPY: two different ways to reduce exposure")
    curve_chart(curves,"drawdown.png","SPY: reduced exposure and drawdown",True)
    comparison_chart(rows,"market_comparison.png","Sizing rules across equities, bonds and gold",["Ulcer sizing","Volatility sizing","Buy and hold"])
    finish(rows,{"ulcer_window":63,"ulcer_budget":0.03,"annual_volatility_budget":0.10,
                 "note":"different budgets, not risk matched; matched-exposure control uses hindsight"})

if __name__ == "__main__":
    main()
