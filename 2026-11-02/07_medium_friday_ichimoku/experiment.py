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


def ichimoku(data):
    def midpoint(window):
        return (data.high.rolling(window).max()+data.low.rolling(window).min())/2
    tenkan=midpoint(9);kijun=midpoint(26)
    span_a=((tenkan+kijun)/2).shift(26)
    span_b=midpoint(52).shift(26)
    top=pd.concat([span_a,span_b],axis=1).max(axis=1,skipna=False)
    bottom=pd.concat([span_a,span_b],axis=1).min(axis=1,skipna=False)
    true_range=pd.concat([data.high-data.low,(data.high-data.close.shift(1)).abs(),
                         (data.low-data.close.shift(1)).abs()],axis=1).max(axis=1)
    atr=true_range.rolling(14).mean()  # Simple average, not Wilder smoothing.
    clearance=(data.close-top)/atr
    return pd.DataFrame({"tenkan":tenkan,"kijun":kijun,"span_a":span_a,"span_b":span_b,
        "top":top,"bottom":bottom,"atr":atr,"clearance":clearance})


def cloud_signal(data,minimum_clearance):
    lines=ichimoku(data);state=0.0;values=[]
    for day in data.index:
        row=lines.loc[day]
        if pd.notna(row.clearance):
            if data.loc[day,"close"]<row.bottom:state=0.0
            elif row.clearance>minimum_clearance:state=1.0
        values.append(state)
    return pd.Series(values,index=data.index)


def main():
    rows=[];curves={}
    for symbol in ["SPY","TLT","GLD"]:
        d=load(symbol);lines=ichimoku(d);lines.to_csv(OUT/f"{symbol}_indicator.csv")
        for label,clearance in [("Cloud breakout",0.0),("Half-ATR clearance",0.5)]:
            signal=cloud_signal(d,clearance);order=orders_from_signal(signal)
            for period,start,stop in [("development","2011-01-01","2016-12-30"),("test",START,END),("late","2022-01-01",END)]:
                eq,log=simulate(d.open,order,start,stop);record(rows,symbol,label,eq,log,period)
                if period=="test" and symbol=="SPY":curves[label]=eq
        for period,start,stop in [("development","2011-01-01","2016-12-30"),("test",START,END),("late","2022-01-01",END)]:
            eq,log=buy_hold(d.open,start,stop);record(rows,symbol,"Buy and hold",eq,log,period)
            if period=="test" and symbol=="SPY":curves["Buy and hold"]=eq
        if symbol=="SPY":
            sample=d.loc["2022-01-01":"2023-12-31"];visible=lines.reindex(sample.index)
            signal=cloud_signal(d,0.5);entries=signal.diff().eq(1).reindex(sample.index)
            fig,ax=plt.subplots(figsize=(9,6));ax.plot(sample.index,sample.close,color="#24476c",label="Adjusted close",linewidth=1.7)
            ax.fill_between(sample.index,visible.bottom,visible.top,color="#b5cddc",alpha=0.65,label="Cloud visible on that date")
            ax.plot(sample.index,visible.top+0.5*visible.atr,color="#d18c20",linewidth=1.2,label="Half-ATR entry boundary")
            ax.scatter(sample.index[entries],sample.close[entries],marker="^",color="#008779",s=55,label="Entry signal at close")
            ax.set(title="SPY: price must clear the cloud",xlabel="Date",ylabel="Adjusted price (USD)")
            ax.legend(frameon=False,fontsize=12);ax.grid(alpha=0.2);fig.autofmt_xdate();fig.tight_layout();fig.savefig(HERE/"price_cloud.png",dpi=180);plt.close(fig)
    curve_chart(curves,"equity_curve.png","SPY: Ichimoku entries after trading costs")
    curve_chart(curves,"drawdown.png","SPY: Ichimoku account drawdowns",True)
    comparison_chart(rows,"market_comparison.png","The same cloud-clearance rule in three markets",["Cloud breakout","Half-ATR clearance","Buy and hold"])
    finish(rows,{"periods":[9,26,52],"cloud_displacement_sessions":26,"atr_window":14,
        "atr_smoothing":"simple rolling mean","primary_clearance_atr":0.5,
        "exit":"close below current visible cloud bottom; next-open execution",
        "no_future_features":"lagging Chikou plotting series is not calculated or used"})

if __name__=="__main__":main()
