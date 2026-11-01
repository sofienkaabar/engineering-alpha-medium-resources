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


def price_chart(frame, entry, filename, title, line=None):
    sample = frame.loc["2022-01-01":"2023-12-31"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sample.index, sample.close, color="#24476c", label="Adjusted close", linewidth=1.7)
    if line is not None:
        ax.plot(sample.index, line.reindex(sample.index), color="#d18c20", label="200-session average")
    chosen = entry.reindex(sample.index).fillna(False).astype(bool)
    ax.scatter(sample.index[chosen], sample.close[chosen], marker="^", color="#008779", s=45, label="Signal at close")
    ax.set(title=title, xlabel="Date", ylabel="Adjusted price (USD)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
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


def main():
    rows=[];chart_curves={};cost_rows=[]
    for symbol in ["SPY","QQQ","IWM"]:
        data=load(symbol)
        # The previous completed close is known before both today's open
        # and today's scheduled closing order. No same-close signal/fill.
        trend=data.close.gt(data.close.rolling(200).mean())
        prior_trend=trend.shift(1,fill_value=False)
        for period,start,stop in [("development","2011-01-01","2016-12-30"),("test",START,END),("late","2022-01-01",END)]:
            d=data.loc[start:stop]
            active=prior_trend.reindex(d.index).iloc[:-1]
            ratios={"Overnight":d.open.shift(-1)/d.close,"Daytime":d.close/d.open}
            for label,ratio in ratios.items():
                for filtered in [False,True]:
                    name=label+" trend filter" if filtered else label
                    mask=active if filtered else pd.Series(True,index=active.index)
                    net=(ratio.iloc[:-1]*(1-COST)/(1+COST)-1).where(mask,0.0)
                    eq=pd.Series(np.r_[1.0,(1+net).cumprod()],index=d.index,name="equity")
                    log=pd.DataFrame({"net_return":net,"exposure":mask.astype(float),
                        "turnover":((1+ratio.iloc[:-1])/(1+COST)).where(mask,0),"changed":mask})
                    record(rows,symbol,name,eq,log,period)
                    if period=="test" and symbol=="SPY" and name in ["Overnight","Overnight trend filter","Daytime"]:
                        chart_curves[name]=eq
            eq,log=buy_hold(data.open,start,stop);record(rows,symbol,"Buy and hold",eq,log,period)
            if period=="test" and symbol=="SPY":chart_curves["Buy and hold"]=eq
        d=data.loc[START:END];active=prior_trend.reindex(d.index).iloc[:-1]
        for fee in [0.0001,0.0005,0.001]:
            for label,ratio in [("Overnight",d.open.shift(-1)/d.close),("Daytime",d.close/d.open)]:
                for filtered in [False,True]:
                    name=label+" trend filter" if filtered else label
                    mask=active if filtered else pd.Series(True,index=active.index)
                    net=(ratio.iloc[:-1]*(1-fee)/(1+fee)-1).where(mask,0)
                    eq=pd.Series(np.r_[1.0,(1+net).cumprod()],index=d.index)
                    years=(d.index[-1]-d.index[0]).days/365.25
                    cost_rows.append({"market":symbol,"variant":name,"fee_bps":fee*10000,
                        "annual_return_pct":100*(eq.iloc[-1]**(1/years)-1),
                        "max_drawdown_pct":100*(eq/eq.cummax()-1).min()})
        if symbol=="SPY":
            price_chart(data,trend&~trend.shift(1,fill_value=False),"price_signals.png",
                        "SPY: the completed-close trend filter",data.close.rolling(200).mean())
    pd.DataFrame(cost_rows).to_csv(OUT/"cost_sensitivity.csv",index=False)
    curve_chart(chart_curves,"equity_curve.png","SPY: daily session strategies after costs")
    curve_chart(chart_curves,"drawdown.png","SPY: opening-valuation account drawdowns",True)
    comparison_chart(rows,"market_comparison.png","Does a trend filter rescue overnight trading?",["Overnight","Overnight trend filter","Buy and hold"])
    fig,ax=plt.subplots(figsize=(9,6));c=pd.DataFrame(cost_rows).query("market == 'SPY'")
    for label in ["Overnight","Overnight trend filter","Daytime"]:
        group=c.loc[c.variant==label];ax.plot(group.fee_bps,group.annual_return_pct,marker="o",label=label)
    ax.axhline(0,color="black",linewidth=0.8)
    ax.set(title="SPY: all scenarios include trading costs",xlabel="Cost per side (basis points)",ylabel="Annualized net return (%)")
    ax.legend(frameon=False);ax.grid(alpha=0.2);fig.tight_layout();fig.savefig(HERE/"cost_sensitivity.png",dpi=180);plt.close(fig)
    finish(rows,{"signal_timing":"both session schedules use previous completed close versus its 200-session average",
                 "fee_scenarios_bps":[1,5,10],"exposure":"allocation during traded segments, not wall-clock exposure"})

if __name__=="__main__":main()
