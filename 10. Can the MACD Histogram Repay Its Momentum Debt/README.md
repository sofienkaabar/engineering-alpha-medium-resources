# Can the MACD Histogram Repay Its Momentum Debt?

I test this idea with my standalone `experiment.py`. I use Python 3.11. I install the packages and run it from this folder:

```powershell
python -m pip install numpy==1.26.4 pandas==2.3.3 requests==2.34.2
python experiment.py
```

I need an internet connection for the daily prices. I print my numerical comparisons in the terminal; I do not save article text, images, data or reports.

I charge two basis points per purchase or sale and a fixed 3% annual holding charge on invested FX capital, accrued over calendar days. I apply the same charges to my passive comparison. I use this as a research cost scenario, not a reconstruction of historical broker swaps. I value each unlevered account in its quote currency. I calculate the indicators from observed daily reference rates, use the following available reference rate as the execution proxy, and I liquidate at the final evaluation reference rate. I assign zero return to idle cash. I measure drawdowns at reference-rate observations.

I evaluate January 2017 through October 30, 2026, with an additional later chronological comparison. I use earlier history for indicator warm-up or model training. I treat these as retrospective research results, not a genuinely unseen holdout or a promise of future returns.

I use official daily reference rates without interpolating missing observations. I use no OHLC fields and claim no executable bid/ask history. I may obtain different numbers if the source revises its daily rates.
