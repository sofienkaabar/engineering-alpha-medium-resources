# How Far Above the Ichimoku Cloud Should You Buy?

I built a volatility-adjusted cloud-clearance indicator and tested whether demanding a stronger breakout improves the strategy after costs.

Article by Sofien Kaabar for Medium, Friday, November 6, 2026.

## Run it

From this folder:

```bash
python -m pip install -r requirements.txt
python experiment.py
```

Or, from the repository root after installing the root requirements:

```bash
python run_all.py --article 7
```

## Inputs and outputs

The `data/` folder includes the exact market responses used in the research run. With those files present, the experiment runs offline. If an expected cache file is absent, the script attempts to populate it over the network using its fixed historical date request. Restoring the supplied cache preserves the recorded input snapshot.

Running the script writes the chart PNGs in this folder and the CSV/JSON results in `results/`. Those outputs are also included for comparison. The main performance period is the January 3, 2017 open through the October 30, 2026 open.

## Cost and timing assumptions

Each purchase and sale costs 0.1% of actual traded notional in the main comparison. All strategy and buy-and-hold account performance includes costs. Cash earns zero; taxes and fixed minimum commissions are excluded. No strategy shorts or borrows. Drawdowns use opening valuations, rather than intraday extremes.

The cloud uses 9/26/52-session range midpoints and a 26-session forward displacement. ATR is a simple 14-session average of true range. Entry clearance is zero or half an ATR; both variants exit below the visible cloud bottom. Close-based decisions execute at the following open.

The article explains the experiment and its limitations. The code and ledgers make the tested rules inspectable.
