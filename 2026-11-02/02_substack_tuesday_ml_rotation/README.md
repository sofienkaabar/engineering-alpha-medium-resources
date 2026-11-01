# Can Technical Indicators Help a Model Rotate Between ETFs?

A price-based ranking model takes on simple relative strength, with monthly decisions and net-of-cost portfolio results.

Article by Sofien Kaabar for Engineering Alpha, Tuesday, November 3, 2026.

## Run it

From this folder:

```bash
python -m pip install -r requirements.txt
python experiment.py
```

Or, from the repository root after installing the root requirements:

```bash
python run_all.py --article 2
```

## Inputs and outputs

The `data/` folder includes the exact market responses used in the research run. With those files present, the experiment runs offline. If an expected cache file is absent, the script attempts to populate it over the network using its fixed historical date request. Restoring the supplied cache preserves the recorded input snapshot.

Running the script writes the chart PNGs in this folder and the CSV/JSON results in `results/`. Those outputs are also included for comparison. The main performance period is the January 3, 2017 open through the October 30, 2026 open.

## Cost and timing assumptions

Each purchase and sale costs 0.1% of actual traded notional in the main comparison. All strategy and buy-and-hold account performance includes costs. Cash earns zero; taxes and fixed minimum commissions are excluded. No strategy shorts or borrows. Drawdowns use opening valuations, rather than intraday extremes.

The model uses expanding monthly training with a 23-feature-row purge. Forward labels include a purchase and sale charge. The target spans 21 sessions after the next open; actual holdings last until the next monthly rebalance. This experiment takes longer because it refits the model at each monthly decision.

The article explains the experiment and its limitations. The code and ledgers make the tested rules inspectable.
