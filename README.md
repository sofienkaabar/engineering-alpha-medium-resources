# Engineering Alpha and Medium resources

Python experiments and charts accompanying articles by **Sofien Kaabar**. Each article has its own folder, full runnable script, requirements, saved market inputs and generated results.

## Run one experiment

Use Python 3.11. The recorded research environment used Python 3.11.7. Download this repository with GitHub's **Code > Download ZIP** and extract it, or clone it. Open a terminal in the extracted repository folder.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_all.py --article 7
```

On macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_all.py --article 7
```

The example runs the Ichimoku experiment. Change the article number to any number from 1 to 7. Run `python run_all.py --all` with the same environment to execute every experiment.

## November 2 to 6, 2026

1. [Can a Moving Average Rescue an Overnight Trading Strategy?](2026-11-02/01_substack_monday_overnight/README.md)

   I split daily candles into two tradable sessions, then tested a trend filter with every entry and exit charged to the account.

2. [Can Technical Indicators Help a Model Rotate Between ETFs?](2026-11-02/02_substack_tuesday_ml_rotation/README.md)

   A price-based ranking model takes on simple relative strength, with monthly decisions and net-of-cost portfolio results.

3. [Does a 1% Buffer Improve the 200-Day Moving Average Rule?](2026-11-02/03_substack_thursday_buffer/README.md)

   A quieter moving-average strategy still has to earn its place: I tested the same entry and exit band on stocks, Treasuries and gold.

4. [Sizing ETF Positions With the Ulcer Index: A Python Test](2026-11-02/04_substack_friday_ulcer/README.md)

   I turned the depth of recent drawdowns into an objective allocation rule and compared it with volatility sizing after costs.

5. [Does Waiting for a Rebound Improve a Five-Day-Low Strategy?](2026-11-02/05_medium_monday_rebound/README.md)

   A positive close feels like useful confirmation. I tested whether it improves a fixed five-session reversal trade after costs.

6. [Do Narrow Inside Bars Make Better Trading Signals?](2026-11-02/06_medium_wednesday_inside_nr7/README.md)

   I combined an objective inside-bar definition with a seven-session range filter and tested the resulting long entries after costs.

7. [How Far Above the Ichimoku Cloud Should You Buy?](2026-11-02/07_medium_friday_ichimoku/README.md)

   I built a volatility-adjusted cloud-clearance indicator and tested whether demanding a stronger breakout improves the strategy after costs.

## Find the results

Each experiment prints a compact summary and writes `results/summary.csv`, account equity files, trade ledgers and a run manifest. Charts sit next to `experiment.py`. Additional files include monthly model choices, indicator values, signal counts or cost scenarios where relevant.

The saved market inputs allow the supplied results to be reproduced without fetching a revised historical series. If you choose to refresh the inputs, adjusted historical prices may change.

## Troubleshooting

- Use the same Python executable for installing requirements and running scripts.
- If you see `ModuleNotFoundError`, install that folder's requirements with `python -m pip install -r requirements.txt`.
- Run the code as the supplied `.py` script; it uses its own location to find inputs and write outputs.
- If the script tries to download data unexpectedly, check that the article's `data/` folder was extracted with the repository.
- A failed trading idea is still a valid reproduced result. The reports retain losses and weaker comparisons.
