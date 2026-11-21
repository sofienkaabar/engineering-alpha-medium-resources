# Can a Ranking Model Pick Stocks Better Than Simple Momentum?

I run my standalone calculation with Python 3.11:

```powershell
python -m pip install numpy==1.26.4 pandas==2.3.3 requests==2.34.2 xgboost==2.1.1 threadpoolctl==3.6.0
python experiment.py
```

I need an internet connection. I print my comparisons in the terminal and keep the downloaded data in memory. I create no article, chart, data or report files.

I rebalance only at later openings, use matured training groups and charge 0.1% on actual purchase and sale amounts. I study the named fixed stock panel, not a survivorship-free market universe.

I keep the markets, periods and rules fixed in the script. I may see revised results if historical records change.
