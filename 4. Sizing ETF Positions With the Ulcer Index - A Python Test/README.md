# 4. Sizing ETF Positions With the Ulcer Index - A Python Test

I compare drawdown-based sizing with volatility sizing. I label the matched-exposure controls as hindsight.

I use Python 3.11. From this folder, I install the packages and run the file:

```bash
python -m pip install numpy==1.26.4 pandas==2.3.3 requests==2.34.2
python experiment.py
```

I fetch market data when the file runs, so you need an internet connection. I print the comparisons in the terminal and do not create report files.

I charge 0.1% per purchase or sale in the main comparison, including buy and hold. I keep cash at zero yield and use no borrowing or short selling. I use 2017 to 2026 for the main results; the dates and parameters are near the top of the file.
