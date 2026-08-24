# 75x Trading Framework

A Streamlit dashboard that connects to your MetaTrader 5 terminal and walks
through a Smart Money Concepts (SMC) style trading process step by step:
**Bias -> Narrative -> Point of Interest -> Confirmation -> Risk management**,
plus a trade journal, historical backtesting, and a multi-symbol portfolio
tester. It is built around the Callisto FX pre-trade checklist and
high-probability-order-block criteria (liquidity sweep, FVG/displacement,
higher-timeframe confluence, inducement).

**This tool does not place real trades.** Everything it shows - including
"backtests" and the "Mock Trade" results - is a simulation run against
historical price data, to help you study and evaluate a setup. You still
decide, and place, every trade yourself in MetaTrader 5.

---

## 1. What you need before starting

- **Windows**, because the MetaTrader 5 Python library only talks to a
  MetaTrader 5 terminal running on the same machine.
- **MetaTrader 5** (the desktop terminal app) installed, with a broker
  account logged in (a free demo account is fine).
- **Python 3.10 or newer** (this project was set up with Python 3.14).

If you already have a `.venv` folder inside this project, Python and the
required packages are likely already installed - skip to [Section 3](#3-running-the-app).

## 2. One-time setup

Open a terminal (PowerShell) in this project folder and run:

```powershell
# 1. Create a virtual environment (a private, isolated copy of Python for this project)
python -m venv .venv

# 2. Activate it
.venv\Scripts\Activate.ps1

# 3. Install everything the app needs
pip install -r requirements.txt
```

You only need to do this once. If PowerShell blocks the activation script
with a "running scripts is disabled" error, open PowerShell as Administrator
and run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try
again.

## 3. Running the app

Every time you want to use the dashboard:

1. **Open MetaTrader 5** and make sure you're logged into your broker
   account (you'll see live prices moving in the Market Watch panel).
2. Open a terminal in this project folder and run:
   ```powershell
   .venv\Scripts\Activate.ps1
   streamlit run app.py
   ```
3. A browser tab opens automatically at `http://localhost:8501` with the
   dashboard. If it doesn't open on its own, copy that address into your
   browser manually.
4. To stop the app, go back to the terminal and press `Ctrl+C`.

**If you edit the code while the app is running:** Streamlit usually detects
the change and offers a "Rerun" button in the browser. If something behaves
oddly after an edit (an error mentioning a function that should exist),
stop the app (`Ctrl+C`) and run `streamlit run app.py` again - that always
guarantees the latest code is used.

## 4. First-run checklist

- In the **sidebar**, set **Symbol** to your broker's exact name for the
  instrument you want (e.g. `XAUUSDm`, `EURUSDm` - many brokers add a
  suffix like `m` to the plain name). You can find the exact name in MT5's
  Market Watch panel.
- The sidebar should show **"MT5 connected"** in green. If it shows an
  error instead, see [Troubleshooting](#6-troubleshooting).
- **Minimum R:R** is the smallest reward-to-risk ratio the framework will
  accept for a trade target (the framework's own rule is "at least 1:1").
  You can leave this at the default.

## 5. What each tab does

The tabs follow the framework top to bottom - each one builds on the last.

**Tab 1 - Bias & Narrative.** Shows the trend direction (Bullish/Bearish/
Unclear) on the Daily, 4-Hour, 1-Hour, and 15-Minute charts, and how many of
them agree ("narrative score"). Higher timeframes matter more - the rule is
never to trade without knowing this first.

**Tab 2 - Point of Interest (POI).** Finds candidate zones on your chosen
timeframe: order blocks, Fair Value Gaps (FVGs), and untested session
liquidity (Asian/London/NY highs and lows). Each order block/FVG gets an
"HP" (high-probability) score out of 4, based on whether it shows a
liquidity sweep, an FVG in its impulse leg, agreement with the higher
timeframe bias, and inducement. Pick one as your **active POI** for the
current setup.

**Tab 3 - Confirmation.** Once price has tapped your active POI, this tab
watches a lower timeframe (M1/M3/M5) for a CHoCH, an Inverse FVG, or an
engulfing candle - in that order of preference - as the trigger to actually
enter. It also runs a "Mock Trade": a simulated backtest of what would have
happened entering right after that confirmation, including the Callisto
pip-based TP1-TP5 ladder with partial profit-taking for FX/metal symbols.

**Tab 4 - Setup & Journal.** A checklist confirming all four pillars (Bias,
Narrative, POI, Confirmation) are in place before you'd call it a valid
setup, plus a form to manually log trades and a running journal with win
rate / average R statistics, backed by `journal.csv` in this folder.

**Tab 5 - POI Backtest Results.** Automatically runs the Tab 2/3 process
across every quality-passing POI currently visible on your chosen symbol,
so you can see several candidate trades at once instead of one at a time.

**Tab 6 - Portfolio.** The same pipeline again, but looped across a whole
list of symbols (stocks, currencies, crypto - edit the text boxes to match
what your broker offers) and over a much longer stretch of history, so you
get a larger, more representative sample instead of just a handful of
trades. Includes:
- Controls for how far back to look, how long to wait for a confirmation,
  and how many candidates to test per symbol.
- A **Diagnostics** panel explaining exactly where candidates dropped out
  if a run finds few or no trades.
- A **Parameter optimization** panel that tests different TP-ladder shapes,
  profit-scaling plans, and minimum HP scores against each other, ranking
  them by risk-adjusted return (total R divided by the worst drawdown) -
  useful for tuning the strategy's mechanics without touching the Minimum
  R:R setting, which always stays fixed.

## 6. Troubleshooting

**Sidebar shows an MT5 connection error.** Make sure the MetaTrader 5
terminal is open and you're logged in, and that the Symbol name in the
sidebar exactly matches Market Watch (including any broker suffix).

**"Symbol not found" / a portfolio symbol gets skipped.** That ticker
doesn't exist under that exact name with your broker - open MT5's Market
Watch, search for the instrument, and copy its exact name into the
dashboard.

**A Tab 6 run finds no trades.** Open the Diagnostics panel it shows - it
tells you whether candidates failed the quality filter, were never tapped
by price, or were tapped but never confirmed. Try a larger "Confirmation
window (days)" or a smaller "POI lookback (bars)" first.

**A run feels very slow.** Large lookbacks and candidate counts in Tab 6
mean more historical data to fetch and test - start with smaller values
(the defaults are already tuned to be reasonable) and increase gradually.

**An error mentions a function/argument that "shouldn't" be missing.**
This almost always means the running app is using an old copy of the code
from before a recent edit. Stop it (`Ctrl+C`) and run
`streamlit run app.py` again.

**Nothing shows up in journal.csv.** It's created automatically the first
time you log a trade (manually in Tab 4, or via a "Log to journal" button
elsewhere) - it won't exist before that.
