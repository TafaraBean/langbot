# 75x Trading Framework

A Streamlit dashboard that connects to your MetaTrader 5 terminal and walks
through a Smart Money Concepts (SMC) style trading process step by step:
**Bias -> Narrative -> Point of Interest -> Confirmation -> Risk management**,
plus a trade journal, historical backtesting, a multi-symbol portfolio
tester, and a demo-account-only live trading tab. It is built around the
Callisto FX pre-trade checklist and high-probability-order-block criteria
(liquidity sweep, FVG/displacement, higher-timeframe confluence, inducement).
The repo also contains a separate indicator dashboard (`dashboard.py`) and
two command-line backtests - see
[Section 6](#6-the-indicator-dashboard-and-command-line-backtests).

**Tabs 1-6 never place trades.** Everything they show - including
"backtests" and the "Mock Trade" results - is a simulation run against
historical price data, to help you study and evaluate a setup. You still
decide, and place, every trade yourself in MetaTrader 5. **Tab 7 (Live
Trading) is the exception:** once you arm it, it sends real orders to the
connected MetaTrader 5 account, and it refuses to arm unless that account is
a demo account (details in [Section 5](#5-what-each-tab-does)). This project
is for research and education - only ever run it against a demo account.

---

## 1. What you need before starting

- **Windows**, because the `MetaTrader5` Python package is Windows-only and
  only talks to a MetaTrader 5 terminal running on the same machine.
- **MetaTrader 5** (the desktop terminal app) installed, with a broker
  account logged in (a free demo account is fine for Tabs 1-6, and Tab 7
  will only trade on one).
- **Python 3.10 or newer, 64-bit** (this project was set up with Python 3.14).

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

**Other things you can run** (same activated terminal, MetaTrader 5 still
open and logged in):

```powershell
streamlit run dashboard.py               # separate indicator dashboard (see Section 6)
python backtest_orderblocks.py XAUUSDm   # order-block backtest, prints results to the terminal
python backtest_kde.py XAUUSDm           # volume-profile (KDE) peak backtest, prints results
```

The symbol argument is optional (it defaults to `XAUUSDm`). If `app.py` is
already running, `dashboard.py` opens on the next free port (e.g.
`http://localhost:8502`). The two backtest scripts may print harmless
Streamlit warnings (such as "No runtime found"), because they reuse the
app's cached data helpers outside of Streamlit.

## 4. First-run checklist

- In the **sidebar**, set **Symbol** to your broker's exact name for the
  instrument you want (e.g. `XAUUSDm`, `EURUSDm` - many brokers add a
  suffix like `m` to the plain name). You can find the exact name in MT5's
  Market Watch panel.
- The sidebar should show **"MT5 connected"** in green. If it shows an
  error instead, see [Troubleshooting](#8-troubleshooting).
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
timeframe (H4, H1 or M15): order blocks, Fair Value Gaps (FVGs), and untested
session liquidity (Asian/London/NY highs and lows). Each order block/FVG gets
an "HP" (high-probability) score out of 4, based on whether it shows a
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
across the quality-passing POIs currently visible on your chosen symbol (up
to the 10 most recent), so you can see several candidate trades at once
instead of one at a time.

**Tab 6 - Portfolio.** The same pipeline again, but looped across a whole
list of symbols (stocks, currencies, crypto - edit the text boxes to match
what your broker offers) and over a much longer stretch of history, so you
get a larger, more representative sample instead of just a handful of
trades. Includes:
- Controls for how far back to look, how long to wait for a confirmation,
  and how many candidates to test per symbol.
- A **Point-in-time bias** option (on by default) that judges each POI
  against the bias as it was when that POI formed, instead of today's bias.
- A **Diagnostics** panel explaining exactly where candidates dropped out
  if a run finds few or no trades.
- A **Parameter optimization** panel that tests different TP-ladder shapes,
  profit-scaling plans, and minimum HP scores against each other, ranking
  them by risk-adjusted return (total R divided by the worst drawdown) -
  useful for tuning the strategy's mechanics without touching the Minimum
  R:R setting, which always stays fixed.

**Tab 7 - Live Trading.** The only part of the app that can place real
orders. It runs the same Bias -> POI -> Confirmation pipeline live on a fixed
list of symbols - `XAUUSDm`, `XAGUSDm` and `USTECm`, set in `SYMBOLS` in
`live_engine.py` (edit it if your broker names them differently, and make
sure all three are in MT5's Market Watch) - using M15 for POIs and M5 for
confirmations, and checks for a new signal each time an M5 candle closes.
When a fresh confirmation appears it sizes the position from your "Risk %
of balance per trade" and the stop distance, and places a market order with
the stop-loss attached. For FX/metal symbols it then
closes the TP1-TP5 ladder in stages as price reaches each level; for other
symbols (e.g. `USTECm`) a single take-profit sits on the order. When a
position closes, the realised result is written to `live_journal.csv` and
shown under **Live results**. Tab 7 has its own risk, TP-ladder, scale-out,
HP-minimum and Minimum R:R settings (it doesn't use the sidebar's Minimum
R:R); the defaults - Tight ladder, Back-loaded scale-out, HP 2+, R:R 2.0 -
come from an earlier Tab 6 optimization run. Safeguards built into the code:
- **Demo accounts only.** It checks the connected account's trade mode; on
  a real account the arming controls are disabled, and every cycle
  re-checks and refuses to trade if the account isn't a demo.
- **Two-step arming.** Tick "I confirm this is a demo account...", then
  switch on **ARM LIVE TRADING**. It always starts disarmed - a restart
  never silently resumes live trading.
- **Dry run when not armed.** The pipeline still runs every poll interval
  (15/20/30 seconds) and the action log shows what it *would* have done,
  but nothing is sent to MT5.
- **Flatten all & Disarm** closes every position the bot is tracking and
  disarms.
- Every order is tagged with the bot's own magic number, so it only ever
  manages its own positions - trades you place by hand are left alone. It
  holds at most one position per symbol, and never acts on the same signal
  twice (even after a restart).

Keep the app open (and armed) while a bot position is running: the
stop-loss - and, for non-FX symbols, the take-profit - sit on the order in
MT5, but the TP1-TP5 partial closes are sent by the app itself, and only
while Tab 7 is open in a browser and armed.

## 6. The indicator dashboard and command-line backtests

These are separate from the 75x Framework app and don't share its journal
or settings.

**`dashboard.py` - Indicator dashboard** (`streamlit run dashboard.py`). One
chart for any symbol and timeframe (M1 up to monthly) where you switch on
entry constraints - Bulls/Bears Power, RSI, MACD, a moving average
(EMA/SMA), ATR - and every bar where all the enabled constraints hold is
marked as a mock long or short entry (your choice in the sidebar). Nothing
is logged or saved - the chart is the whole output, and it refreshes itself
every 15 seconds. On top of that you can overlay:
- **Smart Money Concepts** - a port of LuxAlgo's "Smart Money Concepts"
  indicator (`smc.py`): swing/internal structure (BOS/CHoCH), order blocks,
  fair value gaps, equal highs/lows, premium/discount zones, strong/weak
  highs/lows, and previous daily/weekly/monthly highs and lows.
- **Order Block Detector** - a port of LuxAlgo's volume-pivot "Order Block
  Detector" (`order_blocks.py`).
- **Volume Profile** - a volume-weighted density (KDE) of price, with its
  busiest price levels drawn as lines (`volume_profile.py`).
- **KDE Peak Strategy - Timeframe Comparison** - a button that runs the
  `backtest_kde.py` strategy on every timeframe and ranks them. It can take
  a few minutes.

**`backtest_orderblocks.py`** (`python backtest_orderblocks.py [SYMBOL]`). A
no-lookahead, long-only backtest: buys the first tap of every bullish order
block (`smc.py`'s internal order blocks and `order_blocks.py`'s volume-pivot
ones), with the stop just below the block and a fixed 2R target. Defaults to M1 over up to 50,000 bars, and
prints win rate, average/total R, max drawdown and the full trade list.

**`backtest_kde.py`** (`python backtest_kde.py [SYMBOL]`). The same
entry/exit mechanics, but the zones are price levels where a volume-profile
KDE peaks - recomputed every 50 bars using only data up to that point, so
the backtest never sees the future. Prints a similar summary (without the
trade list).

## 7. Project structure

Built with Python, Streamlit (UI), Plotly (charts), pandas/numpy, and the
`MetaTrader5` package (price data and orders).

```
langbot/
├── app.py                   # Main app: the 75x Framework dashboard, Tabs 1-7 (streamlit run app.py)
├── structure.py             # Framework logic for app.py: swings, bias/narrative, BOS/CHoCH, order blocks,
│                            #   FVG/IFVG, engulfing, HP score, stops, TP ladder, trade simulation
├── sessions.py              # Asian/London/NY AM session windows (defined in GMT+8, shown in local time)
│                            #   and prior-session liquidity highs/lows
├── journal.py               # CSV trade journal: append rows, load, win-rate / average-R stats
├── live_engine.py           # Tab 7 live loop: reconcile, detect signal, size and enter, manage TP legs, journal
├── live_state.py            # Reads/writes live_state.json for the live engine
├── mt5_data.py              # MT5 connection and cached candle (OHLC) downloads - shared by both apps
├── mt5_trade.py             # MT5 orders: account info + demo check, lot sizing, market orders, partial/full closes
├── dashboard.py             # Separate app: indicator-constraint dashboard (streamlit run dashboard.py)
├── indicators.py            # RSI, MACD, EMA/SMA, ATR, Bulls/Bears Power and constraint checks (dashboard.py)
├── smc.py                   # Port of LuxAlgo "Smart Money Concepts" (dashboard.py, backtest_orderblocks.py)
├── order_blocks.py          # Port of LuxAlgo "Order Block Detector" (volume-pivot order blocks)
├── volume_profile.py        # Volume-weighted KDE price profile and peaks, incl. a no-lookahead walk-forward
├── backtest_orderblocks.py  # Command line: order-block tap backtest
├── backtest_kde.py          # Command line: KDE-peak tap backtest + all-timeframe sweep (used by dashboard.py)
├── requirements.txt         # Python dependencies
├── journal.csv              # Runtime file: trade journal (Tabs 3-5)
├── live_journal.csv         # Runtime file: closed live trades (Tab 7)
└── live_state.json          # Runtime file: live engine bookkeeping (Tab 7)
```

`structure.py` and `smc.py` are two independent SMC implementations:
`app.py` uses only the lighter heuristics in `structure.py`, while `smc.py`
is the line-by-line LuxAlgo port used by the indicator dashboard and the
order-block backtest.

**Runtime files.** The app creates and updates these three files itself, in
the folder you launch it from (so always start it from this project folder):
- `journal.csv` - created (header row only) the first time the app runs
  with MT5 connected; a row is added every time you log a trade in Tab 3, 4
  or 5.
- `live_journal.csv` - same columns, also created on that first run; a row
  is added each time a live position closes.
- `live_state.json` - rewritten every Tab 7 poll while the app is open
  (armed or not): the last processed M5 candle per symbol, the signals
  already acted on, and any open bot position with the TP legs already
  closed. Don't delete or hand-edit it while the bot has a position open -
  it's how the engine keeps track of that position and which TP legs it has
  already taken.

## 8. Troubleshooting

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

**Tab 7's confirm box and ARM toggle are greyed out.** The connected
account isn't a demo account (or MT5 isn't logged in) - the red message at
the top of Tab 7 says which. Log into a demo account in MT5 and reload the
page.

**Nothing shows up in journal.csv.** It's created with just a header row
the first time the app runs with MT5 connected, and a row is added each
time you log a trade (**Save trade** in Tab 4, or a "Log ... to journal"
button in Tab 3 or 5). It's written to the folder you started Streamlit
from, so if it seems to be missing, check that you ran
`streamlit run app.py` from this project folder.
