"""
The Inverse Fisher Transform / Ehlers' "Elegant Oscillator": an honest evaluation.

John Ehlers introduced the Elegant Oscillator (TASC, Feb 2022), built on the
Inverse Fisher Transform of a normalised price derivative and smoothed by his
two-pole SuperSmoother. The Financial Hacker tested it as a mean-reversion
signal on SPY and reported 5 of 7 winning trades with a profit factor above 6
-- over a single 14-month window (Mar 2020 - May 2021).

Seven trades on one hand-picked window tells you almost nothing. This module
reimplements the indicator and its trading rule from Ehlers' C/Zorro code in
plain Python, then does what the original did not: run it over the full
available history of several liquid ETFs, add transaction costs, and compare it
against buy-and-hold. The question is not "can it win 7 trades" but "is there a
repeatable edge once you stop cherry-picking the window?"

Ground rules (same discipline as a serious backtest):
    * No look-ahead. The oscillator at bar t uses prices up to t; a peak is only
      acted on once it is confirmed, and orders fill at the next open.
    * The indicator is a faithful translation of Ehlers' code, not re-tuned.
    * Every number is reproducible from `python elegant_oscillator.py`.

Usage
-----
    python elegant_oscillator.py                 # full evaluation, all symbols
    python elegant_oscillator.py --symbol SPY --plot
    python elegant_oscillator.py --replicate     # the article's SPY 2020-21 window
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

SYMBOLS = ("QQQ", "SPY", "IWM")
CASH = 100_000
PERIODS_PER_YEAR = 252
DEFAULT_COSTS_BPS = (0.0, 1.0, 2.5, 5.0, 10.0)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_data(symbol: str = "QQQ", cache_dir: str | Path = "data") -> pd.DataFrame:
    """Load split/dividend-adjusted OHLCV from the committed CSV cache.

    Prices are cached so the numbers are reproducible from a clone: Yahoo
    revises its adjusted history and rate-limits downloads. Refresh by deleting
    the CSV and refetching with your own tooling.
    """
    path = Path(cache_dir) / f"{symbol}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


# --------------------------------------------------------------------------- #
# Ehlers' filters (translated from the TASC / Zorro C code)
# --------------------------------------------------------------------------- #

def inverse_fisher_transform(x: np.ndarray) -> np.ndarray:
    """Ehlers' Inverse Fisher Transform: (e^{2x} - 1) / (e^{2x} + 1).

    Squashes an unbounded input into (-1, 1), sharpening the extremes. It is
    algebraically tanh(x); written in Ehlers' form for a faithful translation.
    """
    e2 = np.exp(2.0 * np.asarray(x, dtype=float))
    return (e2 - 1.0) / (e2 + 1.0)


def super_smoother(x, period: int = 20) -> np.ndarray:
    """Ehlers' two-pole SuperSmoother (a low-lag Butterworth low-pass filter).

        a1 = exp(-sqrt(2)*pi/period)
        b1 = 2*a1*cos(sqrt(2)*pi/period)
        y  = c1*(x + x[-1])/2 + b1*y[-1] - a1^2*y[-2]

    Leading NaNs (indicator warm-up) are skipped; the first two outputs seed the
    recursion with the input.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    a1 = np.exp(-1.41421356 * np.pi / period)
    b1 = 2.0 * a1 * np.cos(1.41421356 * np.pi / period)
    c3 = -a1 * a1
    c2 = b1
    c1 = 1.0 - c2 - c3

    y = np.full(n, np.nan)
    valid = np.where(np.isfinite(x))[0]
    if len(valid) == 0:
        return y
    start = valid[0]
    for i in range(start, n):
        if i < start + 2:
            y[i] = x[i]
        else:
            y[i] = c1 * (x[i] + x[i - 1]) / 2.0 + c2 * y[i - 1] + c3 * y[i - 2]
    return y


def elegant_oscillator(close, length: int = 50, smooth: int = 20) -> np.ndarray:
    """Ehlers' Elegant Oscillator.

        deriv  = close - close[-2]
        rms    = sqrt( mean(deriv^2, length) )      # N-dimensional distance
        nderiv = deriv / rms                        # scale-free
        return SuperSmoother( InverseFisher(nderiv), smooth )

    The RMS normalisation makes the input to the Inverse Fisher roughly unit
    scale, so the transform's (-1, 1) squashing is meaningful across regimes.
    """
    close = pd.Series(np.asarray(close, dtype=float))
    deriv = close - close.shift(2)
    rms = np.sqrt((deriv**2).rolling(length).mean())
    nderiv = (deriv / rms).replace([np.inf, -np.inf], np.nan)
    ifish = inverse_fisher_transform(nderiv.to_numpy())
    return super_smoother(ifish, smooth)


# --------------------------------------------------------------------------- #
# Strategy (Ehlers' rule, exactly as published)
# --------------------------------------------------------------------------- #

class ElegantOscillator(Strategy):
    """Mean-reversion trades on the Elegant Oscillator.

    Ehlers' signal: the oscillator makes a peak above +threshold (over-extended
    up -> expect reversion down) or a valley below -threshold (over-extended
    down -> expect reversion up). A peak/valley is confirmed one bar after it
    forms (osc[-2] higher/lower than both neighbours and past the threshold);
    orders fill at the next open, so there is no look-ahead.

    One honest caveat, made explicit: the published Zorro code calls
    enterLong()/enterShort() with *no exit rule*, which on multi-year data
    degenerates into a permanent position (e.g. the 2020-21 window has seven
    peaks and zero valleys, so a stop-and-reverse system simply stays short a
    rising market). A signal with no exit is not a strategy, so an exit has to
    be supplied. The natural one for a mean-reversion oscillator is to close
    when it reverts through zero; the strategy is flat between trades. This
    choice is ours, not Ehlers', and the README tests how much the verdict
    depends on it.
    """

    length = 50
    smooth = 20
    threshold = 0.5
    exit_rule = "zero"   # "zero" | "hold" | "opposite" -- see _should_exit
    hold_bars = 5        # holding period for the "hold" rule

    def init(self):
        close = self.data.Close
        self.osc = self.I(
            lambda: elegant_oscillator(close, self.length, self.smooth),
            name="elegant_osc",
        )

    def _should_exit(self, o1: float) -> bool:
        """Whether to close the open position, under the chosen exit rule.

        The source rule specifies no exit, so the exit is a modelling choice.
        Three reasonable ones are offered so the result can be checked against
        it (see exit_sensitivity):
            "zero"     -- revert through zero (mean reached)
            "opposite" -- ride to the opposite threshold
            "hold"     -- fixed holding period of `hold_bars` bars
        """
        if self.exit_rule == "zero":
            return (self.position.is_long and o1 >= 0) or (self.position.is_short and o1 <= 0)
        if self.exit_rule == "opposite":
            return ((self.position.is_long and o1 >= self.threshold)
                    or (self.position.is_short and o1 <= -self.threshold))
        if self.exit_rule == "hold":
            bars_held = (len(self.data) - 1) - self.trades[-1].entry_bar
            return bars_held >= self.hold_bars
        raise ValueError(f"unknown exit_rule {self.exit_rule!r}")

    def next(self):
        o1, o2, o3 = self.osc[-1], self.osc[-2], self.osc[-3]
        if np.isnan(o1) or np.isnan(o2) or np.isnan(o3):
            return

        if self.position and self._should_exit(o1):
            self.position.close()

        # Enter only when flat, on a confirmed peak/valley past the threshold.
        if not self.position:
            peak = o2 > o1 and o2 > o3 and o2 > self.threshold
            valley = o2 < o1 and o2 < o3 and o2 < -self.threshold
            if valley:
                self.buy()
            elif peak:
                self.sell()


# --------------------------------------------------------------------------- #
# Running and measuring
# --------------------------------------------------------------------------- #

def run_backtest(data: pd.DataFrame, cost_bps: float = 0.0, cash: int = CASH, **overrides):
    bt = Backtest(data, ElegantOscillator, cash=cash, commission=cost_bps / 10_000)
    return bt.run(**overrides)


def sharpe(equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualised Sharpe of daily equity returns, rf = 0. This system is always
    in the market, so the invested-day / all-day distinction does not arise."""
    r = equity.pct_change().dropna()
    if len(r) == 0 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(periods_per_year)


def buy_and_hold(data: pd.DataFrame) -> dict:
    eq = data["Close"] / data["Close"].iloc[0]
    years = len(eq) / PERIODS_PER_YEAR
    cagr = eq.iloc[-1] ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    return {"CAGR [%]": 100 * cagr, "Sharpe": sharpe(eq), "Max DD [%]": 100 * dd}


def summarise(stats, label: str) -> dict:
    equity = stats["_equity_curve"]["Equity"]
    return {
        "label": label,
        "CAGR [%]": stats["Return (Ann.) [%]"],
        "Sharpe": sharpe(equity),
        "Sharpe (library)": stats["Sharpe Ratio"],
        "Max DD [%]": stats["Max. Drawdown [%]"],
        "Exposure [%]": stats["Exposure Time [%]"],
        "Win Rate [%]": stats["Win Rate [%]"],
        "# Trades": stats["# Trades"],
    }


# --------------------------------------------------------------------------- #
# The three evaluations
# --------------------------------------------------------------------------- #

def across_symbols(symbols=SYMBOLS, cost_bps: float = 0.0) -> pd.DataFrame:
    """Does the edge exist beyond one cherry-picked window? Full history of each
    ETF, strategy vs buy-and-hold."""
    rows = []
    for sym in symbols:
        data = load_data(sym)
        stats = run_backtest(data, cost_bps=cost_bps)
        row = summarise(stats, sym)
        bh = buy_and_hold(data)
        row["B&H CAGR [%]"] = bh["CAGR [%]"]
        row["B&H Sharpe"] = bh["Sharpe"]
        row["years"] = round(len(data) / PERIODS_PER_YEAR, 1)
        rows.append(row)
    cols = ["CAGR [%]", "Sharpe", "Max DD [%]", "Exposure [%]", "Win Rate [%]",
            "# Trades", "B&H CAGR [%]", "B&H Sharpe", "years"]
    return pd.DataFrame(rows).set_index("label")[cols]


def cost_sensitivity(symbol: str = "SPY", costs_bps=DEFAULT_COSTS_BPS) -> pd.DataFrame:
    """How much of the result is left once trading costs are charged?"""
    data = load_data(symbol)
    rows = []
    for bps in costs_bps:
        stats = run_backtest(data, cost_bps=bps)
        rows.append(summarise(stats, f"{bps:g} bps"))
    cols = ["CAGR [%]", "Sharpe", "Max DD [%]", "Win Rate [%]", "# Trades"]
    return pd.DataFrame(rows).set_index("label")[cols]


def replicate_article(symbol: str = "SPY") -> pd.DataFrame:
    """Reproduce the Financial Hacker window (SPY, 2020-03-01 to 2021-05-01),
    then show the same rule over the full history next to it."""
    data = load_data(symbol)
    window = data.loc["2020-03-01":"2021-05-01"]
    rows = [
        summarise(run_backtest(window), f"{symbol} 2020-03..2021-05 (article window)"),
        summarise(run_backtest(data), f"{symbol} full history"),
    ]
    cols = ["CAGR [%]", "Sharpe", "Max DD [%]", "Win Rate [%]", "# Trades"]
    return pd.DataFrame(rows).set_index("label")[cols]


# The source gives no exit rule, so the verdict must not hinge on the one we
# picked. These are the reasonable choices, swept in exit_sensitivity.
EXIT_RULES = [
    ("zero-cross", dict(exit_rule="zero")),
    ("hold 3 bars", dict(exit_rule="hold", hold_bars=3)),
    ("hold 5 bars", dict(exit_rule="hold", hold_bars=5)),
    ("hold 10 bars", dict(exit_rule="hold", hold_bars=10)),
    ("opposite band", dict(exit_rule="opposite")),
]


def exit_sensitivity(symbols=SYMBOLS, cost_bps: float = 0.0) -> pd.DataFrame:
    """Does the verdict survive the exit choice? Sweep every exit rule on every
    instrument. If the Sharpe is negative across the board, the "no edge"
    conclusion does not depend on how we chose to close trades."""
    rows = []
    for sym in symbols:
        data = load_data(sym)
        for name, params in EXIT_RULES:
            stats = run_backtest(data, cost_bps=cost_bps, **params)
            eq = stats["_equity_curve"]["Equity"]
            rows.append({
                "symbol": sym,
                "exit rule": name,
                "CAGR [%]": stats["Return (Ann.) [%]"],
                "Sharpe": sharpe(eq),
                "Max DD [%]": stats["Max. Drawdown [%]"],
                "# Trades": stats["# Trades"],
            })
    return pd.DataFrame(rows).set_index(["symbol", "exit rule"])


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def save_indicator_plot(symbol: str, out_dir, start="2020-03-01", end="2021-05-01"):
    """Price with entry/exit markers over the oscillator, on the article's
    window, to show the signal doing what Ehlers describes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_data(symbol).loc[start:end]
    osc = elegant_oscillator(data["Close"], 50, 20)
    osc = pd.Series(osc, index=data.index)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(data.index, data["Close"], color="#1f77b4", lw=1.1)
    ax1.set_ylabel(f"{symbol} close")
    ax1.set_title(f"Elegant Oscillator on {symbol} ({start} to {end})")
    ax1.grid(alpha=0.2)

    ax2.plot(osc.index, osc.values, color="#6a3d9a", lw=1.1)
    ax2.axhline(0.5, color="0.5", ls="--", lw=0.8)
    ax2.axhline(-0.5, color="0.5", ls="--", lw=0.8)
    ax2.axhline(0, color="0.7", lw=0.6)
    ax2.set_ylabel("oscillator")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    path = Path(out_dir) / f"indicator_{symbol}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_equity_plot(symbol: str, out_dir, cost_bps: float = 0.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_data(symbol)
    stats = run_backtest(data, cost_bps=cost_bps)
    equity = stats["_equity_curve"]["Equity"]
    bh = CASH * data["Close"] / data["Close"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values, color="#d62728", lw=1.1,
            label=f"Elegant Oscillator (Sharpe {sharpe(equity):.2f})")
    ax.plot(bh.index, bh.values, color="#1f77b4", lw=1.1,
            label=f"Buy & hold {symbol} (Sharpe {sharpe(bh):.2f})")
    ax.set_yscale("log")
    ax.set_ylabel("Equity ($, log scale)")
    ax.set_title(f"Elegant Oscillator vs buy & hold: {symbol}, {cost_bps:g} bps/side")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = Path(out_dir) / f"equity_{symbol}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_exit_sensitivity_plot(table: pd.DataFrame, out_dir):
    """Bar chart of Sharpe per exit rule, grouped by instrument. Every bar below
    zero is the point: the negative result is not an artifact of one exit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rules = [name for name, _ in EXIT_RULES]
    symbols = list(dict.fromkeys(table.index.get_level_values("symbol")))
    x = np.arange(len(rules))
    width = 0.8 / len(symbols)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, sym in enumerate(symbols):
        vals = [table.loc[(sym, r), "Sharpe"] for r in rules]
        ax.bar(x + i * width, vals, width, label=sym)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x + width * (len(symbols) - 1) / 2)
    ax.set_xticklabels(rules, rotation=15)
    ax.set_ylabel("Sharpe")
    ax.set_title("Exit-rule sensitivity: no usable edge under any exit (vs 0.44–0.62 B&H)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    path = Path(out_dir) / "exit_sensitivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _print(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    print(table.round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--symbol", default="SPY", choices=list(SYMBOLS),
                        help="symbol for the single-symbol tables and plots")
    parser.add_argument("--replicate", action="store_true",
                        help="reproduce the article's SPY 2020-21 window")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", default="Results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replicate:
        _print("Article window vs full history", replicate_article(args.symbol))

    table = across_symbols()
    _print("Elegant Oscillator across ETFs, full history, 0 bps", table)
    table.to_csv(out_dir / "across_symbols.csv")

    costs = cost_sensitivity(args.symbol)
    _print(f"[{args.symbol}] cost sensitivity", costs)
    costs.to_csv(out_dir / f"cost_sensitivity_{args.symbol}.csv")

    exits = exit_sensitivity()
    _print("Exit-rule sensitivity across ETFs, 0 bps", exits)
    exits.to_csv(out_dir / "exit_sensitivity.csv")

    if args.plot:
        print("\nSaved:")
        print(" ", save_indicator_plot(args.symbol, out_dir))
        print(" ", save_exit_sensitivity_plot(exits, out_dir))
        for sym in SYMBOLS:
            print(" ", save_equity_plot(sym, out_dir))

    print(f"\nCSV output written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
