"""Homework 4 — Algory QI Education, Fall 2026
"""

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

LOOKBACK = 60

def load_and_split(split_date="2011-09-09", prices=None, warmup=LOOKBACK):
    """Split ^SP500TR since 1990 into training and test at `split_date`.

    Training ends on the split date's close. Test includes `warmup` closes
    ending on that date, so the 60-day window is full on the first day traded out
    of sample. Those overlapping days are history the strategy reads, never days
    it earns a return on. Cut the series at one date with no overlap and the
    first 60 days of the test set have no window, so the out-of-sample period
    silently starts three months late.

    September 9, 2011 is a Friday. A split date has to be a trading day, or
    "ends on that close" names a close that does not exist.

    Pass `prices` to split a series you already have, which is how the checker
    tests this without a download. Return (training, test).
    """
    if prices is None:
        data = yf.download("^SP500TR", start="1990-01-01")
        prices = data["Close"]

    split_pos = prices.index.get_loc(split_date)
    # Include the split close in the initial window; earn only later returns.
    warmup_date = prices.index[split_pos - warmup + 1]

    training = prices.loc[:split_date]
    test = prices.loc[warmup_date:]


    print("Training days", len(training))
    print("Test days", len(test))

    print("Mean training price", np.mean(training))
    print("Mean test price", np.mean(test))

    return training, test


def always_hold(window):
    """Always invest. The control case, and the one that proves the backtester."""
    return True


def always_cash(window):
    """Never invest. Must produce exactly 0% return."""
    return False


def make_random_hold(p=0.7, seed=0):
    """Return a strategy that invests with probability p each day.

    A factory, not a plain function, because the generator has to live somewhere
    and a default argument is the wrong place: `rng=np.random.default_rng(0)` in
    a signature is evaluated once when the module loads, so every backtest in the
    session shares one advancing generator and the same backtest returns a
    different number each time you run it. Building the generator here gives one
    per strategy, so `backtest(train, make_random_hold())` is reproducible.
    """
    rng = np.random.default_rng(seed)

    def random_hold(window):
        return bool(rng.random() < p)

    return random_hold


def five_day_reversal(window):
    """The rule from class: hold unless the last 5 days were positive.

    Measure that return from the close 5 days before today to today's close,
    both of which are inside the window.
    """
    five_day_return = window[-1] / window[-6] - 1
    return bool(five_day_return <= 0)


# training strategies
def below_average(window):
    """Hold below the 60-close average; otherwise use cash."""
    return bool(window[-1] < np.mean(window))

def quiet_recent_moves(window):
    """Hold when the last 5 absolute daily returns average below the full window."""
    returns = window[1:] / window[:-1] - 1
    recent_average = np.mean(np.abs(returns[-5:]))
    window_average = np.mean(np.abs(returns))
    return bool(recent_average < window_average)

def no_abnormal_moves(window):
    """Hold unless a recent return is over 2 baseline standard deviations away."""
    returns = window[1:] / window[:-1] - 1
    baseline = returns[:-5]
    recent = returns[-5:]
    average = np.mean(baseline)
    standard_deviation = np.std(baseline)
    abnormal = np.abs(recent - average) > 2 * standard_deviation
    return bool(not np.any(abnormal))

def above_average(window):
    """Idea 4: hold when the latest close is above the 60-close average."""
    return bool(window[-1] > np.mean(window))

def short_average_above_long(window):
    """Idea 5: hold when the 15-close average exceeds the 60-close average."""
    return bool(np.mean(window[-15:]) > np.mean(window))

def dip_in_uptrend(window):
    """Idea 6: hold above the 60-close average after a negative 5-day return."""
    five_day_return = window[-1] / window[-6] - 1
    return bool(above_average(window) and five_day_return < 0)

def trend_with_risk_filter(window):
    """Idea 7: hold above the 60-close average with no abnormal recent moves."""
    return bool(above_average(window) and no_abnormal_moves(window))

def my_strategy(window):
    """Final rule: hold when the 15-close average exceeds the 60-close average."""
    return short_average_above_long(window)


def backtest(prices, strategy, lookback=LOOKBACK):
    """Walk forward one day at a time and apply the strategy's decision.

    The convention, stated exactly, because everything else depends on it:

        on day i, the strategy is handed prices[i - lookback + 1 : i + 1],
        the `lookback` closes ENDING WITH today's, today's included;
        if it says hold you earn prices[i + 1] / prices[i] - 1, otherwise zero.

    So i runs from lookback - 1 to len(prices) - 2. The first decision is made on
    the 60th close and the first return earned runs from the 60th close to the
    61st. Today's close is known when you decide, so withholding it throws away a
    day of information for nothing; tomorrow's close is what you are predicting,
    so reading it is the bug this whole session is about.

    Return a dict with at least:
        total_return, annualised_return, max_drawdown, fraction_invested

    Annualise by trading days, 252 a year, compounded: value ** (252 / days) - 1.

    Sanity checks before you trust it:
        always_hold  must match buy-and-hold over the days actually traded,
                     prices[-1] / prices[lookback - 1] - 1, not over the
                     whole file
        always_cash  must return exactly 0.0
    If either fails you have an off-by-one, and every number after it is fiction.
    """
    prices = np.asarray(prices, dtype=float).reshape(-1)
    decisions = []
    portfolio = [1.0]
    index = [1.0]


    for i in range(lookback-1, len(prices)-1):
        window = prices[i-lookback+1:i+1]
        hold = bool(strategy(window))

        tomorrow = prices[i+1]/prices[i] - 1

        if hold:
            portfolio.append(portfolio[-1]*(tomorrow+1))
        else:
            portfolio.append(portfolio[-1])

        index.append(index[-1] * (1 + tomorrow))
        decisions.append(hold)

    portfolio = np.array(portfolio)
    index = np.array(index)
    days = len(decisions)

    total_return = portfolio[-1]-1
    annualized_return = portfolio[-1]**(252/days)-1

    index_total_return = index[-1]-1
    index_annualized_return = index[-1]**(252/days)-1

    running_peak = np.maximum.accumulate(portfolio)
    max_drawdown = np.max(1-portfolio/running_peak)

    index_running_peak = np.maximum.accumulate(index)
    index_max_drawdown = np.max(1 - index / index_running_peak)
    fraction_invested = np.mean(decisions)

    print()
    print(f"{'Investment':<16} {'Total return':>15} {'Annualised':>15}")
    print("-"*48)
    print(f"{'Strategy':<16} {total_return:>+15.2%} {annualized_return:>+15.2%}")
    print(f"{'Buy and hold':<16} {index_total_return:>+15.2%} {index_annualized_return:>+15.2%}")
    print()

    plt.figure()
    plt.plot(portfolio, label="Strategy")
    plt.plot(index, label="Buy and Hold", linestyle="--")
    plt.xlabel("Trading Days")
    plt.ylabel("Value of $1")
    plt.title("Strategy vs. Index")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "total_return": total_return,
        "annualised_return": annualized_return,
        "index_total_return": index_total_return,
        "index_annualized_return": index_annualized_return,
        "max_drawdown": max_drawdown,
        "index_max_drawdown": index_max_drawdown,
        "fraction_invested": fraction_invested,
    }


def main():
    print("\n" + "="*64)
    print("DATA SUMMARY")
    print("="*64)
    train, test = load_and_split()
    print("\n" + "="*64)
    print("TRAINING: CONTROL STRATEGIES AND FIVE-DAY REVERSAL")
    print("="*64)
    for name, strat in [("always hold", always_hold),
                        ("always cash", always_cash),
                        ("random 70%", make_random_hold(0.7, seed=0)),
                        ("5-day reversal", five_day_reversal)]:
        print(f"\n{name.upper()}")
        print("-"*64)
        r = backtest(train, strat)

    # evaluate every candidate on training data only.
    print("\n" + "="*64)
    print("Q6: CANDIDATE STRATEGIES — TRAINING DATA")
    print("="*64)
    variants = [
        ("Idea 1: hold below the 60-close average; otherwise cash", below_average),
        ("Idea 2: hold if last 5 mean absolute returns < full-window average; otherwise cash",
         quiet_recent_moves),
        ("Idea 3: hold if no last-5 return exceeds 2 earlier-history standard deviations from its mean; otherwise cash",
         no_abnormal_moves),
        ("Idea 4: hold above the 60-close average; otherwise cash", above_average),
        ("Idea 5: hold if the 15-close average > 60-close average; otherwise cash",
         short_average_above_long),
        ("Idea 6: hold above the 60-close average after a negative 5-day return; otherwise cash",
         dip_in_uptrend),
        ("Idea 7: hold above the 60-close average with no abnormal last-5 returns; otherwise cash",
         trend_with_risk_filter),
    ]
    for rule, strat in variants:
        print(f"\n{rule}")
        print("-"*64)
        r = backtest(train, strat)
        print(f"{rule} | training annualised return {r['annualised_return']:+.2%}\n")

    # fixed before inspecting test results.
    print("\n" + "="*64)
    print("Q7: FINAL STRATEGY — IDEA 5")
    print("=" * 64)
    print(
        "Hold the index when the average of the latest 15 closes exceeds "
        "the average of all 60 closes; otherwise hold cash. "
        "The rule assumes that recent strength relative to the longer trend "
        "may persist, while weakening trends may justify moving to cash. "
        "This is a trend-following hypothesis, not a guarantee of future performance."
    )

    # one evaluation of the fixed rule
    print("\n" + "="*64)
    print("Q8: TEST SET — FINAL IDEA 5 STRATEGY")
    print("="*64)
    result = backtest(test, my_strategy)
    print(f"{'Strategy maximum drawdown:':<30} {result['max_drawdown']:>8.2%}")
    print(f"{'Index maximum drawdown:':<30} {result['index_max_drawdown']:>8.2%}")
    print(f"{'Fraction of days invested:':<30} {result['fraction_invested']:>8.2%}")
    print()

    # what does your strategy say to do today?
    print("\n" + "="*64)
    print("Q9: WHAT TODAY?")
    print("="*64)

    latest_window = np.asarray(test.iloc[-LOOKBACK:],dtype=float).reshape(-1)
    latest_close = test.index[-1]
    signal = my_strategy(latest_window)
    print("Strategy Today:", "HOLD" if signal else "CASH")
    print("Last close used:", latest_close)

if __name__ == "__main__":
    main()
