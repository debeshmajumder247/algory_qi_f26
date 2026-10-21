# S&P 500 Strategy Backtester

A Python research project for comparing rule-based investment strategies against buy-and-hold using historical S&P 500 Total Return Index data. The focus is a transparent daily backtesting engine: explicit signal timing, consistent benchmark comparisons, and a chronological training/test split.

Developed as part of Algory QI Education coursework (Fall 2026).

## Overview

The project explores when a portfolio should hold the index and when it should remain in cash. Each strategy receives a rolling window of closing prices and returns a Boolean investment decision. The same engine evaluates every rule, compounds portfolio returns, and compares performance with an always-invested benchmark over the identical trading period.

The final configured strategy holds the index when the **15-close moving average exceeds the 60-close moving average**, and otherwise holds cash. This is a trend-following hypothesis; the repository does not establish that it will outperform in future markets.

## Features

- Daily walk-forward simulation with a configurable lookback, defaulting to 60 closes.
- Chronological training/test split with historical warmup data for the first test-period decision.
- Buy-and-hold and all-cash controls, plus a seeded random-investment baseline.
- Seven candidate rules covering trend, reversal, and recent-move filters.
- Compounded total return, annualised return, maximum drawdown, and fraction of days invested.
- Portfolio and benchmark equity curves, each starting at $1.
- A final HOLD/CASH signal based on the latest available downloaded close.

## How the Backtest Works

For a decision at close `i`, the engine supplies the strategy with:

```python
prices[i - lookback + 1 : i + 1]
```

If the strategy returns `True`, the portfolio earns the next close-to-close return:

```python
prices[i + 1] / prices[i] - 1
```

Otherwise, its value remains unchanged for that interval. The first decision uses the 60th close, and the first simulated return runs from the 60th close to the 61st. Future prices are excluded from the signal window.

The benchmark earns every return over those same intervals. This alignment matters: comparing the strategy with a benchmark that starts at the beginning of the raw dataset would give them different investment horizons.

### Training and test periods

The script downloads `^SP500TR` through `yfinance`, requesting data from January 1, 1990 onward. The default split date is **September 9, 2011**:

- Training includes the split-date close.
- The test input includes 60 historical closes ending on the split date, followed by subsequent observations.
- Those 60 overlapping closes provide signal history; the test simulation earns returns only after the split-date close.

The main workflow evaluates candidate strategies on training data, then evaluates the configured final rule on the test data. Keeping test results separate from strategy selection is essential to interpreting that evaluation.

## Strategies

| Strategy | Invest when… |
| --- | --- |
| Always hold | Every day; a control for benchmark alignment. |
| Always cash | Never; a control for zero portfolio returns. |
| Random hold | A seeded random draw falls below a chosen probability, defaulting to 70%. |
| Five-day reversal | The latest five-day return is zero or negative. |
| Below average | The latest close is below the 60-close average. |
| Quiet recent moves | The last five absolute daily returns average less than all daily returns in the window. |
| No abnormal moves | None of the last five daily returns lies more than two baseline standard deviations from the baseline mean; the baseline excludes those five returns. |
| Above average | The latest close is above the 60-close average. |
| Short average above long | The 15-close average exceeds the 60-close average. **Final configured rule.** |
| Dip in uptrend | The latest close is above the 60-close average and the five-day return is negative. |
| Trend with risk filter | The latest close is above the 60-close average and the abnormal-move filter passes. |

## Getting Started

Place this README alongside `hw4.py` in the repository root. Use Python 3 in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas yfinance matplotlib
python hw4.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

An internet connection is required for the default market-data download. The script prints data summaries and performance comparisons and opens a chart for each backtest. Depending on the plotting backend, close each chart window to continue to the next evaluation.

### Evaluate an individual strategy

```python
from hw4 import load_and_split, backtest, my_strategy

train, test = load_and_split()
results = backtest(test, my_strategy)

print(f"Annualised return: {results['annualised_return']:.2%}")
print(f"Maximum drawdown: {results['max_drawdown']:.2%}")
print(f"Days invested: {results['fraction_invested']:.2%}")
```

`load_and_split(prices=...)` also accepts an existing date-indexed price series, allowing the split logic to be used without a download. The split date must exist in that series, with enough preceding observations for the requested warmup.

## Performance Metrics

| Metric | Definition |
| --- | --- |
| Total return | Final portfolio value divided by initial value, minus one. |
| Annualised return | Final value raised to `252 / simulated_days`, minus one, with initial value normalised to 1. |
| Maximum drawdown | Largest proportional decline from a previous portfolio peak, reported as a positive loss magnitude. |
| Fraction invested | Share of simulated return intervals for which the strategy holds the index. |

The returned dictionary also includes benchmark total return, annualised return, and maximum drawdown. Performance figures are generated at runtime; no fixed performance claims are included here because the download endpoint and available history can change.

## Validation Controls

Two useful invariants for checking the engine are:

```python
always_hold_total_return == prices[-1] / prices[lookback - 1] - 1
always_cash_total_return == 0.0
```

The first comparison should use floating-point tolerance. These controls are included in the training workflow for inspection; the current script does not contain an automated assertion suite.

The random baseline uses a generator created by `make_random_hold(p, seed)`. Create a fresh strategy with the same seed to reproduce its decision sequence; reusing an existing strategy instance advances its generator state.

## Assumptions and Limitations

- **Simplified execution:** Signals use the current closing price and assume exposure begins at that same close. This avoids future-price input but does not model the practical delay between observing a final close and executing a trade.
- **Frictionless trading:** There are no transaction costs, slippage, taxes, or liquidity constraints.
- **Binary allocation:** The portfolio is fully invested or fully in cash, without leverage or short selling. Cash earns zero interest.
- **Index proxy:** The total-return index is a research series, not a directly tradable security. Fund fees and tracking differences are excluded.
- **Data dependence:** Availability, revisions, missing values, and downloaded data structure can affect results. The script does not provide comprehensive input validation and expects a single valid price series with more observations than the lookback.
- **Research scope:** The workflow uses one chronological split. Comparing multiple candidates can overfit the training sample, and repeatedly inspecting test results can weaken the value of the holdout.
- **Reproducibility:** Dependencies are not pinned and downloaded data is not snapshotted. Exact numerical reproduction requires preserving both.

## Engineering Decisions Worth Discussing

- **Signal/return alignment:** The window ends at the decision close, while portfolio performance starts with the next interval. This makes the timing convention inspectable and helps expose off-by-one errors.
- **Warmup without overlapping returns:** Historical test context preserves the intended evaluation start date while keeping earned test returns after the split.
- **Shared strategy interface:** A `strategy(window) -> bool` contract lets different hypotheses use the same portfolio accounting and benchmark logic.
- **Independent random state:** A strategy factory makes the random baseline reproducible without relying on a shared generator created at import time.
- **Multiple evaluation dimensions:** Return, drawdown, and market exposure describe different aspects of a strategy rather than reducing the comparison to a single return figure.

## Potential Extensions

- Add automated invariant tests and validation for short, missing, or invalid price data.
- Separate calculation, reporting, and plotting so experiments can run without interactive windows.
- Model execution delays, transaction costs, and interest on cash.
- Add rolling walk-forward evaluation and parameter-sensitivity analysis.
- Save dated datasets, pin dependencies, and export experiment results for reproducibility.
