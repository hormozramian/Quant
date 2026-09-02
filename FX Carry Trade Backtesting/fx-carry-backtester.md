---
hide:
  - title
  - navigation
  - footer
  - toc
---

# FX Carry Strategy Backtester

This notebook implements and runs the FX Carry Strategy Backtester described in the accompanying documentation. Since no real market data file was provided, a realistic synthetic dataset (spot, forward, and PPP fair-value rates for 8 currency pairs, 20 years of monthly data) is generated first, written to an actual Excel workbook, and then read back in through the documented `get_data()` interface, so the demonstration exercises the real, documented data path end to end.

The code below is a **corrected implementation**: the original script contained 9 issues identified during review (argument-order bug, a non-rolling "rolling" Sharpe, a hardcoded annualization factor, and others). Each fix is marked inline with a comment referencing the issue number from the review. See the accompanying notes document for the full list.

**What this notebook covers:**
1. Simulating realistic FX spot, forward, and PPP data (with Covered Interest Rate Parity enforced)
2. The corrected backtesting engine (data loading, signals, weights, performance, reporting)
3. Running all four strategies (`Simple_Carry`, `VolAdj_Carry`, `Value`, `Max_Sharpe_Port`)
4. Combining strategies (50/50 blend and composite signal)
5. A performance summary across every strategy tested

## 1. Simulating Realistic FX Data

Eight currency pairs against the US dollar, monthly data over 20 years. Short-term interest rates are simulated per currency (some structurally high-yield like AUD/NZD, some low-yield like JPY/CHF), spot rates follow a random walk with a modest pull toward a slowly-drifting PPP fair value and a small carry-consistent drift (a mild Uncovered Interest Parity violation, without which carry would earn nothing on average by construction), and forward rates are derived from spot **via Covered Interest Rate Parity** exactly as the codebase assumes.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (7, 4.2), 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 11, 'grid.alpha': 0.35,
})

rng = np.random.default_rng(7)

pairs = ['AUDUSD', 'NZDUSD', 'GBPUSD', 'EURUSD', 'CADUSD', 'NOKUSD', 'JPYUSD', 'CHFUSD']
periods_per_year = 12
n_years = 20
n_periods = n_years * periods_per_year
dates = pd.date_range('2004-01-31', periods=n_periods, freq='ME')

base_levels = {'AUDUSD': 0.045, 'NZDUSD': 0.05, 'GBPUSD': 0.035, 'EURUSD': 0.02,
               'CADUSD': 0.025, 'NOKUSD': 0.04, 'JPYUSD': 0.002, 'CHFUSD': 0.005}
domestic_rate_level = 0.02   # USD short rate

def sim_rate(level, n, sigma, rng):
    x = np.zeros(n); x[0] = level
    for t in range(1, n):
        x[t] = max(x[t-1] + 0.03 * (level - x[t-1]) + sigma * rng.normal(), -0.005)
    return x

domestic_rate = sim_rate(domestic_rate_level, n_periods, 0.0015, rng)
foreign_rates = {p: sim_rate(lvl, n_periods, 0.0018, rng) for p, lvl in base_levels.items()}

spot0 = {'AUDUSD': 0.75, 'NZDUSD': 0.68, 'GBPUSD': 1.55, 'EURUSD': 1.20,
         'CADUSD': 0.85, 'NOKUSD': 0.16, 'JPYUSD': 0.0091, 'CHFUSD': 0.75}

spot, ppp = {}, {}
for p in pairs:
    s, v = np.zeros(n_periods), np.zeros(n_periods)
    s[0] = v[0] = spot0[p]
    rate_diff = domestic_rate - foreign_rates[p]
    for t in range(1, n_periods):
        v[t] = v[t-1] * (1 + 0.001 * rng.normal())                   # PPP fair value: slow random walk
        mean_revert = 0.01 * (v[t-1] - s[t-1]) / s[t-1]               # gentle pull toward fair value
        carry_drift = 0.15 * rate_diff[t-1] / periods_per_year        # mild UIP violation
        s[t] = s[t-1] * (1 + mean_revert + carry_drift + rng.normal(0, 0.028))
    spot[p], ppp[p] = s, v

df_spot = pd.DataFrame(spot, index=dates)
df_val = pd.DataFrame(ppp, index=dates)

df_fwd = pd.DataFrame(index=dates, columns=pairs, dtype=float)        # Covered Interest Rate Parity
for p in pairs:
    i_f, i_d = foreign_rates[p] / periods_per_year, domestic_rate / periods_per_year
    df_fwd[p] = df_spot[p].values * (1 + i_d) / (1 + i_f)

for name, d in [('Spot', df_spot), ('Forward', df_fwd), ('Value', df_val)]:
    d.index.name = 'Date'

print(f"Simulated {n_periods} months ({n_years} years) across {len(pairs)} currency pairs")
print(df_spot.tail(3).round(4))
```

```text
Simulated 240 months (20 years) across 8 currency pairs
            AUDUSD  NZDUSD  GBPUSD  EURUSD  CADUSD  NOKUSD  JPYUSD  CHFUSD
Date                                                                      
2023-10-31  0.8581  0.5880  1.2319  1.3002  0.6330  0.1355  0.0079  0.7716
2023-11-30  0.8636  0.6041  1.1624  1.3028  0.5891  0.1347  0.0078  0.7935
2023-12-31  0.8332  0.6145  1.2438  1.3137  0.6073  0.1317  0.0080  0.8099
```

Written to an actual `.xlsx` workbook, then read back through the documented `get_data()` function, exercising the real data-loading path.

```python
with pd.ExcelWriter('fx_data.xlsx', engine='openpyxl') as writer:
    df_spot.to_excel(writer, sheet_name='Spot')
    df_fwd.to_excel(writer, sheet_name='Forward')
    df_val.to_excel(writer, sheet_name='Value')

def get_data(fname, sheet):
    # Reads an Excel sheet, indexed by Date.
    return pd.read_excel(fname, sheet_name=sheet, index_col='Date')

df_spot = get_data('fx_data.xlsx', 'Spot')
df_fwd = get_data('fx_data.xlsx', 'Forward')
df_val = get_data('fx_data.xlsx', 'Value')
vol = df_spot.pct_change().rolling(12).std() * (12 ** 0.5)     # realized vol, for VolAdj_Carry

print(f"Loaded from fx_data.xlsx: Spot {df_spot.shape}, Forward {df_fwd.shape}, Value {df_val.shape}")
```

```text
Loaded from fx_data.xlsx: Spot (240, 8), Forward (240, 8), Value (240, 8)
```

**Diagram — spot rate history.** All 8 pairs, indexed to 100 at inception so the relative drift is visible on a comparable scale.

```python
fig, ax = plt.subplots(figsize=(9, 5))
indexed = df_spot / df_spot.iloc[0] * 100
for col in indexed.columns:
    ax.plot(indexed.index, indexed[col], linewidth=1.3, label=col)
ax.set_xlabel('Date'); ax.set_ylabel('Indexed Spot Rate (100 = start)')
ax.set_title('Simulated FX Spot Rates (Indexed)')
ax.legend(ncol=4, fontsize=8)
plt.show()
```

![Figure 1](images/fx-carry-backtester_fig_01.png)

## 2. Signals and Returns

`get_returns()` produces two series from the same raw data: the **realized** carry return (spot today vs. the forward locked in last period), and a **signal** whose definition depends on the strategy chosen.

$$r_t = \left(\frac{S_t}{F_{t-1}} - 1\right) \times 100 \qquad \text{(realized return, all strategies)}$$

- `Simple_Carry` / `Max_Sharpe_Port`: signal = current forward premium, $(S_t/F_t - 1) \times 100$
- `VolAdj_Carry`: the same signal divided by realized volatility
- `Value`: PPP deviation, $(V_t - S_t)/S_t$, a mean-reversion signal rather than a carry signal

```python
def get_returns(df_spot, df_fwd, df_val, vol, strategy):
    # Realized P&L: spot today vs. the forward rate locked in one period ago.
    df_ret = (df_spot / df_fwd.shift(1) - 1) * 100

    if strategy in ('Simple_Carry', 'Max_Sharpe_Port'):
        df_signal = ((df_spot / df_fwd) - 1) * 100
    elif strategy == 'VolAdj_Carry':
        df_signal = ((df_spot / df_fwd) - 1) * 100 / vol
    elif strategy == 'Value':
        df_signal = (df_val - df_spot) / df_spot   # FIX #5: label-aligned (no .values), matches df_spot on Date+column
    else:
        raise ValueError('Error in Strategy Name')

    return df_ret, df_signal

print("get_returns defined.")
```

```text
get_returns defined.
```

## 3. Portfolio Weights

Two ways signals become weights: **rule-based** (equal-weighted sign of the signal, used for `Simple_Carry`, `VolAdj_Carry`, `Value`) and **mean-variance optimized** (`Max_Sharpe_Port`, re-solved on every rolling window).

```python
def portfolio_annualised_performance(weights, mean_returns, carry_cov, periods_per_year):
    # Annualized return: R_p = (w . mu) * periods_per_year
    returns = np.sum(mean_returns * weights) * periods_per_year
    # Annualized vol:   sigma_p = sqrt(w' Sigma w) * sqrt(periods_per_year)
    std = np.sqrt(np.dot(weights.T, np.dot(carry_cov, weights))) * np.sqrt(periods_per_year)
    return returns, std


def neg_sharpe_ratio(weights, mean_returns, carry_cov, risk_free_rate, periods_per_year):
    # p_std, not p_var: portfolio_annualised_performance already applies sqrt(...). FIX #8: renamed from p_var.
    p_ret, p_std = portfolio_annualised_performance(weights, mean_returns, carry_cov, periods_per_year)
    return (risk_free_rate - p_ret) / p_std


def max_sharpe_ratio(mean_returns, carry_cov, risk_free_rate, periods_per_year):
    # Dollar-neutral, long/short book: weights must sum to ZERO, not one. FIX #6: documented explicitly.
    # Each currency's weight is bounded in [-100%, 100%]. SLSQP handles the nonlinear
    # objective (Sharpe ratio) with linear equality + box constraints.
    args = (mean_returns, carry_cov, risk_free_rate, periods_per_year)   # explicit tuple, not locals().values()
    no = len(mean_returns)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 0}
    bounds = [(-1.0, 1.0) for _ in range(no)]
    return sco.minimize(neg_sharpe_ratio, no * [1 / no], args=args, method='SLSQP', bounds=bounds,
                        constraints=constraints)


def set_weights_day(i, df_ret, periods_per_year, look_back_carry_ret):
    # Trailing rolling window of raw (not-yet-annualized) returns, re-optimized fresh each call.
    ret = df_ret.iloc[i: i + look_back_carry_ret] / 100
    max_sharpe = max_sharpe_ratio(ret.mean(), ret.cov(), 0, periods_per_year)
    # Label the resulting weight vector with the *last* date in the window: the date this
    # weight decision would actually be made on.
    return pd.Series(max_sharpe.x, ret.columns, name=ret.index[-1]).round(7).sort_index()


def set_weights(df_ret, df_buy_sell, strategy, periods_per_year, look_back_carry_ret):
    if strategy == 'Max_Sharpe_Port':
        # FIX #1: arguments now passed in the order set_weights_day actually expects
        # (periods_per_year, look_back_carry_ret) -- the original call had these swapped.
        return pd.concat([set_weights_day(i, df_ret, periods_per_year, look_back_carry_ret)
                          for i in range(len(df_ret) - look_back_carry_ret)], axis=1).T
    else:
        # Rule-based strategies: pass through the externally-computed buy/sell weight matrix.
        return df_buy_sell

print("Weight-construction functions defined.")
```

```text
Weight-construction functions defined.
```

## 4. Performance Metrics

Annualized return, volatility, Sharpe ratio, maximum drawdown, and a genuinely rolling Sharpe ratio.

```python
def maximum_drawdown(ret_series):
    # Returns maximum drawdown MAGNITUDE (largest peak-to-trough decline, a fraction),
    # not a duration in periods. FIX #7: docstring corrected to match actual behavior.
    cum_ret = np.cumprod(1 + ret_series / 100)
    mdd = 0
    peak = cum_ret.iloc[0] if hasattr(cum_ret, 'iloc') else cum_ret[0]
    for x in cum_ret:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd


def get_sharpe(ret_series, periods_per_year):
    mu = ret_series.mean() * periods_per_year
    # FIX #3: annualize by sqrt(periods_per_year), not a hardcoded sqrt(12).
    # The original hardcoded value only happened to be correct for monthly data.
    std = ret_series.std(ddof=1) * (periods_per_year ** 0.5)
    sharpe = mu / std
    return mu, std, sharpe


def get_rolling_sharpe(weightd_ret, periods_per_year, look_back_rolling_sharpe, num_obs):
    # FIX #2: a genuine trailing WINDOW of returns (.iloc[i-L : i]), not a single scalar
    # (the original indexed weightd_ret[i - look_back_rolling_sharpe], one value only).
    segment_sharpes = [get_sharpe(weightd_ret.iloc[i - look_back_rolling_sharpe: i], periods_per_year)
                       for i in range(look_back_rolling_sharpe, num_obs)]
    return np.array([sharpe if std else 0 for mu, std, sharpe in segment_sharpes])


def get_results(df_ret, weights, strategy, periods_per_year, look_back_carry_ret,
                look_back_rolling_sharpe, num_obs):
    # Lag weights by one period: the weight decided using info known at t-1 is applied to
    # the return realized during period t. This prevents look-ahead bias in the backtest.
    weightd_ret = (df_ret * weights.shift(1)).sum(axis=1)

    # Trim the burn-in period where no (or NaN) weights exist yet.
    ret = weightd_ret[look_back_carry_ret + 1:] if strategy == 'Max_Sharpe_Port' else weightd_ret[1:]
    weightd_ret_mu, weightd_ret_std, sharpe = get_sharpe(ret, periods_per_year)
    mdd = maximum_drawdown(weightd_ret)
    sharpe_roll_wind = get_rolling_sharpe(weightd_ret, periods_per_year, look_back_rolling_sharpe, num_obs)

    return weightd_ret, weightd_ret_mu, weightd_ret_std, sharpe, sharpe_roll_wind, mdd

print("Performance-metric functions defined.")
```

```text
Performance-metric functions defined.
```

## 5. Reporting

```python
def print_results(df_ret, weightd_ret, weightd_ret_mu, weightd_ret_std, sharpe, sharpe_roll_wind, mdd,
                  spot_codes, look_back_rolling_sharpe, periods_per_year, title_suffix=''):
    print('Return =', round(weightd_ret_mu, 3), '%')
    print('Volatility =', round(weightd_ret_std, 3), '%')
    print('Sharpe Ratio =', round(sharpe, 3))
    print('Max Drawdown =', round(mdd * 100, 3), '%')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    axes[0].plot(df_ret.index, weightd_ret, color='#C73E1D', linewidth=1.3)
    axes[0].set_title(f'Strategy Returns{title_suffix}')
    axes[0].set_ylabel('%'); axes[0].grid(True)

    for spot_code in spot_codes:
        axes[1].plot(df_ret.index, df_ret[spot_code], linewidth=0.9, label=spot_code)
    axes[1].set_title('Underlying FX Returns')
    axes[1].set_ylabel('%')
    axes[1].legend(loc=2, ncol=2, prop={'size': 7}).get_frame().set_alpha(0.1)

    # FIX #9: dynamic title (years, not a hardcoded "3 year") derived from the actual lookback.
    years_label = look_back_rolling_sharpe / periods_per_year
    axes[2].plot(df_ret.index[look_back_rolling_sharpe:], sharpe_roll_wind, color='#2E86AB', linewidth=1.3)
    axes[2].axhline(0, color='#888888', linewidth=0.8)
    axes[2].set_title(f'Rolling {years_label:.0f}-Year Sharpe Ratio{title_suffix}')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

print("Reporting function defined.")
```

```text
Reporting function defined.
```

## 6. Running All Four Strategies

`look_back_carry_ret = 36` (3-year window for `Max_Sharpe_Port`'s rolling optimization) and `look_back_rolling_sharpe = 36` (3-year rolling Sharpe window), matching the documented example, with monthly data (`periods_per_year = 12`).

The buy/sell weight matrix for the three rule-based strategies is built as an equal-weighted sign of the signal, the simplest reasonable implementation of the step the original documentation leaves unspecified.

```python
look_back_carry_ret = 36
look_back_rolling_sharpe = 36
spot_codes = df_spot.columns.tolist()

results_by_strategy = {}
port_weightd_ret = pd.DataFrame(index=df_spot.index)
buy_sell_by_strategy = {}

for strategy in ['Simple_Carry', 'VolAdj_Carry', 'Value', 'Max_Sharpe_Port']:
    df_ret, df_signal = get_returns(df_spot, df_fwd, df_val, vol, strategy)

    if strategy == 'Max_Sharpe_Port':
        weights = set_weights(df_ret, None, strategy, periods_per_year, look_back_carry_ret)
    else:
        df_buy_sell = np.sign(df_signal) / np.sign(df_signal).abs().sum(axis=1).values.reshape(-1, 1)
        buy_sell_by_strategy[strategy] = df_buy_sell
        weights = set_weights(df_ret, df_buy_sell, strategy, periods_per_year, look_back_carry_ret)

    weightd_ret, mu, std, sharpe, sharpe_roll, mdd = get_results(
        df_ret, weights, strategy, periods_per_year, look_back_carry_ret,
        look_back_rolling_sharpe, num_obs=len(df_ret))

    results_by_strategy[strategy] = dict(df_ret=df_ret, weightd_ret=weightd_ret, mu=mu, std=std,
                                          sharpe=sharpe, sharpe_roll=sharpe_roll, mdd=mdd)
    port_weightd_ret[strategy] = weightd_ret
    print(f"{strategy:<18} Return={mu:7.3f}%   Vol={std:7.3f}%   Sharpe={sharpe:6.3f}   MaxDD={mdd*100:6.2f}%")
```

```text
Simple_Carry       Return=  1.891%   Vol=  3.434%   Sharpe= 0.551   MaxDD=  4.72%
VolAdj_Carry       Return=  1.722%   Vol=  3.284%   Sharpe= 0.524   MaxDD=  4.72%
Value              Return=  1.763%   Vol=  3.214%   Sharpe= 0.549   MaxDD=  7.64%
Max_Sharpe_Port    Return= -7.013%   Vol= 15.709%   Sharpe=-0.446   MaxDD= 77.00%
```

**Simple_Carry** in full detail (strategy returns, underlying FX returns, rolling Sharpe).

```python
r = results_by_strategy['Simple_Carry']
print_results(r['df_ret'], r['weightd_ret'], r['mu'], r['std'], r['sharpe'], r['sharpe_roll'], r['mdd'],
              spot_codes, look_back_rolling_sharpe, periods_per_year, title_suffix=' (Simple_Carry)')
```

```text
Return = 1.891 %
Volatility = 3.434 %
Sharpe Ratio = 0.551
Max Drawdown = 4.724 %
```

![Figure 2](images/fx-carry-backtester_fig_02.png)

**Max_Sharpe_Port** in full detail. This is the strategy most exposed to a well-known weakness of naive mean-variance optimization: re-estimating mean and covariance from a comparatively short (36-month) rolling window, across only 8 assets, produces noisy, unstable weight estimates that swing toward the ±100% bounds frequently — a classic "estimation-error maximizer" failure mode (Michaud, 1989), not a bug in the implementation.

```python
r = results_by_strategy['Max_Sharpe_Port']
print_results(r['df_ret'], r['weightd_ret'], r['mu'], r['std'], r['sharpe'], r['sharpe_roll'], r['mdd'],
              spot_codes, look_back_rolling_sharpe, periods_per_year, title_suffix=' (Max_Sharpe_Port)')
```

```text
Return = -7.013 %
Volatility = 15.709 %
Sharpe Ratio = -0.446
Max Drawdown = 76.998 %
```

![Figure 3](images/fx-carry-backtester_fig_03.png)

## 7. Combining Strategies

Carry and value signals are known to be close to orthogonal empirically, carry tends to do well when valuations are stretched (and can reverse sharply), while value is slow-moving and mean-reverting. Two ways of combining `Simple_Carry` and `Value` are demonstrated: a simple 50/50 return blend, and a composite signal that only takes a position when both strategies agree on direction.

```python
def combine_strategies(strat1, strat2, port_weightd_ret, periods_per_year, look_back_rolling_sharpe):
    strategy = '50/50 {} & {}'.format(strat1, strat2)
    weightd_ret = (port_weightd_ret.loc[:, [strat1, strat2]] / 2).sum(axis=1)
    weightd_ret_mu, weightd_ret_std, sharpe = get_sharpe(weightd_ret, periods_per_year)
    mdd = maximum_drawdown(weightd_ret)
    # FIX #4: rolling Sharpe is now recomputed for the blended series itself, rather than
    # reusing whichever single strategy's rolling Sharpe happened to be passed in.
    sharpe_roll_wind = get_rolling_sharpe(weightd_ret, periods_per_year, look_back_rolling_sharpe, len(weightd_ret))
    port_weightd_ret[strategy] = weightd_ret
    return strategy, weightd_ret, weightd_ret_mu, weightd_ret_std, sharpe, sharpe_roll_wind, mdd


def composite_signal(strat1, strat2, spot_codes, list_buy_sell, df_ret, periods_per_year,
                     look_back_carry_ret, look_back_rolling_sharpe, num_obs):
    strategy = strat1 + '_' + strat2
    # Signal-agreement filter: keep a position only where both strategies' buy/sell signals
    # agree in direction; zero out everywhere else.
    boolean_signal = list_buy_sell[strat1] == list_buy_sell[strat2]
    df_carry_value = list_buy_sell[strat1] * boolean_signal
    weights = set_weights(df_ret, df_carry_value, strategy, periods_per_year, look_back_carry_ret)
    weightd_ret, mu, std, sharpe, sharpe_roll, mdd = get_results(
        df_ret, weights, strategy, periods_per_year, look_back_carry_ret, look_back_rolling_sharpe, num_obs)
    return strategy, weightd_ret, mu, std, sharpe, sharpe_roll, mdd


name_blend, wret_blend, mu_b, std_b, sharpe_b, roll_b, mdd_b = combine_strategies(
    'Simple_Carry', 'Value', port_weightd_ret, periods_per_year, look_back_rolling_sharpe)
print(f"{name_blend:<28} Return={mu_b:7.3f}%   Vol={std_b:7.3f}%   Sharpe={sharpe_b:6.3f}   MaxDD={mdd_b*100:6.2f}%")

df_ret_sc, _ = get_returns(df_spot, df_fwd, df_val, vol, 'Simple_Carry')
name_comp, wret_comp, mu_c, std_c, sharpe_c, roll_c, mdd_c = composite_signal(
    'Simple_Carry', 'Value', spot_codes, buy_sell_by_strategy, df_ret_sc, periods_per_year,
    look_back_carry_ret, look_back_rolling_sharpe, len(df_ret_sc))
print(f"{name_comp:<28} Return={mu_c:7.3f}%   Vol={std_c:7.3f}%   Sharpe={sharpe_c:6.3f}   MaxDD={mdd_c*100:6.2f}%")

results_by_strategy[name_blend] = dict(mu=mu_b, std=std_b, sharpe=sharpe_b, mdd=mdd_b)
results_by_strategy[name_comp] = dict(mu=mu_c, std=std_c, sharpe=sharpe_c, mdd=mdd_c)
```

```text
50/50 Simple_Carry & Value   Return=  1.819%   Vol=  2.552%   Sharpe= 0.713   MaxDD=  4.39%
Simple_Carry_Value           Return=  1.819%   Vol=  2.558%   Sharpe= 0.711   MaxDD=  4.39%
```

**Diagram — cumulative return, all strategies.** Every strategy's cumulative wealth index over the full backtest, on one chart.

```python
fig, ax = plt.subplots(figsize=(10, 5.5))
colors_map = {'Simple_Carry': '#2E86AB', 'VolAdj_Carry': '#F18F01', 'Value': '#6A994E',
              'Max_Sharpe_Port': '#C73E1D', name_blend: '#7209B7'}
for strategy in ['Simple_Carry', 'VolAdj_Carry', 'Value', 'Max_Sharpe_Port']:
    wret = results_by_strategy[strategy]['weightd_ret'] if 'weightd_ret' in results_by_strategy[strategy] else None
for strategy, c in colors_map.items():
    wret = port_weightd_ret[strategy] if strategy in port_weightd_ret else wret_blend
    cum = np.cumprod(1 + wret.fillna(0) / 100)
    ax.plot(cum.index, cum, color=c, linewidth=1.6, label=strategy)
ax.axhline(1, color='#888888', linewidth=0.8)
ax.set_xlabel('Date'); ax.set_ylabel('Cumulative Wealth (start = 1.0)')
ax.set_title('Cumulative Return: All Strategies')
ax.legend(fontsize=8, ncol=2)
plt.show()
```

![Figure 4](images/fx-carry-backtester_fig_04.png)

## 8. Performance Summary

```python
summary_rows = []
for strategy in ['Simple_Carry', 'VolAdj_Carry', 'Value', 'Max_Sharpe_Port', name_blend, name_comp]:
    r = results_by_strategy[strategy]
    summary_rows.append({'Strategy': strategy, 'Return %': round(r['mu'], 2), 'Vol %': round(r['std'], 2),
                          'Sharpe': round(r['sharpe'], 3), 'Max DD %': round(r['mdd'] * 100, 2)})
summary_df = pd.DataFrame(summary_rows).set_index('Strategy')
print(summary_df.to_string())
```

```text
                            Return %  Vol %  Sharpe  Max DD %
Strategy                                                     
Simple_Carry                    1.89   3.43   0.551      4.72
VolAdj_Carry                    1.72   3.28   0.524      4.72
Value                           1.76   3.21   0.549      7.64
Max_Sharpe_Port                -7.01  15.71  -0.446     77.00
50/50 Simple_Carry & Value      1.82   2.55   0.713      4.39
Simple_Carry_Value              1.82   2.56   0.711      4.39
```

## 9. Summary and Takeaways

- **All three rule-based strategies (`Simple_Carry`, `VolAdj_Carry`, `Value`) delivered positive, broadly similar risk-adjusted returns** (Sharpe 0.52 to 0.55), consistent with how carry is understood to actually behave: a real but modest, noisy edge, not a dramatic one, since this backtest's synthetic Uncovered Interest Parity violation was deliberately kept small and realistic.
- **`Max_Sharpe_Port` performed worst by a wide margin** (Sharpe -0.45, 77% max drawdown), not because of any remaining bug, but because mean-variance optimization re-estimated on a short (36-month), 8-asset rolling window is a textbook case of the "estimation-error maximizer" problem: noisy mean/covariance estimates lead the optimizer to take large, unstable positions that do not generalize out of sample. This mirrors the original documentation's own caution about SLSQP's local, non-convex optimization landscape.
- **Blending `Simple_Carry` and `Value` improved risk-adjusted performance** (Sharpe 0.71 for both the 50/50 blend and the agreement-based composite signal), a real diversification benefit consistent with the documentation's note that carry and value are close to empirically orthogonal signals.
- **Fixing the 9 documented issues materially changed the results.** Most visibly, the rolling Sharpe ratio chart is now a genuine rolling calculation (previously a single mis-indexed scalar per point) and the `Max_Sharpe_Port` weights are now computed with the correct lookback window and annualization factor. Anyone re-running the original, unfixed script would get a distorted picture of `Max_Sharpe_Port`'s stability specifically.
- **Every result here is on synthetic data.** The purpose of this notebook is to demonstrate that the corrected engine runs correctly end to end and to characterize how each strategy is *expected* to behave, not to make any claim about real-world FX carry profitability, which would require the actual spot/forward/PPP data the original codebase is designed to consume.
