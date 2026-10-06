import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import statsmodels.api as sm

#%%

stock = 'QQQ'

#%%
# --- DATA DOWNLOAD & RETURNS ---
# Same as S07: download stock + SP500, compute monthly returns

mkt_info_us = yf.download([stock]+['^GSPC'], period = '10y', interval = '1mo', group_by='column')['Close']

#%%

mkt_info_us = mkt_info_us.pct_change().dropna().iloc[:-1]
mkt_info_us.head()

#%%
# --- SCATTER PLOT: stock vs market ---

sns.set_style('whitegrid')

fig, ax = plt.subplots(figsize=(7, 7))

sns.regplot(
    x=mkt_info_us['^GSPC'],
    y=mkt_info_us[stock],
    ax=ax,
    truncate=False,
    line_kws={'color': 'orange', 'alpha': 0.7},
    scatter_kws={'alpha': 0.6},
    color="b",
)

ax.axvline(0, alpha=0.35)
ax.axhline(0, alpha=0.35)

ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

ax.set_title(stock + ' Returns vs SP500 Returns', fontweight='bold')
ax.set_xlabel('SP500')
ax.set_ylabel(stock)

plt.tight_layout()
plt.show()

#%%
# --- RISK FREE RATE FROM FRED ---
# We subtract the risk-free rate to work with excess returns
# Same as S07

fred_key = '0123'

mkt_info_us = mkt_info_us.resample('ME').last()
fred = Fred(api_key=fred_key)

rfr = fred.get_series('GS10') / 100
rfr = rfr.resample('ME').last().loc[mkt_info_us.index[0]:mkt_info_us.index[-1]]

def resample_rate(value):
    return (1 + value) ** (1/12) - 1

rfr_monthly = rfr.apply(resample_rate).ffill()

#%%
# --- EXCESS RETURNS ---

mkt_info_us = mkt_info_us.sub(rfr_monthly, axis='index')

X = mkt_info_us['^GSPC']
X = sm.add_constant(X)

y = mkt_info_us[stock]

#%%
# --- STATIC OLS: full sample regression ---

model = sm.OLS(y, X).fit()
model.summary()

model.params

#%%
# =============================================================================
# BLOCK 2 — ROLLING REGRESSION + RETURN & RISK DECOMPOSITION
# =============================================================================
# For each rolling window we store:
#   - alpha, beta                        (regression params)
#   - factor_return                      (fitted value at last obs of window)
#   - total_var                          (variance of stock excess returns in window)
#   - systematic_var                     (variance explained by the factor = total - idio)
#   - idiosyncratic_var                  (variance of OLS residuals)
#   - r_squared                          (from model directly)
#
# The key insight: residuals from OLS ARE the idiosyncratic returns,
# so their variance is the idiosyncratic variance. No formula needed beyond model.resid.

rolling_reg_results = {}
window_size = 24

for i in range(len(X) - window_size + 1):

    day = X.index[i + window_size - 1]

    window_X = X.iloc[i:i + window_size]
    window_y = y.iloc[i:i + window_size]

    model = sm.OLS(window_y, window_X).fit()

    # --- Return decomposition ---
    factor_return = model.predict()[-1]

    # --- Risk decomposition ---
    # Total variance of the stock excess returns in this window
    total_var = window_y.var()

    # Idiosyncratic variance = variance of the residuals (what the factor does NOT explain)
    idio_var = model.resid.var()

    # Systematic variance = what is left (explained by the factor)
    systematic_var = total_var - idio_var

    # R-squared: how much of total variance is explained by the factor
    r_squared = model.rsquared

    # --- Store everything ---
    rolling_reg_results[day] = {
        'alpha'           : model.params['const'],
        'beta'            : model.params['^GSPC'],
        'factor_return'   : factor_return,
        'total_var'       : total_var,
        'systematic_var'  : systematic_var,
        'idiosyncratic_var': idio_var,
        'r_squared'       : r_squared,
    }

rolling_reg_results = pd.DataFrame(rolling_reg_results).T

#%%
# =============================================================================
# BLOCK 3 — IDIOSYNCRATIC RETURN & CUMULATIVE RETURN DECOMPOSITION
# =============================================================================
# factor_return   : the return explained by market exposure (fitted value)
# actual_return   : the real excess return of the stock
# idio_return     : what the factor model did NOT explain = actual - factor
#
# Cumulative sum lets us see how each component accumulated over time

actual_return = y.iloc[window_size - 1:].rename('actual_return')

return_decomp = pd.DataFrame({
    'actual_return'   : actual_return,
    'factor_return'   : rolling_reg_results['factor_return'],
}).dropna()

# Idiosyncratic return is the residual between actual and factor
return_decomp['idio_return'] = return_decomp['actual_return'] - return_decomp['factor_return']

#%%
# Cumulative sum of each component
# cumsum() here shows how each piece has been accumulating — not compounding,
# but useful to see the direction and magnitude of each source of return

cumsum_decomp = return_decomp.cumsum()

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(cumsum_decomp.index, cumsum_decomp['actual_return'],  label='Actual Return',   color='black',  linewidth=1.5)
ax.plot(cumsum_decomp.index, cumsum_decomp['factor_return'],  label='Factor Return',   color='steelblue', linewidth=1.5)
ax.plot(cumsum_decomp.index, cumsum_decomp['idio_return'],    label='Idiosyncratic Return', color='orange', linewidth=1.5)

ax.axhline(0, color='black', linestyle='dotted', alpha=0.5)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title(f'{stock} — Cumulative Return Decomposition (Rolling {window_size}m CAPM)', fontweight='bold')
ax.set_ylabel('Cumulative Return')
ax.legend()

plt.tight_layout()
plt.show()

#%%
# =============================================================================
# BLOCK 4 — PLOTS
# =============================================================================

# --- Plot 1: Rolling Alpha & Beta (same as S07) ---

fig, (ax_beta, ax_alpha) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

fig.suptitle(f'{stock} — Rolling CAPM Parameters ({window_size}m window)', fontweight='bold')

ax_beta.plot(rolling_reg_results.index, rolling_reg_results['beta'], color='steelblue')
ax_beta.axhline(y=1, color='black', linestyle='dotted')
ax_beta.set_title('Beta')
ax_beta.set_ylabel('Beta')

ax_alpha.plot(rolling_reg_results.index, rolling_reg_results['alpha'], color='orange')
ax_alpha.axhline(y=0, color='black', linestyle='dotted')
ax_alpha.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax_alpha.set_title('Alpha')
ax_alpha.set_ylabel('Alpha')

plt.tight_layout()
plt.show()

#%%
# --- Plot 2: Stacked area — % contribution of factor vs idio to total return ---
# For each period, we look at the absolute contribution of each component
# and express it as a share of total absolute return
# This shows when the factor was "driving" the stock vs when idio was dominant

abs_total = return_decomp[['factor_return', 'idio_return']].abs().sum(axis=1)

pct_factor = return_decomp['factor_return'].abs() / abs_total
pct_idio   = return_decomp['idio_return'].abs()   / abs_total

fig, ax = plt.subplots(figsize=(12, 5))

ax.stackplot(
    return_decomp.index,
    pct_factor,
    pct_idio,
    labels=['Factor (Market)', 'Idiosyncratic'],
    colors=['steelblue', 'orange'],
    alpha=0.7
)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title(f'{stock} — Return Attribution: Factor vs Idiosyncratic (% of abs. total)', fontweight='bold')
ax.set_ylabel('Share of Total Return')
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()

#%%
# --- Plot 3: Stacked area — % of variance explained by factor vs idio ---
# R-squared IS the share of systematic variance, so we can use it directly.
# This is the cleanest way to show risk decomposition over time:
# when R² is high, the stock is mostly "market-driven"
# when R² is low, idiosyncratic risk is dominant

fig, ax = plt.subplots(figsize=(12, 5))

ax.stackplot(
    rolling_reg_results.index,
    rolling_reg_results['r_squared'],
    1 - rolling_reg_results['r_squared'],
    labels=['Systematic Risk (R²)', 'Idiosyncratic Risk (1 - R²)'],
    colors=['steelblue', 'orange'],
    alpha=0.7
)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title(f'{stock} — Risk Decomposition: Systematic vs Idiosyncratic ({window_size}m rolling)', fontweight='bold')
ax.set_ylabel('Share of Total Variance')
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()