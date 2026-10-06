import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import statsmodels.api as sm
#%% =============================================================================
# CONFIGURATION
# =============================================================================

stock = 'SOXX'
window_size = 24          # rolling window in months
FF_FILE = 'kn_portfolio_w/ff-factor_data_weekly.csv'   # Fama-French CSV in the same folder

# =============================================================================
# BLOCK 1 — DATA LOADING
# =============================================================================
# ── 1a. Stock prices from yfinance ───────────────────────────────────────────
prices = yf.download([stock], period='10y', interval='1d', group_by='column')['Close']
# ret_raw = prices.pct_change().dropna().iloc[:-1]   # monthly simple returns


ff = pd.read_csv(
    FF_FILE,
    sep=',',            # adjust sep=';' if your file uses semicolons
    index_col=0,

)
ff.index = pd.to_datetime(ff.index, format='%Y%m%d')
# ff.columns = ff.columns.str.strip()
ff = ff / 100.0         # convert percentages → decimals


ff = ff.loc[prices.index[0]:prices.index[-1]]
prices = prices.loc[ff.index, :]


ret_raw = prices.pct_change().iloc[1:]


# Keep only the rows that overlap with our stock returns
common_idx = ret_raw.index.intersection(ff.index)
ret = ret_raw.loc[common_idx]
ff = ff.loc[common_idx]

# ── 1c. Excess returns of the stock ──────────────────────────────────────────
# FF already provides RF; subtract it from stock return
xret = ret.sub(ff['RF'], axis = 0)      # stock excess return

# Factors (already in decimals)
mkt = ff['Mkt-RF']
smb = ff['SMB']
hml = ff['HML']

# ── 1d. Build regressor matrix for Fama-French 3-factor model ─────────────────
X3 = sm.add_constant(pd.DataFrame({
    'Mkt-RF': mkt,
    'SMB'   : smb,
    'HML'   : hml,
}))

#%% =============================================================================
# BLOCK 2 — STATIC OLS (full sample) — sanity check
# =============================================================================

model_static = sm.OLS(xret, X3).fit()
print(model_static.summary())
print("\nStatic betas:")
print(model_static.params)

# =============================================================================
# BLOCK 3 — ROLLING REGRESSION
# =============================================================================
# For each window we store:
#   alpha, beta_mkt, beta_smb, beta_hml          (params)
#   mkt_return, smb_return, hml_return, alpha_t  (period return contributions)
#   total_var, sys_var, idio_var, r_squared       (risk decomposition)
#   factor_var_mkt, factor_var_smb, factor_var_hml (per-factor variance contributions)
#
# RISK DECOMPOSITION NOTE
# ────────────────────────
# From the covariance decomposition shown in the slides:
#   Σ = βΩβ' + Σ_ids
# For a single stock (scalar case):
#   Var(r_i) = β'Ωβ + σ²_ids
# where Ω is the 3×3 covariance matrix of the factors in the window.
#
# We can further attribute the systematic variance to each factor by expanding β'Ωβ:
#   Var_sys = Σ_k Σ_j  β_k * β_j * Cov(f_k, f_j)
#
# The diagonal contribution of factor k is: β_k² * Var(f_k)
# The cross-factor terms are shared across pairs. A clean per-factor attribution
# assigns to factor k its diagonal term PLUS half of each cross term it participates in.
# This ensures the three factor contributions sum exactly to the total systematic variance.

rolling_results = {}

for i in range(len(xret) - window_size + 1):

    day = xret.index[i + window_size - 1]

    w_y  = xret.iloc[i:i + window_size]
    w_X  = X3.iloc[i:i + window_size]

    model = sm.OLS(w_y, w_X).fit()

    b_mkt = model.params['Mkt-RF']
    b_smb = model.params['SMB']
    b_hml = model.params['HML']
    alpha  = model.params['const']

    # ── Return decomposition for the LAST observation in the window ──────────
    # Actual factor realisation at period t (last obs of window)
    last_mkt = mkt.iloc[i + window_size - 1]
    last_smb = smb.iloc[i + window_size - 1]
    last_hml = hml.iloc[i + window_size - 1]

    # Contribution of each factor to that period's return
    ret_alpha = alpha
    ret_mkt   = b_mkt * last_mkt
    ret_smb   = b_smb * last_smb
    ret_hml   = b_hml * last_hml

    # Total factor return (fitted value at last obs)
    factor_return = ret_alpha + ret_mkt + ret_smb + ret_hml

    # ── Risk decomposition using the covariance formula Var = β'Ωβ + σ²_ids ─
    w_factors = w_X[['Mkt-RF', 'SMB', 'HML']]
    Omega = w_factors.cov()         # 3×3 factor covariance matrix in this window

    betas = np.array([b_mkt, b_smb, b_hml])#.reshape((3,1))

    # b = betas.T@w

    # Total systematic variance: β'Ωβ  (scalar)
    beta_sigma = betas @ Omega#.values
    contribs = beta_sigma*betas

    sys_var = sum(contribs)
    # sys_var = float(betas @ Omega.values @ betas)

    # Idiosyncratic variance: variance of OLS residuals
    idio_var  = model.resid.var()

    # Total variance = systematic + idiosyncratic
    total_var = sys_var + idio_var

    # ── Per-factor variance attribution ──────────────────────────────────────

    r_squared = model.rsquared

    rolling_results[day] = {
        'alpha'      : alpha,
        'beta_mkt'   : b_mkt,
        'beta_smb'   : b_smb,
        'beta_hml'   : b_hml,
        # Return components
        'ret_alpha'  : ret_alpha,
        'ret_mkt'    : ret_mkt,
        'ret_smb'    : ret_smb,
        'ret_hml'    : ret_hml,
        'factor_return': factor_return,
        # Risk components
        'total_var'  : total_var,
        'sys_var'    : sys_var,
        'idio_var'   : idio_var,
        'var_mkt'    : contribs['Mkt-RF'],
        'var_smb'    : contribs['SMB'],
        'var_hml'    : contribs['HML'],
        'r_squared'  : r_squared,
    }

    # break


rolling_results = pd.DataFrame(rolling_results).T
# Index is already datetime (inherited from xret which uses yfinance/FF dates directly)

actual_return = xret.iloc[window_size - 1:]#.rename('actual_return')
actual_return.columns = ['actual_return']
# =============================================================================
# BLOCK 4 — RETURN DECOMPOSITION
# =============================================================================
# For each period:
#   actual_return  = true excess return of the stock
#   factor_return  = alpha + β_mkt*mkt_t + β_smb*smb_t + β_hml*hml_t
#   idio_return    = actual - factor  (residual not explained by factors)
#   ret_alpha      = alpha (constant skill contribution)
#   ret_mkt        = β_mkt * mkt_t
#   ret_smb        = β_smb * smb_t
#   ret_hml        = β_hml * hml_t
# return_decomp = {
#     'actual_return': actual_return,
#     'factor_return': rolling_results['factor_return'],
#     'ret_alpha'    : rolling_results['ret_alpha'],
#     'ret_mkt'      : rolling_results['ret_mkt'],
#     'ret_smb'      : rolling_results['ret_smb'],
#     'ret_hml'      : rolling_results['ret_hml'],
# }
# return_decomp = pd.DataFrame(return_decomp)

rolling_results['actual_return'] = actual_return.copy()
rolling_results['idio_return'] = rolling_results['actual_return'].sub(rolling_results['factor_return'], axis = 0)
#%% =============================================================================
# BLOCK 5 — PLOTS
# =============================================================================

sns.set_style('whitegrid')
COLORS = {
    'actual' : 'black',
    'mkt'    : 'steelblue',
    'smb'    : '#e67e22',
    'hml'    : '#27ae60',
    'alpha'  : '#8e44ad',
    'idio'   : '#e74c3c',
    'factor' : '#2980b9',
}

# ── Plot 1: Cumulative return decomposition ───────────────────────────────────
cumsum = rolling_results[['actual_return', 'factor_return', 'ret_mkt', 'ret_smb', 'ret_hml', 'ret_alpha', 'idio_return']].cumsum()

fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(cumsum.index, cumsum['actual_return'], label='Actual Return',    color=COLORS['actual'],  linewidth=2.0)
ax.plot(cumsum.index, cumsum['factor_return'], label='Total FF Return',  color=COLORS['factor'],  linewidth=1.5, linestyle='--')
ax.plot(cumsum.index, cumsum['ret_mkt'],       label='Market (Mkt-RF)',  color=COLORS['mkt'],     linewidth=1.4)
ax.plot(cumsum.index, cumsum['ret_smb'],       label='Size (SMB)',       color=COLORS['smb'],     linewidth=1.4)
ax.plot(cumsum.index, cumsum['ret_hml'],       label='Value (HML)',      color=COLORS['hml'],     linewidth=1.4)
ax.plot(cumsum.index, cumsum['ret_alpha'],     label='Alpha',            color=COLORS['alpha'],   linewidth=1.4)
ax.plot(cumsum.index, cumsum['idio_return'],   label='Idiosyncratic',    color=COLORS['idio'],    linewidth=1.4)

ax.axhline(0, color='black', linestyle='dotted', alpha=0.4)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title(f'{stock} — Cumulative Return Decomposition (Rolling {window_size}d Fama-French 3-Factor)', fontweight='bold')
ax.set_ylabel('Cumulative Return')
ax.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.show()

#%% ── Plot 2: Stacked area — return attribution (% of abs. total) ──────────────
abs_components = rolling_results[['ret_mkt', 'ret_smb', 'ret_hml', 'ret_alpha', 'idio_return']].abs()
abs_total      = abs_components.sum(axis=1)

pct = abs_components.div(abs_total, axis=0)

fig, ax = plt.subplots(figsize=(13, 5))

ax.stackplot(
    rolling_results.index,
    pct['ret_mkt'],
    pct['ret_smb'],
    pct['ret_hml'],
    pct['ret_alpha'],
    pct['idio_return'],
    labels=['Market (Mkt-RF)', 'Size (SMB)', 'Value (HML)', 'Alpha', 'Idiosyncratic'],
    colors=[COLORS['mkt'], COLORS['smb'], COLORS['hml'], COLORS['alpha'], COLORS['idio']],
    alpha=0.75,
)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title(f'{stock} — Return Attribution by Factor (% of abs. total, Rolling {window_size}d)', fontweight='bold')
ax.set_ylabel('Share of Total Return')
ax.legend(loc='upper left', fontsize=8, ncol=3)
plt.tight_layout()
plt.show()


#%%── Plot 3: Risk decomposition — variance per factor + idio (stacked area) ───
# Uses the covariance formula: Var(r_i) = β'Ωβ + σ²_ids
# β'Ωβ is split into per-factor contributions (diagonal + shared cross terms)
# This is more rigorous than using R² because it accounts for factor correlations.

risk_df = rolling_results[['var_mkt', 'var_smb', 'var_hml', 'idio_var', 'total_var']].copy()

# Express each component as % of total variance
risk_pct = risk_df[['var_mkt', 'var_smb', 'var_hml', 'idio_var']].div(risk_df['total_var'], axis=0)

fig, ax = plt.subplots(figsize=(13, 5))

ax.stackplot(
    rolling_results.index,
    risk_pct['var_mkt'],
    risk_pct['var_smb'],
    risk_pct['var_hml'],
    risk_pct['idio_var'],
    labels=['Market Risk (Mkt-RF)', 'Size Risk (SMB)', 'Value Risk (HML)', 'Idiosyncratic Risk'],
    colors=[COLORS['mkt'], COLORS['smb'], COLORS['hml'], COLORS['idio']],
    alpha=0.75,
)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title(
    f'{stock} — Risk Decomposition via Covariance Formula Σ=βΩβ\' + Σ_ids  (Rolling {window_size}d)',
    fontweight='bold'
)
ax.set_ylabel('Share of Total Variance')
ax.legend(loc='upper left', fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

#%% ── Plot 4: Rolling betas ─────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(13, 10))
fig.suptitle(f'{stock} — Rolling FF3 Parameters ({window_size}d window)', fontweight='bold')

params = [
    ('beta_mkt', 'Market Beta (Mkt-RF)', COLORS['mkt'],   1.0),
    ('beta_smb', 'Size Beta (SMB)',       COLORS['smb'],   0.0),
    ('beta_hml', 'Value Beta (HML)',      COLORS['hml'],   0.0),
    ('alpha',    'Alpha',                 COLORS['alpha'], 0.0),
]

for ax, (col, title, color, hline) in zip(axes, params):
    ax.plot(rolling_results.index, rolling_results[col], color=color)
    ax.axhline(y=hline, color='black', linestyle='dotted', alpha=0.5)
    ax.set_title(title)
    if col == 'alpha':
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.show()

#%% ── Plot 5: Absolute variance levels over time ────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 5))

ax.stackplot(
    rolling_results.index,
    rolling_results['var_mkt'],
    rolling_results['var_smb'],
    rolling_results['var_hml'],
    rolling_results['idio_var'],
    labels=['Market Risk', 'Size Risk', 'Value Risk', 'Idiosyncratic Risk'],
    colors=[COLORS['mkt'], COLORS['smb'], COLORS['hml'], COLORS['idio']],
    alpha=0.75,
)

ax.set_title(f'{stock} — Absolute Variance Decomposition by Factor (Rolling {window_size}d)', fontweight='bold')
ax.set_ylabel('Variance')
ax.legend(loc='upper left', fontsize=8, ncol=2)
plt.tight_layout()
plt.show()



