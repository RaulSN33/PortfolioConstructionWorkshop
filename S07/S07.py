import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import statsmodels.api as sm

#%%

stock = 'ESGV'

#%%

mkt_info_us = yf.download([stock]+['^GSPC'], period = '10y', interval = '1mo', group_by='column')['Close']

#%%

mkt_info_us = mkt_info_us.pct_change().dropna().iloc[:-1]
mkt_info_us.head()


#%%

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

# Add zero reference lines
ax.axvline(0, alpha=0.35)
ax.axhline(0, alpha=0.35)

# Percent formatting
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Labels and title
ax.set_title(stock + ' Returns vs SP500 Returns', fontweight='bold')
ax.set_xlabel('SP500')
ax.set_ylabel(stock)

plt.tight_layout()
plt.show()

#%%
# Fred Data for the 10y rate time series
fred_key= '123'

mkt_info_us = mkt_info_us.resample('ME').last()
fred = Fred(api_key = fred_key)

rfr = fred.get_series('GS10')/100
rfr = rfr.resample('ME').last().loc[mkt_info_us.index[0]:mkt_info_us.index[-1]]

def resample_rate(value):
    return (1 + value) ** (1/12)-1

rfr_monthly = rfr.apply(resample_rate).ffill()

#%%

mkt_info_us = mkt_info_us.sub(rfr_monthly, axis = 'index')

X = mkt_info_us['^GSPC']
X = sm.add_constant(X)

y = mkt_info_us[stock]

#%%

model = sm.OLS(y, X).fit()
model.summary()

model.params

#%%

rolling_reg_results = {}
window_size = 24

for i in range(len(X) - window_size+1):

    day = X.index[i + window_size - 1]

    window_returns = X.iloc[i:i + window_size]
    window_market = y.iloc[i:i + window_size]

    model = sm.OLS(
        window_market,
        window_returns
    ).fit()

    model.predict()[-1]
    rolling_reg_results[day] = model.params.to_dict()
    rolling_reg_results[day]['factor_return'] = model.predict()[-1]



rolling_reg_results = pd.DataFrame(rolling_reg_results).T



#%%
rolling_reg_results.columns = ['alpha', 'beta', 'factor_return']


# Create figure and two subplots sharing x-axis
fig, (ax_beta, ax_alpha) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(10, 6)
)

fig.suptitle(f'{stock} Rolling CAPM for last 10 years')

# --- Top plot: Beta ---
ax_beta.plot(rolling_reg_results.index, rolling_reg_results['beta'])
ax_beta.axhline(y=1, color='black', linestyle='dotted')
ax_beta.set_title('Beta')
ax_beta.set_ylabel('Beta')

# --- Bottom plot: Alpha ---
ax_alpha.plot(rolling_reg_results.index, rolling_reg_results['alpha'])
ax_alpha.axhline(y=0, color='black', linestyle='dotted')
ax_alpha.set_title('Alpha')
ax_alpha.set_ylabel('Alpha')

# Improve spacing
plt.tight_layout()
plt.show()

#%%

return_comparison = pd.concat(
    [
        rolling_reg_results['factor_return'],
        mkt_info_us[stock].iloc[window_size:]
    ],
    axis = 1
).dropna()

index_returns = (1+return_comparison).cumprod()

index_returns.plot()
plt.show()
