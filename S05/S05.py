#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import yfinance as yf

#%%

stocks = [
    'F',
    'PFE',
    'CVX',
    'WMT',
    'SBUX',
    'AMZN',
    'DIS',
    'COST',
    'MMM',
    'HP'
]

stock_prices = yf.download(
    stocks,
    start='2015-01-01',
    interval='1mo',
)
#%%

stock_prices = stock_prices['Close']
returns = stock_prices.pct_change()

#%%

miu = returns.mean()

#%%

sigma = returns.cov()

#%%

std_matrix = sigma.values
std_matrix = np.diag(std_matrix)**(1/2)

#%%


std_dev = returns.std()


#%%

fig, ax = plt.subplots()

ax.scatter(
    x=std_dev,
    y=miu,
)

for ticker in miu.index:
    ax.annotate(
        ticker,
        (std_dev.loc[ticker], miu.loc[ticker]),
        textcoords="offset points",
        xytext=(5, 5),
    )

ax.set_xlabel("Volatility (Standard Deviation)")
ax.set_ylabel("Monthly Return")

ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.grid(alpha = 0.5)

fig.suptitle('Monthly Historical risk-return tradeoff')
plt.show()

#%%

n_components = len(stocks)
# portfolio_w = np.repeat(1/n_components, n_components)

portfolio_w = np.array([
    0.05,  # AMZN
    0.05,  # COST
    0.5,  # CVX
    0.15,  # DIS
    0.55,  # F
    0.05,  # HP
    0.05,  # MMM
    -0.5,  # PFE
    0.05,  # SBUX
    0.05,  # WMT
])

sum(portfolio_w)

#%%

portfolio_miu = portfolio_w @ miu

portfolio_std = portfolio_w @ sigma @ portfolio_w
portfolio_std = portfolio_std ** (1/2)

#%%

fig, ax = plt.subplots()

ax.scatter(
    x=std_dev,
    y=miu,
)

ax.scatter(
    x=portfolio_std,
    y=portfolio_miu,
    marker='*',
    color = 'orange',
    s = 200
)

ax.annotate(
    'Portfolio',
    (portfolio_std, portfolio_miu),
    textcoords="offset points",
    xytext=(5, 5),
)

for ticker in miu.index:
    ax.annotate(
        ticker,
        (std_dev.loc[ticker], miu.loc[ticker]),
        textcoords="offset points",
        xytext=(5, 5),
    )

# Axis labels
ax.set_xlabel("Volatility (Standard Deviation)")
ax.set_ylabel("Monthly Return")

ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.grid(alpha = 0.5)

fig.suptitle('Monthly Historical risk-return tradeoff')
plt.show()

#%%

corr = returns.corr()

sns.heatmap(
    corr
)

plt.show()