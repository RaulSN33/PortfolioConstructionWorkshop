#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import yfinance as yf

#%%

def port_return(w, r):
    return w@r

def port_volatility(w,sigma):
    return np.sqrt(w@sigma@w)

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
    # 'IEV',
    # 'SPY',
    # 'QQQ',
]

stock_prices = yf.download(
    stocks,
    start='2015-01-01',
    interval='1mo',
)
#%%

stock_prices = stock_prices['Close']
returns = stock_prices.pct_change()

first = stock_prices.iloc[1]
last = stock_prices.iloc[-1]
total_months = stock_prices.shape[0]
#%%

miu = returns.mean()#*12
# miu = (last/first)**(12/total_months)-1

#%%

sigma = returns.cov()

#%%

std_matrix = sigma.values
std_matrix = np.sqrt(np.diag(std_matrix))
std_dev = returns.std()#*np.sqrt(252)
#%%

"""
Global minimum variance

w_gmv = 
  
      ones' @ Sigma ^-1
  ------------------------
  ones' @ Sigma ^-1 @ ones

"""

ones = np.ones(len(returns.columns))
sigma_inv = np.linalg.pinv(sigma)

num = sigma_inv@ones
denom = ones@sigma_inv@ones

portfolio_gmv = num / denom


#%%

a = ones@sigma_inv@ones # sum(sigma * 1) ==  u = linalg.solve(sigma, ones)
b = ones@sigma_inv@miu # z = linalg.solve(sigma, miu)
c = miu@sigma_inv@miu  # miu @ z

#%%

"""
Max Sharpe

w_gmv = 

       Sigma ^-1 @ miu
  ------------------------
  ones' @ Sigma ^-1 @ miu

"""
portfolio_max_sharpe = (sigma_inv@miu)/(ones@sigma_inv@miu)

#%%


min_vol = np.sqrt(1/a)
step = 0.001
num = 65
range_vols = np.linspace(min_vol, min_vol+step*num, num = num)

range_vols

#%%

result_miu = b/a + np.sqrt(c-b**2/a)*np.sqrt(range_vols**2 - 1/a)
result_miu

#%%

plt.scatter(x = range_vols, y = result_miu)
plt.show()

#%%

gmv_miu = port_return(portfolio_gmv, miu)
gmv_std = port_volatility(portfolio_gmv, sigma)


max_sharpe_miu = port_return(portfolio_max_sharpe, miu)
max_sharpe_std = port_volatility(portfolio_max_sharpe, sigma)

#%%

fig, ax = plt.subplots()

# Asset points
ax.scatter(
    x=std_dev,
    y=miu,
    label="Assets"
)

# GMV / 1N Portfolio point
ax.scatter(
    x=gmv_std,
    y=gmv_miu,
    marker='*',
    color='orange',
    s=200,
    label="GMV"
)
ax.scatter(
    x=max_sharpe_std,
    y=max_sharpe_miu,
    marker='*',
    color='green',
    s=200,
    label="Max Sharpe"
)

# Efficient frontier line (FIXED)
ax.plot(
    range_vols,
    result_miu,
    linewidth=2,
    label="Efficient Frontier",
    color = 'black',
    alpha = 0.5
)

# Annotate portfolio
ax.annotate(
    'GMV',
    (gmv_std, gmv_miu),
    textcoords="offset points",
    xytext=(6, 6),
    fontweight='bold',
    color='orange'
)
# Annotate portfolio
ax.annotate(
    'Max Sharpe',
    (max_sharpe_std, max_sharpe_miu),
    textcoords="offset points",
    xytext=(6, 6),
    fontweight='bold',
    color='green'
)

# Annotate individual assets
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

# Percent formatting
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

ax.grid(alpha=0.5)
ax.legend()

fig.suptitle('Monthly Historical risk-return tradeoff')
fig.tight_layout()

plt.show()


#%%
rf_rate = 0.001# * (1/12)
ex_returns = miu - rf_rate

u = np.linalg.solve(sigma, ex_returns)  # Sigma^(-1) * mu_tilda
w_tan = u / u.sum()
w_tan


tan_miu = port_return(w_tan, miu)
tan_std = port_volatility(w_tan, sigma)

fig, ax = plt.subplots()


ax.scatter(
    x=std_dev,
    y=miu,
    label="Assets"
)


ax.scatter(
    x=gmv_std,
    y=gmv_miu,
    marker='*',
    color='orange',
    s=200,
    label="GMV"
)


ax.scatter(
    x=tan_std,
    y=tan_miu,
    marker='*',
    color='green',
    s=200,
    label="Max Sharpe"
)


ax.scatter(
    0,
    rf_rate,
    color='red',
    s=120,
    zorder=5,
    label='Risk-Free Rate'
)


ax.plot(
    range_vols,
    result_miu,
    linewidth=2,
    color='black',
    alpha=0.5,
    label="Efficient Frontier"
)

cml_slope = (tan_miu - rf_rate) / tan_std

# Use same volatility range as frontier
x_vals = np.linspace(0, max(range_vols)*1.1, 200)
y_vals = rf_rate + cml_slope * x_vals

ax.plot(
    x_vals,
    y_vals,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Capital Market Line'
)

ax.annotate(
    'GMV',
    (gmv_std, gmv_miu),
    textcoords="offset points",
    xytext=(6, 6),
    fontweight='bold',
    color='orange'
)

ax.annotate(
    'Tangency Portfolio',
    (tan_std, tan_miu),
    textcoords="offset points",
    xytext=(6, 6),
    fontweight='bold',
    color='green'
)

ax.annotate(
    'Risk-Free Rate',
    (0, rf_rate),
    textcoords="offset points",
    xytext=(6, 6),
    fontweight='bold',
    color='red'
)

ax.set_xlabel("Volatility (Standard Deviation)")
ax.set_ylabel("Monthly Return")

ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

ax.grid(alpha=0.5)
ax.legend()

fig.suptitle('Monthly Historical Risk-Return Tradeoff')
fig.tight_layout()

plt.show()