#%%
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from fredapi import Fred
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import statsmodels.api as sm

#%%

INDUSTRY_ETF_MAP = {
    'Communication Services': 'VOX',
    'Consumer Discretionary': 'VCR',
    'Consumer Staples': 'VDC',
    'Energy': 'VDE',
    'Financials': 'VFH',
    'Health Care': 'VHT',
    'Industrials': 'VIS',
    'Information Technology': 'VGT',
    'Materials': 'VAW',
    'Real Estate': 'VNQ',
    'Utilities': 'VPU',
    'SP500': 'SPY',
}

etf_list = [value for (key, value) in INDUSTRY_ETF_MAP.items()]

def download_industry_etf_returns(tickers, start, end):
    """Live-download the sector proxy ETF closes and return simple returns."""
    prices = yf.download(list(tickers), start=start, end=end)['Close']

    return prices.pct_change()

today = date.today()
#%%

stock = 'ESGV'

# etf_returns = yf.download([stock]+['^GSPC'], period = '10y', interval = '1mo', group_by='column')['Close']
etf_returns = download_industry_etf_returns(
    etf_list,
    "2000-01-01",
    today.strftime("%Y-%m-%d")
)

#%%

#%%

#%%