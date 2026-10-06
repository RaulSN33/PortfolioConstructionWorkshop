import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import pandas as pd
    import numpy as np
    import yfinance as yf
    from datetime import date
    from fredapi import Fred
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    import statsmodels.api as sm


    return Fred, date, np, pd, plt, sm, yf


@app.cell
def _(Fred, date, np, pd, yf):
    FRED_API_KEY="0174cb93931388a2bf305663e4117fd3"
    FRED_SERIES_ID = "DGS10"
    TRADING_DAYS_PER_YEAR = 252
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

    def rfr_substraction(
            returns_df: pd.DataFrame
    ):
        if returns_df.index.dtype == 'O':
            returns_df.index = pd.to_datetime(returns_df.index, format = '%d/%m/%Y')

        returns_df_start = returns_df.index[0]
        returns_df_end = returns_df.index[-1]

        fred = Fred(api_key=FRED_API_KEY)
        rfr = fred.get_series(FRED_SERIES_ID) / 100

        if rfr.index.dtype == 'O':
            rfr.index = pd.to_datetime(rfr.index, format = '%d/%m/%Y')

        rfr_daily = rfr.loc[returns_df.index]
        rfr_daily = rfr_daily.apply(resample_rate)
        rfr_daily.name = 'rfr'

        universe_rets = returns_df.sub(
            rfr_daily.loc[returns_df_start:returns_df_end],
            axis=0
        )

        return universe_rets

    def resample_rate(value):
        return (1 + value) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    def wexp(N, half_life):
        """
        Exponential-decay weights for a window of N rows in ascending date order.

        The last element (the most recent row) gets the largest weight, and the
        weight halves every `half_life` rows further back. Weights sum to 1.
        """
        c = np.log(0.5)/half_life
        age = np.arange(N)[::-1]   # oldest row has age N-1, most recent has age 0
        w = np.exp(c*age)
        return w/np.sum(w)


    today = date.today()
    return (
        download_industry_etf_returns,
        etf_list,
        rfr_substraction,
        today,
        wexp,
    )


@app.cell
def _(download_industry_etf_returns, etf_list, today):
    stock = 'NVDA'
    start_date = "2018-01-01"
    end_date = today.strftime("%Y-%m-%d")
    # etf_returns = yf.download([stock]+['^GSPC'], period = '10y', interval = '1mo', group_by='column')['Close']
    etf_returns = download_industry_etf_returns(
        etf_list,
        start_date,
        end_date
    ).iloc[1:]

    etf_returns
    return end_date, etf_returns, start_date, stock


@app.cell
def _(download_industry_etf_returns, end_date, start_date, stock):
    stock_returns = download_industry_etf_returns(
        [stock],
        start_date,
        end_date

    ).iloc[1:]

    stock_returns
    return (stock_returns,)


@app.cell
def _(etf_returns, rfr_substraction, stock_returns):
    stock_returns_2 = rfr_substraction(
        stock_returns.iloc[:-1]
    )

    etf_returns_2 = rfr_substraction(
        etf_returns.iloc[:-1]
    ).fillna(0)
    return etf_returns_2, stock_returns_2


@app.cell
def _(etf_returns_2):
    etf_returns_2
    return


@app.cell
def _(plt, wexp):
    w = wexp(252, 252/2)
    plt.plot(w)
    plt.show()
    return (w,)


@app.cell
def _(etf_returns_2, pd, sm, stock_returns_2, w):
    betas_dict = {}

    etf_returns_3 = sm.add_constant(etf_returns_2)

    for i in range(252, len(etf_returns_2)):
    
        stock_rets_i = stock_returns_2.iloc[i-252:i]#.fillna(0)
        market_rets_i = etf_returns_3.iloc[i-252:i][["const", "SPY"]].fillna(0)

        index = stock_returns_2.index[i]

        print(index)

        mod_wls = sm.WLS(stock_rets_i, market_rets_i, weights=w)
        res_wls1 = mod_wls.fit()
        # print(res_wls.summary())
    
        residual_returns_i = res_wls1.resid

        last_sector_returns = etf_returns_3.iloc[i-252:i][
        [
            'const', 'VOX', 'VCR', 'VDC', 'VDE', 'VFH', 'VHT', 'VIS', 'VGT', 'VAW', 'VNQ', 'VPU'
        ]
        ]

        mod_wls2 = sm.WLS(residual_returns_i, last_sector_returns, weights=w)
        res_wls = mod_wls2.fit()
        # print(res_wls.summary())
    
        betas_dict[index] = res_wls.params
    # %%


    betas_df = pd.DataFrame(betas_dict).T

    return betas_df, market_rets_i


@app.cell
def _(betas_df, plt):
    betas_df.plot()
    plt.show()
    return


@app.cell
def _(betas_df):
    betas_df
    return


@app.cell
def _(market_rets_i):
    # market_rets_i[market_rets_i.isna()]
    market_rets_i.isna()
    return


@app.cell
def _(market_rets_i):
    market_rets_i
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
