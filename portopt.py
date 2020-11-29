import csv
from os import path

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt


"""portopt
gr-satt"""


def _read_csv(file):
    with open(file, 'r') as assets:
        reader = csv.reader(assets)
        assets = {rows[0]: rows[1] for rows in reader}

    tickers, investment_names = assets.keys(), assets.values()
    return tickers, investment_names


class PortOpt:
    """Investment portfolio optimization"""

    def __init__(self, assets: str = 'assets.csv'):
        self.tickers, self.investment_names = _read_csv(assets)

    # daily adj. close price to dataframe
    def _get_price_data(self, startdate):
        data = pd.DataFrame()
        for ticker in self.tickers:
            data[ticker] = wb.DataReader(ticker, 'yahoo', startdate)['Adj Close']
        return data
    
    # daily log returns
    def _calc_returns(self, data):
        returns = np.log(data / data.shift(1))
        return returns
    
    # simulate n # of portfolio variations
    def _simulate(self, returns, n, rf):
        mean = returns.mean() * 252         # 252 trading days / yr on avg
        cov = returns.cov() * 252
        corr = returns.corr()
        print('\n\n', corr.to_string(index=True))

        ret, vol, wgt = ([] for i in range(3))

        for _ in range(n):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)

            ret.append(np.sum(weights * mean))
            vol.append(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
            wgt.append(weights)

        # create porfolios dataframe
        portfolios = pd.DataFrame({
            'Return': ret,
            'Volatility': vol,
            'Weights': wgt
        })
        # calc sharpe for each portfolio (row) in dataframe
        portfolios['Sharpe'] = (portfolios['Return'] - rf) / portfolios['Volatility']
        return portfolios
 
    def optimize(self, n: int = 1000, rf: float = 0.0009, startdate: str = '2020-6-1'):
        """params:
        n: # of portfolio simulations
        rf: risk-free rate
        startdate: year-month-day
        """
        data = self._get_price_data(startdate)
        returns = self._calc_returns(data)
        portfolios = self._simulate(returns, n, rf)

        # find optimal portfolio (max sharpe)
        opt_port = pd.DataFrame(portfolios.loc[portfolios['Sharpe'] == portfolios['Sharpe'].max()])
        print('\n\n', opt_port.to_string(index=False))
        x, y = opt_port['Volatility'], opt_port['Return']

        opt_weights = []
        for w in opt_port['Weights']:
            for _ in w:
                opt_weights.append(f'{round(_ * 100, 2)} %')
        opt_weights = list(zip(self.tickers, self.investment_names, opt_weights))
        opt = pd.DataFrame(opt_weights, columns=['Ticker', 'Investment Name', 'Allocation'])
        print('\n\n', opt.to_string(index=False))
           
        _plot(portfolios, x, y)

# plot EF
def _plot(data, x, y):
    # plot all portfolios
    plt.scatter(
        data['Volatility'],
        data['Return'],
        c=data['Sharpe'],
        marker='.'
    )
    # highlight best portfolio
    plt.scatter(x, y, marker='.', c='red')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()
