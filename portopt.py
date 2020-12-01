import csv

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
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
            data[ticker] = pdr.DataReader(ticker, 'yahoo', startdate)['Adj Close']
        return data
    
    # daily log returns
    def _calc_returns(self, data):
        returns = np.log(data / data.shift(1))
        return returns
    
    # covariance table
    def covariance(self, startdate: str = '2020-6-1'):
        """params:
        startdate: year-month-day
        """
        data = self._get_price_data(startdate)
        returns = self._calc_returns(data)
        return returns.cov() * 252              # 252 trading days / yr on avg

    # correlation table
    def correlation(self, startdate: str = '2020-6-1'):
        """params:
        startdate: year-month-day 
        """
        data = self._get_price_data(startdate)
        returns = self._calc_returns(data)
        return returns.corr()
    
    # simulate n # of portfolio variations
    def _simulate(self, returns, n, rf):
        mean = returns.mean() * 252
        cov = returns.cov() * 252

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
 
    def optimize(self, n: int = 1000, rf: float = 0.0009, startdate: str = '2020-6-1', plot: bool = True):
        """params:
        n: # of portfolio simulations
        rf: risk-free rate
        startdate: year-month-day
        """
        data = self._get_price_data(startdate)
        returns = self._calc_returns(data)
        portfolios = self._simulate(returns, n, rf)

        # find optimal portfolio (max sharpe)
        portfolio_data = pd.DataFrame(portfolios.loc[portfolios['Sharpe'] == portfolios['Sharpe'].max()])
        x, y = portfolio_data['Volatility'], portfolio_data['Return']

        opt_weights = []
        for w in portfolio_data['Weights']:
            for _ in w:
                opt_weights.append(f'{round(_ * 100, 2)} %')
        del portfolio_data['Weights']
        opt_weights = list(zip(self.tickers, self.investment_names, opt_weights))
        allocations = pd.DataFrame(opt_weights, columns=['Ticker', 'Investment Name', 'Allocation'])
        
        if plot:
            _plot(portfolios, x, y)
        
        return portfolio_data, allocations

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
