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
    
    tickers = assets.keys()
    invesment_names = assets.values()

    return tickers, invesment_names


class PortOpt:
    """Efficient frontier analysis & asset allocation optimization"""

    def __init__(self, assets: str = 'assets.csv'):
        self.tickers, self.investment_names = _read_csv(assets)

    # daily adj. close price to dataframe
    def _get_price_data(self, startdate):
        data = pd.DataFrame()
        for ticker in self.tickers:
            data[ticker] = wb.DataReader(ticker, 'yahoo', startdate)['Adj Close']
        return data
    
    def _calc_returns(self, data):
        returns = np.log(data / data.shift(1))
        return returns
    
    # simulate portfolio variations
    def _simulate(self, returns, n, rf):
        mean = returns.mean() * 252         # 252 trading days / yr on avg
        cov = returns.cov() * 252
        corr = returns.corr()
        print('\n\n', corr)
        
        cov.to_csv('bin/covariance.csv')
        corr.to_csv('bin/correlation.csv')

        ret, vol, wgt = ([] for i in range(3))

        for _ in range(n):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)

            ret.append(np.sum(weights * mean))
            vol.append(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
            wgt.append(weights)

        # create porfolios dataframe
        portfolios = pd.DataFrame({
            'return': ret,
            'volatility': vol,
            'weights': wgt
        })

        # calc sharpe for each portfolio (row) in dataframe
        portfolios['sharpe'] = (portfolios['return'] - rf) / portfolios['volatility']

        return portfolios
    
    # plot EF
    @staticmethod
    def _plot(data):
        plt.scatter(
            data['volatility'],
            data['return'],
            c=data['sharpe'],
            marker='.'
        )
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.show()

    def optimize(self, n: int = 1000, rf: float = 0.0009, startdate: str = '2020-6-1'):
        """params:
        n: # of portfolio simulations
        rf: risk-free rate
        startdate: year-month-day"""
        data = self._get_price_data(startdate)
        returns = self._calc_returns(data)
        portfolios = self._simulate(returns, n, rf)

        # find optimal portfolio (max sharpe)
        opt_port = pd.DataFrame(portfolios.loc[portfolios['sharpe'] == portfolios['sharpe'].max()])
        print('\n\n', opt_port)

        opt_weights = []
        for w in opt_port['weights']:
            for _ in w:
                opt_weights.append(f'{round(_ * 100, 2)} %')
        opt_weights = list(zip(self.tickers, self.investment_names, opt_weights))
        opt = pd.DataFrame(opt_weights, columns=['ticker', 'investment name', 'allocation'])
        print('\n\n', opt)
           
        self._plot(portfolios)
