import csv

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


class Optimize:
    """Efficient frontier analysis & asset allocation optimization"""

    def __init__(self, assets: str = 'assets.csv', n: int = 1000, rf: float = 0.0009, startdate: str = '2020-6-1'):
        """params:
        n: # of portfolio simulations
        rf: risk-free rate
        startdate: year-month-day"""
        self.n = n
        self.rf = rf
        self.startdate = startdate

        self.tickers, self.investment_names = _read_csv(assets)

    # daily adj. close price to dataframe
    def _get_price_data(self):
        data = pd.DataFrame()
        for ticker in self.tickers:
            data[ticker] = wb.DataReader(ticker, 'yahoo', self.startdate)['Adj Close']
        return data
    
    def _calc_returns(self, data):
        returns = np.log(data / data.shift(1))
        return returns
    
    # simulate portfolio variations
    def _simulate(self, returns):
        mean = returns.mean() * 252         # 252 trading days / yr on avg
        cov = returns.cov() * 252
        corr = returns.corr()
        # TODO store mean, cov, corr

        ret, vol, wgt = ([] for i in range(3))

        for _ in range(self.n):
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
        portfolios['sharpe'] = (portfolios['return'] - self.rf) / portfolios['volatility']

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

    def opt(self):
        data = self._get_price_data()
        returns = self._calc_returns(data)
        portfolios = self._simulate(returns)

        # find optimal portfolio (max sharpe)
        opt_port = pd.DataFrame(portfolios.loc[portfolios['sharpe'] == portfolios['sharpe'].max()])
        # TODO store opt_port


        opt_weights = []
        for w in opt_port['weights']:
            for _ in w:
                opt_weights.append(f'{round(_ * 100, 2)} %')
        opt_weights = list(zip(self.tickers, self.investment_names, opt_weights))
        opt = pd.DataFrame(opt_weights, columns=['ticker', 'investment name', 'allocation'])
        # TODO store opt
        print(opt)
           
        self._plot(portfolios)


if __name__ == '__main__':
    Optimize().opt()
