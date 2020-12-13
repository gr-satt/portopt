from portopt import PortOpt


portfolio = PortOpt(assets='MutualFunds.csv')               # assets: filename of csv in assets directory

correlation = portfolio.correlation(startdate='2020-6-1', matrix_plot=True)
covariance  = portfolio.covariance(startdate='2020-6-1', matrix_plot=True)

portfolio_data, allocations = portfolio.optimize(
    n=1000, rf=0.0009, startdate='2020-6-1', ef_plot=True
)


# display
for _ in [correlation, covariance, portfolio_data, allocations]:
    print('\n\n', _)
