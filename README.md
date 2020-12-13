# portopt
Mean-variance portfolio optimization.


#### Usage
```python
from portopt import PortOpt


portfolio = PortOpt(assets='MutualFunds.csv')               # assets: filename of csv in assets directory

correlation = portfolio.correlation(startdate='2020-6-1', matrix_plot=True)
covariance  = portfolio.covariance(startdate='2020-6-1', matrix_plot=True)

portfolio_data, allocations = portfolio.optimize(
    n=1000, rf=0.0009, startdate='2020-6-1', ef_plot=True
)

```

##### `ef_plot`
###### Efficient frontier graph

##### `portfolio_data` 
###### Return, volatility & sharpe of optimized portfolio

##### `allocations`
###### Asset allocations of optimized portfolio

##### `matrix_plot`
###### Plot heatmap for correlation or covariance matrix

##### `correlation`
###### Correlation matrix 

##### `covariance`
###### Covariance matrix
