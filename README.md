# portopt

```python
from portopt import PortOpt


portfolio = PortOpt(assets='assets.csv')

covariance = portfolio.covariance()
correlation = portfolio.correlation()

portfolio_data, allocations = portfolio.optimize(
    n=1000, rf=0.0009, startdate='2020-6-1', plot=True
)

```