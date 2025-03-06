import numpy as np
import scipy.optimize as sco
from data.stock_data import get_stock_data

def optimize_portfolio(tickers):
    """Optimize portfolio weights using Sharpe ratio."""
    df = get_stock_data(tickers)
    if df is None or df.empty:
        return None

    returns = df.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(tickers)
    
    def portfolio_performance(weights):
        ret = np.sum(weights * mean_returns)
        risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return ret, risk
    
    def neg_sharpe(weights):
        ret, risk = portfolio_performance(weights)
        return -ret / risk
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets])
    result = sco.minimize(neg_sharpe, initial_weights, bounds=bounds, constraints=constraints)
    
    return result.x