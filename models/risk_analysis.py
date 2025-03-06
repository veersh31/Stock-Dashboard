import numpy as np
import yfinance as yf

def analyze_risk(df, ticker):
    """Analyze risk metrics for a stock."""
    # Calculate risk metrics
    daily_returns = df['Close'].pct_change().dropna()
    
    # Volatility (annualized)
    volatility = daily_returns.std() * np.sqrt(252)
    
    # Beta (compared to S&P 500)
    spy_data = yf.Ticker("SPY").history(period="1y")
    spy_returns = spy_data['Close'].pct_change().dropna()
    
    # Align the data
    common_dates = daily_returns.index.intersection(spy_returns.index)
    stock_returns_aligned = daily_returns.loc[common_dates]
    spy_returns_aligned = spy_returns.loc[common_dates]
    
    # Calculate beta
    covariance = stock_returns_aligned.cov(spy_returns_aligned)
    spy_variance = spy_returns_aligned.var()
    beta = covariance / spy_variance if spy_variance != 0 else 1
    
    # Value at Risk (VaR) - 95% confidence
    var_95 = np.percentile(daily_returns, 5)
    
    # Maximum Drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio (assuming risk-free rate of 0.02)
    risk_free_rate = 0.02
    sharpe_ratio = (daily_returns.mean() * 252 - risk_free_rate) / volatility
    
    # Risk classification
    if volatility < 0.15:
        risk_level = "Low"
        risk_class = "risk-low"
    elif volatility < 0.25:
        risk_level = "Medium"
        risk_class = "risk-medium"
    else:
        risk_level = "High"
        risk_class = "risk-high"
    
    # Return risk metrics
    risk_metrics = {
        'ticker': ticker,
        'volatility': volatility,
        'beta': beta,
        'var_95': var_95,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'risk_level': risk_level,
        'risk_class': risk_class
    }
    
    return risk_metrics