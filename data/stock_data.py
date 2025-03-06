import yfinance as yf
import pandas as pd
import numpy as np

def get_stock_data(tickers, period='1y'):
    """Fetch stock data for one or more tickers."""
    try:
        # If a single ticker is passed, convert to list
        if isinstance(tickers, str):
            tickers = [tickers]
        
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
            
            data[ticker] = df['Close']
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def get_dashboard_stock_data(ticker):
    """Fetch stock information and historical data for dashboard."""
    try:
        # Fetch stock information
        stock = yf.Ticker(ticker)
        
        # Fetch historical data for 6 months
        history = stock.history(period="6mo")

        # Check if data is available
        if history.empty:
            return None, None, "No historical data available for this ticker"

        return stock, history, None
    except Exception as e:
        return None, None, str(e)

def get_full_stock_data(ticker, period='2y'):
    """Fetch full stock data with technical indicators for ML."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Calculate technical indicators
        # Moving Averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        
        # Volume indicators
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Price_Change_20d'] = df['Close'].pct_change(periods=20)
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    except Exception as e:
        print(f"Error fetching full stock data: {e}")
        return None