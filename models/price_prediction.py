import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime

def prepare_prediction_data(df, prediction_days=7):
    """Prepare data for ML prediction."""
    # Features for prediction
    features = ['Close', 'MA5', 'MA20', 'MA50', 'MACD', 'Signal_Line', 'RSI', 
                'BB_Upper', 'BB_Lower', 'Volume_Change', 'Price_Change', 
                'Price_Change_5d', 'Price_Change_20d', 'Volatility']
    
    # Create target variable (future price)
    df['Target'] = df['Close'].shift(-prediction_days)
    
    # Drop rows with NaN in target
    df = df.dropna(subset=['Target'])
    
    # Split data
    X = df[features]
    y = df['Target']
    
    # Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, features

def train_price_prediction_model(X_train, y_train):
    """Train a price prediction model."""
    # Train Gradient Boosting Regressor
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_future_price(df, model, scaler_X, scaler_y, features, prediction_days):
    """Predict future price based on the trained model."""
    # Prepare last data point for prediction
    last_data = df[features].iloc[-1:].values
    last_data_scaled = scaler_X.transform(last_data)
    future_pred_scaled = model.predict(last_data_scaled)
    future_pred = scaler_y.inverse_transform(future_pred_scaled.reshape(-1, 1)).flatten()[0]
    
    # Calculate future date
    last_date = df.index[-1]
    future_date = last_date + datetime.timedelta(days=prediction_days)
    
    # Current price and change percentage
    current_price = df['Close'].iloc[-1]
    price_change = ((future_pred / current_price) - 1) * 100
    
    return future_pred, future_date, current_price, price_change