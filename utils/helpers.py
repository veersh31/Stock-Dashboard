import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def format_currency(value):
    """Format a value as currency."""
    if isinstance(value, (int, float)):
        return f"${value:,.2f}"
    return "N/A"

def format_percentage(value):
    """Format a value as percentage."""
    if isinstance(value, (int, float)):
        return f"{value:.2f}%"
    return "N/A"