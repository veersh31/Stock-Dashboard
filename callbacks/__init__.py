# callbacks/__init__.py

from callbacks.dashboard_callbacks import register_dashboard_callbacks
from callbacks.portfolio_callbacks import register_portfolio_callbacks
from callbacks.ml_callbacks import register_ml_callbacks

def register_callbacks(app):
    """Register all callbacks with the app."""
    register_dashboard_callbacks(app)
    register_portfolio_callbacks(app)
    register_ml_callbacks(app)