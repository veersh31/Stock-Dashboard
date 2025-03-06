from dash import dcc, html
import dash_bootstrap_components as dbc

def stock_dashboard_layout():
    """Create the stock dashboard layout."""
    return dbc.Container([
        html.H1("STOCK MARKET DASHBOARD", className="text-center dashboard-title mb-5"),
        
        dbc.Row([
            dbc.Col([
                dbc.InputGroup([
                    dbc.Input(
                        id='stock-input', 
                        type='text', 
                        value='AAPL', 
                        placeholder='Enter Stock Ticker', 
                        className="stock-input"
                    ),
                    dbc.Button(
                        'SUBMIT', 
                        id='submit-button', 
                        n_clicks=0, 
                        className="submit-btn",
                        color="primary"
                    ),
                ], size="lg")
            ], width={"size": 6, "offset": 3}, className="mb-4")
        ]),
        
        html.Div(id='error-message', className="text-danger text-center fw-bold mb-3"),
        
        # Stock Metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("LATEST PRICE", className="metric-title text-center"),
                        html.H4(id='stock-price', className="metric-value text-center")
                    ])
                ], className="metric-card bg-white h-100")
            ], width=3, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("MARKET CAP", className="metric-title text-center"),
                        html.H4(id='market-cap', className="metric-value text-center")
                    ])
                ], className="metric-card bg-white h-100")
            ], width=3, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("P/E RATIO", className="metric-title text-center"),
                        html.H4(id='pe-ratio', className="metric-value text-center")
                    ])
                ], className="metric-card bg-white h-100")
            ], width=3, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("DIVIDEND YIELD", className="metric-title text-center"),
                        html.H4(id='div-yield', className="metric-value text-center")
                    ])
                ], className="metric-card bg-white h-100")
            ], width=3, className="mb-4"),
        ]),
        
        # Advanced Metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("BETA", className="metric-title text-center"),
                        html.H4(id='beta', className="metric-value text-center")
                    ])
                ], className="metric-card bg-white h-100")
            ], width=3, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("52-WEEK HIGH", className="metric-title text-center"),
                        html.H4(id='high52', className="metric-value text-center")
                    ])
                ], className="metric-card bg-white h-100")
            ], width=3, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("52-WEEK LOW", className="metric-title text-center"),
                        html.H4(id='low52', className="metric-value text-center")
                    ])
                ], className="metric-card bg-white h-100")
            ], width=3, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("EPS", className="metric-title text-center"),
                        html.H4(id='eps', className="metric-value text-center")
                    ])
                ], className="metric-card bg-white h-100")
            ], width=3, className="mb-4"),
        ]),
        
        # Charts
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id='stock-chart', style={"height": "60vh"})
                ], className="chart-container")
            ], width=12, className="mb-4"),
            
            dbc.Col([
                html.Div([
                    dcc.Graph(id='volume-chart', style={"height": "40vh"})
                ], className="chart-container")
            ], width=12, className="mb-4"),
        ]),
        
        # News Section
        html.H2("LATEST NEWS & SENTIMENT", className="section-title"),
        dbc.Row([
            dbc.Col(html.Div(id='news-section'), width=12)
        ])
    ])