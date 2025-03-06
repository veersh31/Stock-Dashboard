from dash import dcc, html
import dash_bootstrap_components as dbc

def portfolio_optimizer_layout():
    """Create the portfolio optimizer layout."""
    return dbc.Container([
        html.H1("PORTFOLIO OPTIMIZER", className="text-center dashboard-title mb-5"),
        
        dbc.Row([
            dbc.Col([
                dbc.InputGroup([
                    dbc.Input(
                        id='portfolio-input', 
                        type='text', 
                        placeholder='Enter Stock Tickers (comma-separated)', 
                        className="stock-input"
                    ),
                    dbc.Button(
                        'OPTIMIZE PORTFOLIO', 
                        id='optimize-button', 
                        n_clicks=0, 
                        className="submit-btn",
                        color="primary"
                    ),
                ], size="lg")
            ], width={"size": 8, "offset": 2}, className="mb-4")
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id='efficient-frontier', style={"height": "60vh"})
                ], className="chart-container")
            ], width=12, className="mb-4"),
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(
                    id='optimal-weights', 
                    className="bg-white p-4 rounded-3 shadow-sm text-center fw-bold"
                )
            ], width={"size": 8, "offset": 2}, className="mb-4")
        ])
    ])