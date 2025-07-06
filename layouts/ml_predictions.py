from dash import dcc, html
import dash_bootstrap_components as dbc

def ml_predictions_layout():
    """Create the ML predictions layout."""
    return dbc.Container([
        html.H1("ML INVESTMENT ADVISOR", className="text-center dashboard-title mb-5"),
        
        dbc.Row([
            dbc.Col([
                dbc.InputGroup([
                    dbc.Input(
                        id='ml-stock-input', 
                        type='text', 
                        value='AAPL', 
                        placeholder='Enter Stock Ticker', 
                        className="stock-input"
                    ),
                    dbc.Select(
                        id='prediction-days',
                        options=[
                            {'label': '7 Days', 'value': '7'},
                            {'label': '14 Days', 'value': '14'},
                            {'label': '30 Days', 'value': '30'}
                        ],
                        value='7',
                        className="stock-input"
                    ),
                    dbc.Button(
                        'ANALYZE', 
                        id='ml-analyze-button', 
                        n_clicks=0, 
                        className="submit-btn",
                        color="primary"
                    ),
                ], size="lg")
            ], width={"size": 8, "offset": 2}, className="mb-4")
        ]),
        
        html.Div(id='ml-error-message', className="text-danger text-center fw-bold mb-3"),
        

        dbc.Tabs([
            
            dbc.Tab([
                dbc.Row([
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("PRICE PREDICTION", className="text-center text-white mb-0")
                            ], className="prediction-header"),
                            dbc.CardBody([
                                html.Div(id="prediction-summary", className="text-center mb-4"),
                                html.Div([
                                    dcc.Graph(id='prediction-chart', style={"height": "50vh"})
                                ], className="chart-container")
                            ])
                        ], className="prediction-card shadow mb-4")
                    ], width=8),
                    
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("MODEL INSIGHTS", className="text-center text-white mb-0")
                            ], className="prediction-header"),
                            dbc.CardBody([
                                html.Div([
                                    html.H5("MODEL ACCURACY", className="mb-2"),
                                    html.Div(id="model-accuracy"),
                                    html.Div(id="model-confidence", className="mb-4"),
                                    html.H5("FEATURE IMPORTANCE", className="mb-3"),
                                    html.Div(id="feature-importance")
                                ])
                            ])
                        ], className="prediction-card shadow")
                    ], width=4)
                ]),
            ], label="PRICE PREDICTION", tab_id="tab-prediction"),
            
            
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("TRADING SIGNALS", className="text-center text-white mb-0")
                            ], className="prediction-header"),
                            dbc.CardBody([
                                html.Div(id="trading-signal-summary", className="text-center mb-4"),
                                html.Div([
                                    dcc.Graph(id='trading-signal-chart', style={"height": "50vh"})
                                ], className="chart-container")
                            ])
                        ], className="prediction-card shadow mb-4")
                    ], width=8),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("SIGNAL ANALYSIS", className="text-center text-white mb-0")
                            ], className="prediction-header"),
                            dbc.CardBody([
                                html.Div(id="signal-details")
                            ])
                        ], className="prediction-card shadow")
                    ], width=4)
                ])
            ], label="TRADING SIGNALS", tab_id="tab-signals"),
            
            # Risk Analysis Tab
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("RISK ANALYSIS", className="text-center text-white mb-0")
                            ], className="prediction-header"),
                            dbc.CardBody([
                                html.Div(id="risk-summary", className="text-center mb-4"),
                                html.Div([
                                    dcc.Graph(id='risk-chart', style={"height": "50vh"})
                                ], className="chart-container")
                            ])
                        ], className="prediction-card shadow mb-4")
                    ], width=8),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("RISK METRICS", className="text-center text-white mb-0")
                            ], className="prediction-header"),
                            dbc.CardBody([
                                html.Div(id="risk-metrics")
                            ])
                        ], className="prediction-card shadow")
                    ], width=4)
                ])
            ], label="RISK ANALYSIS", tab_id="tab-risk")
        ], id="ml-tabs", active_tab="tab-prediction", className="mb-4")
    ])
