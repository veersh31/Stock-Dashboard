from dash import Input, Output, State
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html
import numpy as np

from models import optimize_portfolio
from data.stock_data import get_stock_data

def register_portfolio_callbacks(app):
    """Register callbacks for the portfolio optimizer."""
    @app.callback(
        [Output('efficient-frontier', 'figure'), Output('optimal-weights', 'children')],
        [Input('optimize-button', 'n_clicks')],
        [State('portfolio-input', 'value')]
    )
    def update_portfolio(n_clicks, input_tickers):
        if not input_tickers:
            return go.Figure(layout=go.Layout(
                title="Enter stock tickers to visualize the efficient frontier",
                template="plotly_white",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14)
            )), "Please enter stock tickers separated by commas (e.g., AAPL, MSFT, GOOG)"

        tickers = [t.strip().upper() for t in input_tickers.split(',')]
        weights = optimize_portfolio(tickers)

        if weights is None:
            return go.Figure(layout=go.Layout(
                title="Invalid stock tickers or insufficient data",
                template="plotly_white",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14)
            )), "Invalid stock tickers or insufficient data. Please check your input."

        
        df = get_stock_data(tickers)
        returns = df.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        num_portfolios = 10000
        results = np.zeros((3, num_portfolios))

        for i in range(num_portfolios):
            rand_weights = np.random.random(len(tickers))
            rand_weights /= np.sum(rand_weights)
            ret, risk = np.sum(rand_weights * mean_returns), np.sqrt(np.dot(rand_weights.T, np.dot(cov_matrix, rand_weights)))
            results[0, i] = risk
            results[1, i] = ret
            results[2, i] = ret / risk  

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results[0], 
            y=results[1], 
            mode='markers',
            marker=dict(
                color=results[2], 
                colorscale='Viridis', 
                size=5,
                colorbar=dict(title="Sharpe Ratio")
            ), 
            name="Efficient Frontier"
        ))

        optimal_ret, optimal_risk = np.sum(weights * mean_returns), np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        fig.add_trace(go.Scatter(
            x=[optimal_risk], 
            y=[optimal_ret], 
            mode='markers',
            marker=dict(
                color='red', 
                size=15,
                symbol='star'
            ), 
            name="Optimal Portfolio"
        ))

        fig.update_layout(
            title={
                'text': "Efficient Frontier",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24, color='#2c3e50')
            },
            xaxis_title={
                'text': "Risk (Standard Deviation)",
                'font': dict(size=16, color='#2c3e50')
            },
            yaxis_title={
                'text': "Expected Return",
                'font': dict(size=16, color='#2c3e50')
            },
            template="plotly_white",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(size=14),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            )
        )

        
        weight_cards = []
        for i, ticker in enumerate(tickers):
            weight_percentage = weights[i] * 100
            color = "success" if weight_percentage > 20 else "primary" if weight_percentage > 10 else "info"
            
            weight_cards.append(
                dbc.Card([
                    dbc.CardBody([
                        html.H5(ticker, className="card-title text-center"),
                        html.Div([
                            dbc.Progress(
                                value=weight_percentage,
                                color=color,
                                striped=True,
                                animated=True,
                                className="mb-2"
                            ),
                            html.P(f"{weight_percentage:.2f}%", className="text-center")
                        ])
                    ])
                ], className="mb-3 shadow-sm")
            )
        
        optimal_sharpe = optimal_ret / optimal_risk if optimal_risk > 0 else 0
        
        return fig, html.Div([
            html.H4("OPTIMAL PORTFOLIO ALLOCATION", className="mb-4"),
            html.Div([
                html.Span("Expected Annual Return: ", className="fw-bold"),
                html.Span(f"{optimal_ret * 252 * 100:.2f}%", className="text-success fw-bold")
            ], className="mb-2"),
            html.Div([
                html.Span("Expected Volatility: ", className="fw-bold"),
                html.Span(f"{optimal_risk * np.sqrt(252) * 100:.2f}%", className="text-danger fw-bold")
            ], className="mb-2"),
            html.Div([
                html.Span("Sharpe Ratio: ", className="fw-bold"),
                html.Span(f"{optimal_sharpe:.4f}", className="text-primary fw-bold")
            ], className="mb-4"),
            dbc.Row([dbc.Col(card, width=4) for card in weight_cards], className="justify-content-center")
        ])
