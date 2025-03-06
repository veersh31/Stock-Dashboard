from dash import Input, Output, State
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html

from data.stock_data import get_dashboard_stock_data
from data.news_data import get_stock_news

def register_dashboard_callbacks(app):
    """Register callbacks for the stock dashboard."""
    @app.callback(
        [Output('stock-price', 'children'),
         Output('market-cap', 'children'),
         Output('pe-ratio', 'children'),
         Output('div-yield', 'children'),
         Output('beta', 'children'),
         Output('high52', 'children'),
         Output('low52', 'children'),
         Output('eps', 'children'),
         Output('stock-chart', 'figure'),
         Output('volume-chart', 'figure'),
         Output('news-section', 'children'),
         Output('error-message', 'children')],
        [Input('submit-button', 'n_clicks')],
        [State('stock-input', 'value')]
    )
    def update_stock_data(n_clicks, ticker):
        # Validate input
        if not ticker:
            return (["N/A"] * 8 + 
                    [
                        go.Figure(layout=go.Layout(
                            title="Please enter a stock ticker",
                            template="plotly_white",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )), 
                        go.Figure(layout=go.Layout(
                            title="Please enter a stock ticker",
                            template="plotly_white",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )), 
                        [], 
                        "Please enter a stock ticker."
                    ])

        # Get stock data
        stock, history, error = get_dashboard_stock_data(ticker)

        # Handle errors
        if error:
            return (["Error"] * 8 + 
                    [
                        go.Figure(layout=go.Layout(
                            title=f"Error: {error}",
                            template="plotly_white",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )), 
                        go.Figure(layout=go.Layout(
                            title=f"Error: {error}",
                            template="plotly_white",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )), 
                        [], 
                        f"Error: {error}"
                    ])

        # Safely extract information with error handling
        try:
            info = stock.info
            
            # Extract metrics with safe fallback
            latest_price = f"${history['Close'].iloc[-1]:,.2f}"
            market_cap = f"${info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "N/A"
            pe_ratio = f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else "N/A"
            div_yield = f"{info.get('dividendYield', 'N/A'):.2%}" if info.get('dividendYield') else "N/A"
            beta = f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else "N/A"
            high52 = f"${info.get('fiftyTwoWeekHigh', 'N/A'):,}"
            low52 = f"${info.get('fiftyTwoWeekLow', 'N/A'):,}"
            eps = f"${info.get('trailingEps', 'N/A'):.2f}" if info.get('trailingEps') else "N/A"

            # Create stock price chart with improved styling
            fig_price = go.Figure()
            
            # Add price line
            fig_price.add_trace(go.Scatter(
                x=history.index, 
                y=history['Close'], 
                mode='lines', 
                name=f'{ticker} Price',
                line=dict(
                    color='#2980b9', 
                    width=3
                ),
                fill='tozeroy',
                fillcolor='rgba(41, 128, 185, 0.1)'
            ))
            
            # Add moving averages
            ma20 = history['Close'].rolling(window=20).mean()
            ma50 = history['Close'].rolling(window=50).mean()
            
            fig_price.add_trace(go.Scatter(
                x=history.index, 
                y=ma20, 
                mode='lines', 
                name='20-Day MA',
                line=dict(
                    color='#e74c3c', 
                    width=2,
                    dash='dash'
                )
            ))
            
            fig_price.add_trace(go.Scatter(
                x=history.index, 
                y=ma50, 
                mode='lines', 
                name='50-Day MA',
                line=dict(
                    color='#27ae60', 
                    width=2,
                    dash='dash'
                )
            ))
            
            # Update layout
            fig_price.update_layout(
                title={
                    'text': f"{ticker} Stock Price",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24, color='#2c3e50')
                },
                xaxis_title={
                    'text': "Date",
                    'font': dict(size=16, color='#2c3e50')
                },
                yaxis_title={
                    'text': "Price (USD)",
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
                ),
                margin=dict(l=40, r=40, t=80, b=40)
            )
            
            # Create volume trend chart with improved styling
            fig_volume = go.Figure()
            
            # Add volume bars with color based on price change
            colors = ['#2ecc71' if history['Close'].iloc[i] > history['Close'].iloc[i-1] 
                     else '#e74c3c' for i in range(1, len(history))]
            colors.insert(0, '#2ecc71')  # Add color for the first day
            
            fig_volume.add_trace(go.Bar(
                x=history.index, 
                y=history['Volume'], 
                marker=dict(
                    color=colors,
                    line=dict(width=0)
                ),
                name='Volume'
            ))
            
            # Add volume moving average
            vol_ma20 = history['Volume'].rolling(window=20).mean()
            fig_volume.add_trace(go.Scatter(
                x=history.index, 
                y=vol_ma20, 
                mode='lines', 
                name='20-Day Volume MA',
                line=dict(
                    color='#f39c12', 
                    width=2
                )
            ))
            
            # Update layout
            fig_volume.update_layout(
                title={
                    'text': f"{ticker} Trading Volume",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24, color='#2c3e50')
                },
                xaxis_title={
                    'text': "Date",
                    'font': dict(size=16, color='#2c3e50')
                },
                yaxis_title={
                    'text': "Volume",
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
                ),
                margin=dict(l=40, r=40, t=80, b=40)
            )

            # Fetch news with improved styling
            news_articles = get_stock_news(ticker)
            
            # If no news articles, provide a message
            if not news_articles:
                news_articles = [
                    dbc.Alert(
                        [
                            html.I(className="fas fa-info-circle me-2"),
                            "No recent news articles found for this stock."
                        ],
                        color="info",
                        className="text-center"
                    )
                ]

            return (
                latest_price, market_cap, pe_ratio, div_yield, 
                beta, high52, low52, eps, 
                fig_price, fig_volume, news_articles, ""
            )

        except Exception as e:
            return (
                ["Error"] * 8 + 
                [
                    go.Figure(layout=go.Layout(
                        title=f"Unexpected error occurred",
                        template="plotly_white",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )), 
                    go.Figure(layout=go.Layout(
                        title=f"Unexpected error occurred",
                        template="plotly_white",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )), 
                    [], 
                    f"Unexpected error: {str(e)}"
                ]
            )