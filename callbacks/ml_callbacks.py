from dash import Input, Output, State
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html
import numpy as np
import datetime

from data.stock_data import get_full_stock_data
from models.price_prediction import prepare_prediction_data, train_price_prediction_model, predict_future_price
from models.trading_signals import generate_trading_signals, get_signal_statistics
from models.risk_analysis import analyze_risk
from sklearn.metrics import mean_squared_error
def register_ml_callbacks(app):
    """Register callbacks for the ML predictions."""
    @app.callback(
        [Output('prediction-summary', 'children'),
         Output('prediction-chart', 'figure'),
         Output('model-accuracy', 'children'),
         Output('model-confidence', 'children'),
         Output('feature-importance', 'children'),
         Output('trading-signal-summary', 'children'),
         Output('trading-signal-chart', 'figure'),
         Output('signal-details', 'children'),
         Output('risk-summary', 'children'),
         Output('risk-chart', 'figure'),
         Output('risk-metrics', 'children'),
         Output('ml-error-message', 'children')],
        [Input('ml-analyze-button', 'n_clicks')],
        [State('ml-stock-input', 'value'),
         State('prediction-days', 'value')]
    )
    def update_ml_predictions(n_clicks, ticker, prediction_days):
        # Initialize empty outputs
        empty_outputs = [
            html.Div(), go.Figure(), html.Div(), html.Div(), html.Div(),
            html.Div(), go.Figure(), html.Div(), html.Div(), go.Figure(), html.Div(), ""
        ]
        
        # Validate input
        if not ticker:
            empty_outputs[-1] = "Please enter a stock ticker."
            return empty_outputs

        try:
            prediction_days = int(prediction_days)
        except:
            prediction_days = 7
        
        try:
            # Get full stock data for ML
            df = get_full_stock_data(ticker)
            
            if df is None or df.empty:
                empty_outputs[-1] = f"No data found for {ticker}"
                return empty_outputs
            
            # Prepare data for prediction
            X_train, X_test, y_train, y_test, scaler_X, scaler_y, features = prepare_prediction_data(df, prediction_days)
            
            # Train model
            model = train_price_prediction_model(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            accuracy = 1 - rmse
            
            # Get feature importance
            feature_importance = model.feature_importances_
            
            # Inverse transform predictions
            y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            # Create prediction chart
            fig_prediction = go.Figure()
            
            # Add actual prices
            fig_prediction.add_trace(go.Scatter(
                x=df.index[-len(y_test):],
                y=y_test_inv,
                mode='lines',
                name='Actual Price',
                line=dict(color='#2980b9', width=2)
            ))
            
            # Add predicted prices
            fig_prediction.add_trace(go.Scatter(
                x=df.index[-len(y_test):],
                y=y_pred_inv,
                mode='lines',
                name='Predicted Price',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
            
            # Add future prediction
            last_date = df.index[-1]
            future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, prediction_days + 1)]
            
            # Predict future price
            future_pred, future_date, current_price, price_change = predict_future_price(
                df, model, scaler_X, scaler_y, features, prediction_days
            )
            
            # Add future prediction point
            fig_prediction.add_trace(go.Scatter(
                x=[future_dates[-1]],
                y=[future_pred],
                mode='markers',
                name=f'Prediction ({prediction_days} days)',
                marker=dict(color='#27ae60', size=12, symbol='star')
            ))
            
            # Update layout
            fig_prediction.update_layout(
                title={
                    'text': f"{ticker} Price Prediction",
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
                )
            )
            
            # Create prediction summary
            price_direction = "up" if price_change > 0 else "down"
            prediction_summary = html.Div([
                html.H3(f"Predicted Price in {prediction_days} Days:", className="mb-3"),
                html.Div([
                    html.Span("Current Price: ", className="fw-bold me-2"),
                    html.Span(f"${current_price:.2f}", className="fw-bold")
                ], className="mb-2"),
                html.Div([
                    html.Span("Predicted Price: ", className="fw-bold me-2"),
                    html.Span(f"${future_pred:.2f}", className=f"fw-bold prediction-{'up' if price_change > 0 else 'down'}")
                ], className="mb-2"),
                html.Div([
                    html.Span("Expected Change: ", className="fw-bold me-2"),
                    html.Span(f"{price_change:.2f}%", className=f"fw-bold prediction-{'up' if price_change > 0 else 'down'}")
                ], className="mb-4"),
                html.Div([
                    dbc.Alert(
                        [
                            html.I(className=f"fas fa-arrow-{'up' if price_change > 0 else 'down'} me-2"),
                            f"Our model predicts the stock price will go {price_direction} by {abs(price_change):.2f}% in the next {prediction_days} days."
                        ],
                        color="success" if price_change > 0 else "danger",
                        className="mb-0"
                    )
                ])
            ])
            
            # Create model accuracy display
            model_accuracy = html.Div([
                dbc.Progress(
                    value=accuracy * 100,
                    color="success" if accuracy > 0.7 else "warning" if accuracy > 0.5 else "danger",
                    striped=True,
                    animated=True,
                    className="mb-2"
                ),
                html.P(f"Model Accuracy: {accuracy * 100:.2f}%", className="text-center")
            ])
            
            # Create model confidence
            confidence_level = "High" if accuracy > 0.7 else "Medium" if accuracy > 0.5 else "Low"
            confidence_color = "success" if accuracy > 0.7 else "warning" if accuracy > 0.5 else "danger"
            
            model_confidence = html.Div([
                html.Div([
                    html.Span("Confidence Level: ", className="fw-bold me-2"),
                    html.Span(confidence_level, className=f"text-{confidence_color} fw-bold")
                ], className="mb-2"),
                html.Div([
                    html.Span("Root Mean Square Error: ", className="fw-bold me-2"),
                    html.Span(f"{rmse:.4f}", className="fw-bold")
                ], className="mb-4"),
                html.Div([
                    html.P("Model Metrics:", className="fw-bold mb-2"),
                    html.Ul([
                        html.Li(f"Training Data: {len(X_train)} days"),
                        html.Li(f"Testing Data: {len(X_test)} days"),
                        html.Li(f"Prediction Horizon: {prediction_days} days")
                    ])
                ], className="model-info-card")
            ])
            
            # Create feature importance visualization
            feature_importance_data = sorted(zip(features, feature_importance), key=lambda x: x[1], reverse=True)
            
            feature_bars = []
            for feature, importance in feature_importance_data[:5]:  # Show top 5 features
                color = "success" if importance > 0.2 else "primary" if importance > 0.1 else "info"
                feature_bars.append(
                    html.Div([
                        html.Div([
                            html.Span(feature, className="float-start"),
                            html.Span(f"{importance:.4f}", className="float-end")
                        ], className="mb-1"),
                        dbc.Progress(
                            value=importance * 100,
                            color=color,
                            className="feature-importance-bar"
                        )
                    ], className="mb-3")
                )
            
            feature_importance_viz = html.Div(feature_bars)
            
            # Generate trading signals
            signals_df = generate_trading_signals(df)
            signal_stats = get_signal_statistics(signals_df)
            
            # Create trading signals chart
            fig_signals = go.Figure()
            
            # Add price line
            fig_signals.add_trace(go.Scatter(
                x=signals_df.index,
                y=signals_df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#2980b9', width=2)
            ))
            
            # Add buy signals
            buy_signals = signals_df[signals_df['Signal_Class'] == 'Buy']
            fig_signals.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    color='#2ecc71',
                    size=10,
                    symbol='triangle-up'
                )
            ))
            
            # Add sell signals
            sell_signals = signals_df[signals_df['Signal_Class'] == 'Sell']
            fig_signals.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(
                    color='#e74c3c',
                    size=10,
                    symbol='triangle-down'
                )
            ))
            
            # Update layout
            fig_signals.update_layout(
                title={
                    'text': f"{ticker} Trading Signals",
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
                )
            )
            
            # Create trading signal summary
            last_signal = signal_stats['last_signal']
            signal_strength = signal_stats['signal_strength']
            
            signal_icon = html.I(className=f"fas fa-arrow-{'up' if last_signal == 'Buy' else 'down' if last_signal == 'Sell' else 'right'} me-2")
            signal_color = "success" if last_signal == 'Buy' else "danger" if last_signal == 'Sell' else "warning"
            
            trading_signal_summary = html.Div([
                html.H3("Current Trading Signal:", className="mb-3"),
                html.Div([
                    dbc.Alert(
                        [
                            signal_icon,
                            f"The current trading signal for {ticker} is {last_signal.upper()}"
                        ],
                        color=signal_color,
                        className="mb-3 text-center fw-bold"
                    )
                ]),
                html.Div([
                    html.Span("Signal Strength: ", className="fw-bold me-2"),
                    dbc.Progress(
                        value=signal_strength * 100,
                        color=signal_color,
                        striped=True,
                        animated=True,
                        className="mb-3"
                    )
                ]),
                html.Div([
                    html.P(f"Based on technical analysis of {ticker}, our model recommends a {last_signal.upper()} position.")
                ])
            ])
            
            # Create signal details
            signal_details_class = "signal-buy" if last_signal == 'Buy' else "signal-sell" if last_signal == 'Sell' else "signal-hold"
            
            signal_details = html.Div([
                html.Div([
                    html.H5("SIGNAL BREAKDOWN", className="mb-3"),
                    html.Div([
                        html.Div([
                            html.Span("Buy Signals: ", className="fw-bold"),
                            html.Span(f"{signal_stats['buy_count']} ({signal_stats['buy_count']/signal_stats['total_signals']*100:.1f}%)", className="text-success")
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Sell Signals: ", className="fw-bold"),
                            html.Span(f"{signal_stats['sell_count']} ({signal_stats['sell_count']/signal_stats['total_signals']*100:.1f}%)", className="text-danger")
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Hold Signals: ", className="fw-bold"),
                            html.Span(f"{signal_stats['hold_count']} ({signal_stats['hold_count']/signal_stats['total_signals']*100:.1f}%)", className="text-warning")
                        ], className="mb-4")
                    ])
                ], className="mb-4"),
                
                html.Div([
                    html.H5("TECHNICAL INDICATORS", className="mb-3"),
                    html.Div([
                        html.Div([
                            html.Span("MACD: ", className="fw-bold"),
                            html.Span(
                                signal_stats['macd_signal'], 
                                className=f"text-{'success' if signal_stats['macd_signal'] == 'Bullish' else 'danger'}"
                            )
                        ], className="mb-2"),
                        html.Div([
                            html.Span("RSI: ", className="fw-bold"),
                            html.Span(
                                f"{signal_stats['rsi_value']:.2f} - {signal_stats['rsi_signal']}",
                                className=f"text-{'success' if signal_stats['rsi_signal'] == 'Oversold' else 'danger' if signal_stats['rsi_signal'] == 'Overbought' else 'warning'}"
                            )
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Bollinger Bands: ", className="fw-bold"),
                            html.Span(
                                signal_stats['bb_signal'],
                                className=f"text-{'success' if signal_stats['bb_signal'] == 'Below Lower Band' else 'danger' if signal_stats['bb_signal'] == 'Above Upper Band' else 'warning'}"
                            )
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Moving Averages: ", className="fw-bold"),
                            html.Span(
                                signal_stats['ma_signal'],
                                className=f"text-{'success' if signal_stats['ma_signal'] == 'Bullish' else 'danger'}"
                            )
                        ], className="mb-2")
                    ])
                ], className="mb-4"),
                
                html.Div([
                    html.H5("RECOMMENDATION", className="mb-3"),
                    html.Div([
                        html.P([
                            f"Based on the analysis of multiple technical indicators, our model suggests a ",
                            html.Span(f"{last_signal.upper()}", className=f"fw-bold text-{signal_color}"),
                            f" position for {ticker}."
                        ], className=signal_details_class)
                    ])
                ])
            ])
            
            # Analyze risk
            risk_metrics = analyze_risk(df, ticker)
            
            # Create risk chart
            # Calculate daily returns for volatility visualization
            returns = df['Close'].pct_change().dropna()
            
            fig_risk = go.Figure()
            
            # Add histogram of returns
            fig_risk.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                marker_color='#3498db',
                opacity=0.7,
                name='Daily Returns'
            ))
            
            # Add normal distribution curve
            x = np.linspace(min(returns), max(returns), 100)
            y = np.exp(-(x - returns.mean())**2 / (2 * returns.std()**2)) / (returns.std() * np.sqrt(2 * np.pi))
            y = y * (len(returns) * (max(returns) - min(returns)) / 50)  # Scale to match histogram
            
            fig_risk.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(color='#e74c3c', width=2),
                name='Normal Distribution'
            ))
            
            # Add VaR line
            fig_risk.add_vline(
                x=risk_metrics['var_95'],
                line_dash="dash",
                line_color="#e74c3c",
                annotation_text="95% VaR",
                annotation_position="top right"
            )
            
            # Update layout
            fig_risk.update_layout(
                title={
                    'text': f"{ticker} Risk Profile",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24, color='#2c3e50')
                },
                xaxis_title={
                    'text': "Daily Returns",
                    'font': dict(size=16, color='#2c3e50')
                },
                yaxis_title={
                    'text': "Frequency",
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
            
            # Create risk summary
            risk_summary = html.Div([
                html.H3("Risk Assessment:", className="mb-3"),
                html.Div([
                    dbc.Alert(
                        [
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            f"{ticker} has a {risk_metrics['risk_level'].upper()} risk profile"
                        ],
                        color="success" if risk_metrics['risk_level'] == "Low" else "warning" if risk_metrics['risk_level'] == "Medium" else "danger",
                        className="mb-3 text-center fw-bold"
                    )
                ]),
                html.Div([
                    html.Span("Volatility: ", className="fw-bold me-2"),
                    html.Span(f"{risk_metrics['volatility']*100:.2f}%", className=f"fw-bold {risk_metrics['risk_class']}")
                ], className="mb-2"),
                html.Div([
                    html.Span("Beta: ", className="fw-bold me-2"),
                    html.Span(f"{risk_metrics['beta']:.2f}", className="fw-bold")
                ], className="mb-2"),
                html.Div([
                    html.Span("Sharpe Ratio: ", className="fw-bold me-2"),
                    html.Span(f"{risk_metrics['sharpe_ratio']:.2f}", className="fw-bold")
                ], className="mb-2")
            ])
            
            # Create risk metrics details
            risk_metrics_details = html.Div([
                html.Div([
                    html.H5("VOLATILITY", className="mb-2"),
                    html.Div([
                        html.Span("Annualized Volatility: ", className="fw-bold"),
                        html.Span(f"{risk_metrics['volatility']*100:.2f}%", className=f"{risk_metrics['risk_class']}")
                    ], className="mb-2"),
                    html.P(
                        "Volatility measures the dispersion of returns. Higher volatility indicates greater risk.",
                        className="text-muted small"
                    ),
                    dbc.Progress(
                        value=min(risk_metrics['volatility']*100*2, 100),  # Scale for better visualization
                        color="success" if risk_metrics['risk_level'] == "Low" else "warning" if risk_metrics['risk_level'] == "Medium" else "danger",
                        className="mb-4"
                    )
                ]),
                
                html.Div([
                    html.H5("MARKET RISK", className="mb-2"),
                    html.Div([
                        html.Span("Beta: ", className="fw-bold"),
                        html.Span(f"{risk_metrics['beta']:.2f}", className=f"{'risk-low' if abs(risk_metrics['beta']) < 0.8 else 'risk-medium' if abs(risk_metrics['beta']) < 1.2 else 'risk-high'}")
                    ], className="mb-2"),
                    html.P(
                        "Beta measures the stock's volatility relative to the market. Beta > 1 indicates higher volatility than the market.",
                        className="text-muted small"
                    ),
                    dbc.Progress(
                        value=min(abs(risk_metrics['beta'])*50, 100),  # Scale for better visualization
                        color="success" if abs(risk_metrics['beta']) < 0.8 else "warning" if abs(risk_metrics['beta']) < 1.2 else "danger",
                        className="mb-4"
                    )
                ]),
                
                html.Div([
                    html.H5("DOWNSIDE RISK", className="mb-2"),
                    html.Div([
                        html.Span("Value at Risk (95%): ", className="fw-bold"),
                        html.Span(f"{risk_metrics['var_95']*100:.2f}%", className="risk-high")
                    ], className="mb-2"),
                    html.P(
                        "VaR represents the maximum expected loss with 95% confidence over a single day.",
                        className="text-muted small"
                    ),
                    html.Div([
                        html.Span("Maximum Drawdown: ", className="fw-bold"),
                        html.Span(f"{risk_metrics['max_drawdown']*100:.2f}%", className="risk-high")
                    ], className="mb-2"),
                    html.P(
                        "Maximum drawdown measures the largest peak-to-trough decline.",
                        className="text-muted small"
                    ),
                    dbc.Progress(
                        value=min(abs(risk_metrics['max_drawdown'])*100, 100),
                        color="danger",
                        className="mb-4"
                    )
                ]),
                
                html.Div([
                    html.H5("RISK-ADJUSTED RETURN", className="mb-2"),
                    html.Div([
                        html.Span("Sharpe Ratio: ", className="fw-bold"),
                        html.Span(f"{risk_metrics['sharpe_ratio']:.2f}", className=f"{'risk-low' if risk_metrics['sharpe_ratio'] < 0.5 else 'risk-medium' if risk_metrics['sharpe_ratio'] < 1 else 'risk-high'}")
                    ], className="mb-2"),
                    html.P(
                        "Sharpe ratio measures return per unit of risk. Higher is better.",
                        className="text-muted small"
                    ),
                    dbc.Progress(
                        value=min(risk_metrics['sharpe_ratio']*50, 100),  # Scale for better visualization
                        color="success",
                        className="mb-4"
                    )
                ]),
                
                html.Div([
                    html.H5("INVESTMENT RECOMMENDATION", className="mb-3"),
                    html.Div([
                        html.P([
                            f"Based on the risk analysis, {ticker} is suitable for ",
                            html.Span(
                                "conservative investors" if risk_metrics['risk_level'] == "Low" else 
                                "balanced portfolios" if risk_metrics['risk_level'] == "Medium" else 
                                "aggressive investors",
                                className=f"fw-bold {risk_metrics['risk_class']}"
                            ),
                            "."
                        ], className=f"signal-{'buy' if risk_metrics['risk_level'] == 'Low' else 'hold' if risk_metrics['risk_level'] == 'Medium' else 'sell'}")
                    ])
                ])
            ])
            
            return (
                prediction_summary, fig_prediction, model_accuracy, model_confidence, feature_importance_viz,
                trading_signal_summary, fig_signals, signal_details,
                risk_summary, fig_risk, risk_metrics_details, ""
            )
            
        except Exception as e:
            empty_outputs[-1] = f"Error in ML analysis: {str(e)}"
            return empty_outputs