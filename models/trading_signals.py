def generate_trading_signals(df):
    """Generate trading signals based on technical indicators."""
    # Create a copy to avoid modifying the original dataframe
    signals_df = df.copy()
    
    # Initialize signals column
    signals_df['Signal'] = 0
    
    # MACD Signal
    signals_df.loc[signals_df['MACD'] > signals_df['Signal_Line'], 'MACD_Signal'] = 1
    signals_df.loc[signals_df['MACD'] < signals_df['Signal_Line'], 'MACD_Signal'] = -1
    
    # RSI Signal
    signals_df.loc[signals_df['RSI'] < 30, 'RSI_Signal'] = 1
    signals_df.loc[signals_df['RSI'] > 70, 'RSI_Signal'] = -1
    signals_df.loc[(signals_df['RSI'] >= 30) & (signals_df['RSI'] <= 70), 'RSI_Signal'] = 0
    
    # Bollinger Bands Signal
    signals_df.loc[signals_df['Close'] < signals_df['BB_Lower'], 'BB_Signal'] = 1
    signals_df.loc[signals_df['Close'] > signals_df['BB_Upper'], 'BB_Signal'] = -1
    signals_df.loc[(signals_df['Close'] >= signals_df['BB_Lower']) & (signals_df['Close'] <= signals_df['BB_Upper']), 'BB_Signal'] = 0
    
    # Moving Average Signal
    signals_df.loc[signals_df['MA5'] > signals_df['MA20'], 'MA_Signal'] = 1
    signals_df.loc[signals_df['MA5'] < signals_df['MA20'], 'MA_Signal'] = -1
    
    # Combine signals (simple average)
    signals_df['Signal'] = (signals_df['MACD_Signal'].fillna(0) + 
                           signals_df['RSI_Signal'].fillna(0) + 
                           signals_df['BB_Signal'].fillna(0) + 
                           signals_df['MA_Signal'].fillna(0)) / 4
    
    # Classify signals
    signals_df.loc[signals_df['Signal'] > 0.3, 'Signal_Class'] = 'Buy'
    signals_df.loc[signals_df['Signal'] < -0.3, 'Signal_Class'] = 'Sell'
    signals_df.loc[(signals_df['Signal'] >= -0.3) & (signals_df['Signal'] <= 0.3), 'Signal_Class'] = 'Hold'
    
    return signals_df

def get_signal_statistics(signals_df):
    """Get statistics about trading signals."""
    # Count signals
    buy_count = len(signals_df[signals_df['Signal_Class'] == 'Buy'])
    sell_count = len(signals_df[signals_df['Signal_Class'] == 'Sell'])
    hold_count = len(signals_df[signals_df['Signal_Class'] == 'Hold'])
    total_signals = len(signals_df)
    
    # Get last signal
    last_signal = signals_df['Signal_Class'].iloc[-1]
    signal_strength = abs(signals_df['Signal'].iloc[-1])
    
    # Get indicator values
    macd_signal = "Bullish" if signals_df['MACD_Signal'].iloc[-1] > 0 else "Bearish"
    rsi_value = signals_df['RSI'].iloc[-1]
    rsi_signal = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
    bb_signal = "Below Lower Band" if signals_df['BB_Signal'].iloc[-1] > 0 else "Above Upper Band" if signals_df['BB_Signal'].iloc[-1] < 0 else "Within Bands"
    ma_signal = "Bullish" if signals_df['MA_Signal'].iloc[-1] > 0 else "Bearish"
    
    return {
        'buy_count': buy_count,
        'sell_count': sell_count,
        'hold_count': hold_count,
        'total_signals': total_signals,
        'last_signal': last_signal,
        'signal_strength': signal_strength,
        'macd_signal': macd_signal,
        'rsi_value': rsi_value,
        'rsi_signal': rsi_signal,
        'bb_signal': bb_signal,
        'ma_signal': ma_signal
    }