# crypto_trading_app.py (fixed)
import streamlit as st
import pandas as pd
import yfinance as yf
from binance.client import Client
import plotly.graph_objs as go
import numpy as np
import ta  # Technical analysis library
from sklearn.ensemble import RandomForestClassifier
import shap
import sqlite3

# Configuration
st.set_page_config(page_title="Crypto Trading Suite", layout="wide")

# Session state initialization
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

# Data Acquisition (fixed column names)
def fetch_data(source, symbol, timeframe, interval):
    try:
        if source == "Yahoo Finance":
            data = yf.download(symbol, period=timeframe, interval=interval)
            # Standardize column names to lowercase
            data.columns = [col.lower() for col in data.columns]
        elif source == "Binance":
            client = Client()
            klines = client.get_historical_klines(symbol, interval, timeframe)
            data = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            # Convert to numeric and keep essential columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric)
            data = data[['timestamp'] + numeric_cols]
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        return data.dropna()
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return pd.DataFrame()

# Technical Analysis (using standardized column names)
def add_technical_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    bb = ta.volatility.BollingerBands(df['close'])
    df['bollinger_high'] = bb.bollinger_hband()
    df['bollinger_low'] = bb.bollinger_lband()
    return df

# Trading Bot Core (unchanged)
class TradingBot:
    def __init__(self):
        self.connection = sqlite3.connect('trades.db')
        self._init_db()
    
    def _init_db(self):
        cursor = self.connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS trades
                     (timestamp DATETIME, symbol TEXT, side TEXT, quantity REAL)''')
        self.connection.commit()
    
    def execute_mock_trade(self, signal, symbol='BTC-USD', quantity=0.001):
        cursor = self.connection.cursor()
        cursor.execute('''INSERT INTO trades VALUES (?, ?, ?, ?)''', 
                      (pd.Timestamp.now(), symbol, signal, quantity))
        self.connection.commit()

# Backtesting Engine (using standardized column names)
def vectorized_backtest(data):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = np.where(data['rsi'] < 30, 1, np.where(data['rsi'] > 70, -1, 0))
    signals['returns'] = data['close'].pct_change()
    signals['strategy'] = signals['signal'].shift(1) * signals['returns']
    return signals

# UI Components (updated column references)
def sidebar_controls():
    with st.sidebar:
        st.header("Trading Controls")
        source = st.selectbox("Data Source", ["Yahoo Finance", "Binance"])
        symbol = st.text_input("Symbol", "BTC-USD")
        timeframe = st.selectbox("Timeframe", ["1d", "1w", "1mo", "1y"])
        interval = st.selectbox("Interval", ["1h", "4h", "1d"])
        return source, symbol, timeframe, interval

def main_dashboard(data, backtest_results):
    tab1, tab2, tab3 = st.tabs(["Market Data", "Technical Analysis", "Trading"])
    
    with tab1:
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close']
        )])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Technical Indicators")
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(data[['rsi', 'macd']])
        with col2:
            st.line_chart(data[['bollinger_high', 'bollinger_low', 'close']])
        
        st.subheader("SHAP Feature Importance")
        if 'shap_values' in backtest_results:
            shap.summary_plot(backtest_results['shap_values'], data, plot_type="bar")
            st.pyplot()
    
    with tab3:
        st.subheader("Live Trading")
        bot = TradingBot()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Start Trading Bot") and not st.session_state.bot_running:
                st.session_state.bot_running = True
        with col2:
            if st.button("Stop Trading Bot") and st.session_state.bot_running:
                st.session_state.bot_running = False
                
        if st.session_state.bot_running:
            st.success("Trading bot active")
            latest_signal = "BUY" if data['rsi'].iloc[-1] < 30 else "SELL" if data['rsi'].iloc[-1] > 70 else "HOLD"
            bot.execute_mock_trade(latest_signal)
            st.metric("Current Signal", latest_signal)
        else:
            st.warning("Trading bot inactive")
        
        st.subheader("Trade History")
        trades = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10", 
                            bot.connection)
        st.dataframe(trades)

# Main App (unchanged)
def main():
    source, symbol, timeframe, interval = sidebar_controls()
    
    data = fetch_data(source, symbol, timeframe, interval)
    if data.empty:
        st.warning("No data found - check your inputs")
        return
    
    data = add_technical_indicators(data)
    backtest_results = vectorized_backtest(data)
    
    # Model training
    features = ['rsi', 'macd', 'bollinger_high', 'bollinger_low']
    X = data[features].dropna()
    y = np.where(data['close'].shift(-1) > data['close'], 1, -1)[:len(X)]
    
    model = RandomForestClassifier()
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    backtest_results['shap_values'] = explainer.shap_values(X)
    
    main_dashboard(data, backtest_results)

if __name__ == "__main__":
    main()
