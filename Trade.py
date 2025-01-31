# crypto_trading_app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from binance.client import Client
import plotly.graph_objs as go
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
import shap
import sqlite3

# Configuration
st.set_page_config(page_title="Crypto Trading Suite", layout="wide")

# Session state initialization
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

# Data Acquisition with enhanced validation
def fetch_data(source, symbol, timeframe, interval):
    try:
        if source == "Yahoo Finance":
            data = yf.download(symbol, period=timeframe, interval=interval)
            if data.empty:
                st.error("No data found from Yahoo Finance")
                return pd.DataFrame()
            data.columns = [col.lower() for col in data.columns]
        elif source == "Binance":
            client = Client()
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=timeframe
            )
            if not klines:
                st.error("No data found from Binance")
                return pd.DataFrame()
            data = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric)
            data = data[['timestamp'] + numeric_cols]
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        return data.dropna()
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return pd.DataFrame()

# Technical Analysis with NaN handling
def add_technical_indicators(df):
    if df.empty:
        return df
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12).macd_diff()
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['bollinger_high'] = bb.bollinger_hband()
        df['bollinger_low'] = bb.bollinger_lband()
        return df.ffill().dropna()
    except Exception as e:
        st.error(f"Technical analysis error: {str(e)}")
        return pd.DataFrame()

# Trading Bot with error handling
class TradingBot:
    def __init__(self):
        self.connection = sqlite3.connect('trades.db', check_same_thread=False)
        self._init_db()
    
    def _init_db(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS trades
                          (timestamp DATETIME, symbol TEXT, side TEXT, quantity REAL)''')
            self.connection.commit()
        except sqlite3.Error as e:
            st.error(f"Database error: {str(e)}")
    
    def execute_mock_trade(self, signal, symbol='BTC-USD', quantity=0.001):
        try:
            cursor = self.connection.cursor()
            cursor.execute('''INSERT INTO trades VALUES (?, ?, ?, ?)''', 
                         (pd.Timestamp.now(), symbol, signal, quantity))
            self.connection.commit()
        except sqlite3.Error as e:
            st.error(f"Trade execution error: {str(e)}")

# Backtesting Engine with validation
def vectorized_backtest(data):
    if len(data) < 20:
        return pd.DataFrame()
    try:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = np.where(data['rsi'] < 30, 1, np.where(data['rsi'] > 70, -1, 0))
        signals['returns'] = data['close'].pct_change()
        signals['strategy'] = signals['signal'].shift(1) * signals['returns']
        return signals.dropna()
    except Exception as e:
        st.error(f"Backtesting error: {str(e)}")
        return pd.DataFrame()

# UI Components
def sidebar_controls():
    with st.sidebar:
        st.header("Trading Controls")
        source = st.selectbox("Data Source", ["Yahoo Finance", "Binance"])
        symbol = st.text_input("Symbol", "BTC-USD" if source == "Yahoo Finance" else "BTCUSDT")
        timeframe = st.selectbox("Timeframe", ["1d", "1w", "1mo", "1y"])
        interval = st.selectbox("Interval", ["1h", "4h", "1d"])
        return source, symbol, timeframe, interval

def main_dashboard(data, backtest_results):
    tab1, tab2, tab3 = st.tabs(["Market Data", "Technical Analysis", "Trading"])
    
    with tab1:
        if not data.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if not data.empty:
            st.subheader("Technical Indicators")
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(data[['rsi', 'macd']])
            with col2:
                st.line_chart(data[['bollinger_high', 'bollinger_low', 'close']])
            
            if 'shap_values' in backtest_results and not backtest_results.empty:
                st.subheader("SHAP Feature Importance")
                shap.summary_plot(backtest_results['shap_values'], data[['rsi', 'macd', 'bollinger_high', 'bollinger_low']])
                st.pyplot()
    
    with tab3:
        st.subheader("Live Trading")
        bot = TradingBot()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Trading Bot", disabled=st.session_state.bot_running):
                st.session_state.bot_running = True
        with col2:
            if st.button("Stop Trading Bot", disabled=not st.session_state.bot_running):
                st.session_state.bot_running = False
        
        if st.session_state.bot_running and not data.empty:
            st.success("Trading bot active")
            latest_signal = "BUY" if data['rsi'].iloc[-1] < 30 else "SELL" if data['rsi'].iloc[-1] > 70 else "HOLD"
            bot.execute_mock_trade(latest_signal)
            st.metric("Current Signal", latest_signal)
        else:
            st.warning("Trading bot inactive")
        
        try:
            trades = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10", bot.connection)
            st.subheader("Trade History")
            st.dataframe(trades)
        except sqlite3.Error as e:
            st.error(f"Trade history error: {str(e)}")

# Main App with robust pipeline
def main():
    source, symbol, timeframe, interval = sidebar_controls()
    
    data = fetch_data(source, symbol, timeframe, interval)
    if data.empty:
        return
    
    data = add_technical_indicators(data)
    if data.empty:
        st.error("No valid data after technical analysis")
        return
    
    if len(data) < 100:
        st.warning(f"Insufficient data points ({len(data)}) for reliable analysis")
        return
    
    try:
        features = ['rsi', 'macd', 'bollinger_high', 'bollinger_low']
        X = data[features].dropna()
        y = np.where(data['close'].shift(-1) > data['close'], 1, -1)[:len(X)]
        
        if len(X) == 0 or len(y) == 0:
            st.error("No valid data for model training")
            return
        
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        
        backtest_results = vectorized_backtest(data)
        if not backtest_results.empty:
            explainer = shap.TreeExplainer(model)
            backtest_results['shap_values'] = explainer.shap_values(X)
        
        main_dashboard(data, backtest_results)
    except Exception as e:
        st.error(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    main()
