# crypto_trading_app.py (fixed data alignment)
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

# Data Acquisition with minimum data check
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
            klines = client.get_historical_klines(symbol, interval, timeframe)
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

# Technical Analysis with forward-fill for indicators
def add_technical_indicators(df):
    if df.empty:
        return df
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12).macd_diff()
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['bollinger_high'] = bb.bollinger_hband()
    df['bollinger_low'] = bb.bollinger_lband()
    return df.fillna(method='ffill').dropna()

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

# Backtesting Engine with data length check
def vectorized_backtest(data):
    if len(data) < 20:  # Minimum data length requirement
        return pd.DataFrame()
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = np.where(data['rsi'] < 30, 1), np.where(data['rsi'] > 70, -1, 0)
    signals['returns'] = data['close'].pct_change()
    signals['strategy'] = signals['signal'].shift(1) * signals['returns']
    return signals

# UI Components (unchanged)
def sidebar_controls():
    with st.sidebar:
        st.header("Trading Controls")
        source = st.selectbox("Data Source", ["Yahoo Finance", "Binance"])
        symbol = st.text_input("Symbol", "BTC-USD")
        timeframe = st.selectbox("Timeframe", ["1d", "1w", "1mo", "1y"])
        interval = st.selectbox("Interval", ["1h", "4h", "1d"])
        return source, symbol, timeframe, interval

def main_dashboard(data, backtest_results):
    # ... (unchanged UI components from previous version) ...

# Main App with data validation
def main():
    source, symbol, timeframe, interval = sidebar_controls()
    
    data = fetch_data(source, symbol, timeframe, interval)
    if data.empty:
        st.warning("No data found - check your inputs")
        return
    
    data = add_technical_indicators(data)
    
    # Data validation for modeling
    if len(data) < 100:  # Minimum data threshold for training
        st.warning(f"Need at least 100 data points for modeling (current: {len(data)})")
        return
    
    # Align features and target
    features = ['rsi', 'macd', 'bollinger_high', 'bollinger_low']
    X = data[features].dropna()
    y = np.where(data['close'].shift(-1) > data['close'], 1, -1)
    
    # Trim to match lengths
    min_length = min(len(X), len(y))
    X = X.iloc[:min_length]
    y = y[:min_length]
    
    if len(X) == 0:
        st.error("No valid data for training after preprocessing")
        return
    
    model = RandomForestClassifier()
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    backtest_results = vectorized_backtest(data)
    
    if not backtest_results.empty:
        backtest_results['shap_values'] = explainer.shap_values(X)
    else:
        st.warning("Backtesting requires more historical data")
    
    main_dashboard(data, backtest_results)

if __name__ == "__main__":
    main()
