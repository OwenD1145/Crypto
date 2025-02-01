# trading_bot.py
import streamlit as st
from binance.client import Client
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import warnings

warnings.filterwarnings('ignore')

# --- Constants ---
RISK_PER_TRADE = 0.02  # 2% of portfolio per trade
STOP_LOSS = 0.02  # 2% stop loss
TAKE_PROFIT = 0.04  # 4% take profit
MAX_RETRIES = 3
REQUEST_DELAY = 6  # Binance rate limit: 10 requests/minute

# --- Security Disclaimer ---
st.sidebar.markdown("""
**WARNING:** Trading carries substantial risk. 
This application is for demonstration purposes only. 
Never share API keys with untrusted parties.
""")

# --- Binance Connection ---
def connect_binance(api_key, api_secret, testnet=True):
    client = Client(api_key, api_secret, testnet=testnet)
    try:
        client.get_account()
        return client
    except Exception as e:
        st.error(f"Connection failed: {str(e)}")
        return None

# --- Data Fetching with Retries ---
@st.cache_data(ttl=3600)
def get_historical_data(client, symbol, interval, start_date):
    for _ in range(MAX_RETRIES):
        try:
            klines = client.get_historical_klines(symbol, interval, start_date)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'tb_base_volume',
                'tb_quote_volume', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            return df.set_index('timestamp').dropna()
        except Exception as e:
            time.sleep(REQUEST_DELAY)
    st.error("Failed to fetch historical data")
    return None

# --- Technical Indicators ---
def add_technical_indicators(df):
    try:
        # RSI
        rsi = RSIIndicator(df['close'], window=14)
        df['rsi'] = rsi.rsi()
        
        # MACD
        macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        
        return df.dropna()
    except ZeroDivisionError:
        st.error("Error in technical indicators calculation")
        return df

# --- Model Training ---
def train_model(df):
    try:
        # Create target (next hour's price movement)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df.dropna(inplace=True)

        X = df[['rsi', 'macd', 'bb_upper', 'bb_lower', 'volume']]
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        model = lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100
        ).fit(X_train, y_train)
        
        # Calculate metrics
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        return model, accuracy, cm
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None, None

# --- Risk Management ---
def calculate_position_size(client, symbol, current_price):
    try:
        balance = client.get_asset_balance(asset='USDT')['free']
        portfolio_value = float(balance)
        risk_amount = portfolio_value * RISK_PER_TRADE
        return round(risk_amount / current_price, 5)
    except:
        return 0.001  # Fallback position size

# --- Trade Execution ---
def execute_trade(client, model, symbol, current_data):
    try:
        features = pd.DataFrame([[
            current_data['rsi'],
            current_data['macd'],
            current_data['bb_upper'],
            current_data['bb_lower'],
            current_data['volume']
        ]], columns=['rsi', 'macd', 'bb_upper', 'bb_lower', 'volume'])
        
        prediction = model.predict(features)[0]
        price = current_data['close']
        quantity = calculate_position_size(client, symbol, price)
        
        if quantity <= 0:
            return "Insufficient funds"
            
        if prediction == 1:
            order = client.create_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity
            )
            return f"Bought {quantity} {symbol}"
        else:
            order = client.create_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity
            )
            return f"Sold {quantity} {symbol}"
    except Exception as e:
        return f"Trade failed: {str(e)}"

# --- UI Components ---
def display_model_metrics(accuracy, cm):
    st.subheader("Model Performance")
    st.metric("Test Accuracy", f"{accuracy:.2%}")
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 1'],
        colorscale='Blues'
    ))
    st.plotly_chart(fig)

def display_live_trading():
    st.subheader("Live Trading Dashboard")
    status_text = st.empty()
    chart_placeholder = st.empty()
    return status_text, chart_placeholder

# --- Main Application ---
def main():
    st.title("AutoBinance Trader")
    
    # API Inputs
    api_key = st.text_input("Binance API Key", type="password")
    api_secret = st.text_input("Binance API Secret", type="password")
    testnet = st.checkbox("Use Testnet", value=True)
    
    if api_key and api_secret:
        client = connect_binance(api_key, api_secret, testnet=testnet)
        if not client:
            return
            
        # Asset Selection
        symbol = st.selectbox("Trading Pair", ["BTCUSDT", "ETHUSDT", "ADAUSDT"])
        interval = Client.KLINE_INTERVAL_1HOUR
        
        # Historical Data
        df = get_historical_data(client, symbol, interval, "2020-01-01")
        if df is not None:
            df = add_technical_indicators(df)
            
            # Display Price Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ))
            st.plotly_chart(fig)
            
            # Model Training
            if st.button("Train Trading Model"):
                with st.spinner("Training model..."):
                    model, accuracy, cm = train_model(df)
                    if model:
                        display_model_metrics(accuracy, cm)
                        st.session_state.model = model
                        
            # Live Trading Session
            if 'model' in st.session_state and st.button("Start Live Trading (1 Hour)"):
                status_text, chart = display_live_trading()
                start_time = time.time()
                
                while time.time() - start_time < 3600:
                    try:
                        # Get current data
                        kline = client.get_klines(symbol=symbol, interval=interval, limit=1)[0]
                        current_data = {
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5]),
                        }
                        current_data = add_technical_indicators(
                            pd.DataFrame([current_data], index=[pd.Timestamp.now()])
                        ).iloc[-1].to_dict()
                        
                        # Execute trade
                        trade_result = execute_trade(client, st.session_state.model, symbol, current_data)
                        status_text.write(f"{time.ctime()} | {trade_result}")
                        
                        # Update chart
                        fig.add_trace(go.Scatter(
                            x=[pd.Timestamp.now()],
                            y=[current_data['close']],
                            mode='markers',
                            marker=dict(color='red' if 'Sold' in trade_result else 'green', size=10),
                            name='Trades'
                        ))
                        chart.plotly_chart(fig, use_container_width=True)
                        
                        time.sleep(REQUEST_DELAY)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        time.sleep(REQUEST_DELAY * 2)

if __name__ == "__main__":
    main()
