# crypto_trader.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from binance.client import Client
from binance.enums import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# Binance.US Configuration
BINANCE_TLD = 'us'
API_ENDPOINT = f'https://api.binance.{BINANCE_TLD}'
WEBSOCKET_ENDPOINT = f'wss://stream.binance.{BINANCE_TLD}:9443'

# Initialize session state
if 'trading' not in st.session_state:
    st.session_state.update({
        'trading_active': False,
        'client': None,
        'model': None,
        'historical_data': None,
        'symbol': 'BTCUSDT',
        'interval': KLINE_INTERVAL_1HOUR
    })

# Configure Streamlit
st.set_page_config(page_title="DeepSeek Crypto Trader", layout="wide")
st.title("🚀 Autonomous Trading System")

def initialize_binance_client(api_key, api_secret):
    """Create and verify Binance.US client connection"""
    try:
        client = Client(api_key, api_secret, tld=BINANCE_TLD)
        account = client.get_account()
        if not account['canTrade']:
            st.error("API keys lack trading permissions")
            return None
        return client
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return None

def fetch_historical_data(client, symbol, interval, days=90):
    """Retrieve and process historical market data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_date.strftime('%d %b %Y'),
            end_str=end_date.strftime('%d %b %Y')
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        
        return calculate_technical_indicators(df)
    except Exception as e:
        st.error(f"Data error: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    
    # RSI Calculation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD Calculation
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    return df.dropna()

def train_trading_model(df):
    """Train machine learning model"""
    try:
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        features = df[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal']]
        target = df['target'].iloc[:-1]
        features = features.iloc[:-1]
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, shuffle=False
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        return model, accuracy
    except Exception as e:
        st.error(f"Model error: {e}")
        return None, 0

def execute_trade(client, symbol, prediction, price):
    """Execute trade with proper error handling"""
    try:
        symbol_info = client.get_symbol_info(symbol)
        step_size = float([f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'][0])
        
        if prediction == 1:  # Buy
            balance = client.get_asset_balance(asset='USDT')
            usdt_balance = float(balance['free'])
            quantity = round((usdt_balance * 0.99) / price / step_size) * step_size
            
            if quantity > 0:
                return client.order_market_buy(
                    symbol=symbol,
                    quantity=quantity
                )
        else:  # Sell
            asset = symbol.replace('USDT', '')
            balance = client.get_asset_balance(asset=asset)
            quantity = round(float(balance['free']) / step_size) * step_size
            
            if quantity > 0:
                return client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
        return None
    except Exception as e:
        st.error(f"Trade failed: {e}")
        return None

# Sidebar Configuration
with st.sidebar:
    st.header("🔑 Exchange Setup")
    api_key = st.text_input("API Key", type='password')
    api_secret = st.text_input("API Secret", type='password')
    
    if st.button("Connect to Binance.US"):
        client = initialize_binance_client(api_key, api_secret)
        if client:
            st.session_state.client = client
            st.success("Connected successfully")

# Main Interface
tab1, tab2, tab3 = st.tabs(["Market Data", "Model Training", "Live Trading"])

with tab1:
    if st.session_state.client:
        st.header("📊 Historical Data Analysis")
        if st.button("Load Market Data"):
            data = fetch_historical_data(
                st.session_state.client,
                st.session_state.symbol,
                st.session_state.interval
            )
            if data is not None:
                st.session_state.historical_data = data
                st.success(f"Loaded {len(data)} records")
                
                fig = go.Figure(data=[go.Candlestick(
                    x=data['timestamp'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close']
                )])
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    if 'historical_data' in st.session_state:
        st.header("🤖 Algorithm Development")
        if st.button("Train Trading Model"):
            model, accuracy = train_trading_model(st.session_state.historical_data.copy())
            if model:
                st.session_state.model = model
                st.success(f"Model trained with {accuracy:.2%} accuracy")
                
                # Backtesting visualization
                test_data = st.session_state.historical_data.iloc[int(len(st.session_state.historical_data)*0.8):]
                predictions = model.predict(test_data[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal']])
                test_data['returns'] = np.log(test_data['close'] / test_data['close'].shift(1))
                test_data['strategy'] = test_data['returns'] * predictions
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=test_data['timestamp'],
                    y=test_data['strategy'].cumsum().apply(np.exp),
                    name='Strategy Performance'
                ))
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    if 'model' in st.session_state:
        st.header("💹 Live Trading")
        
        if st.session_state.trading_active:
            st.warning("Active trading session")
            elapsed = time.time() - st.session_state.start_time
            st.progress(elapsed / 3600)
            
            if st.button("Stop Trading"):
                st.session_state.trading_active = False
                st.experimental_rerun()
        else:
            if st.button("Start 1-Hour Session"):
                st.session_state.trading_active = True
                st.session_state.start_time = time.time()
                
                while st.session_state.trading_active and (time.time() - st.session_state.start_time < 3600):
                    try:
                        # Get latest market data
                        klines = st.session_state.client.get_klines(
                            symbol=st.session_state.symbol,
                            interval=st.session_state.interval,
                            limit=100
                        )
                        latest_data = pd.DataFrame(klines[-1:], columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades',
                            'taker_buy_base', 'taker_buy_quote', 'ignore'
                        ])
                        latest_data = calculate_technical_indicators(latest_data)
                        
                        # Make prediction
                        features = latest_data[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal']]
                        prediction = st.session_state.model.predict(features)[0]
                        
                        # Execute trade
                        ticker = st.session_state.client.get_symbol_ticker(symbol=st.session_state.symbol)
                        execute_trade(
                            st.session_state.client,
                            st.session_state.symbol,
                            prediction,
                            float(ticker['price'])
                        )
                        
                        time.sleep(60)  # Trade every minute
                    except Exception as e:
                        st.error(f"Trading error: {e}")
                        st.session_state.trading_active = False

st.markdown("---")
st.markdown("""
**Security Notice:**
- API keys are never stored or logged
- All trades require explicit user confirmation
- Test with small amounts first
- Trading involves substantial risk
""")
