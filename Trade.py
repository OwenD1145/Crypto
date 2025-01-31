# trading_app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from binance.client import Client
from binance import ThreadedWebsocketManager
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# Initialize session state
if 'trading_active' not in st.session_state:
    st.session_state.update({
        'trading_active': False,
        'start_time': None,
        'client': None,
        'hist_data': None,
        'model': None,
        'twm': None
    })

# Configure page
st.set_page_config(
    page_title="DeepSeek Crypto Trader",
    page_icon="🚀",
    layout="wide"
)

# Technical indicators calculations
def compute_technical_indicators(df):
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['RSI'] = compute_rsi(df['close'], 14)
    df['MACD'] = compute_macd(df['close'])
    df['Signal_Line'] = df['MACD'].rolling(9).mean()
    df['Bollinger_Upper'], df['Bollinger_Lower'] = compute_bollinger_bands(df['close'])
    return df.dropna()

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(prices, fast=12, slow=26):
    return prices.ewm(span=fast).mean() - prices.ewm(span=slow).mean()

def compute_bollinger_bands(prices, window=20):
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return sma + 2*std, sma - 2*std

# Data fetching functions
def fetch_historical_data(client, symbol, interval, days=90):
    try:
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=(datetime.now() - timedelta(days=days)).strftime('%d %b %Y %H:%M:%S')
        )
        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return compute_technical_indicators(df)
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return None

# ML model functions
def prepare_training_data(df):
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = df[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'Bollinger_Upper', 'Bollinger_Lower']]
    return features, df['target']

def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Trading functions
def execute_trade_action(client, symbol, prediction, price):
    try:
        if prediction == 1:  # Buy signal
            balance = client.get_asset_balance(asset='USDT')
            usdt_balance = float(balance['free'])
            if usdt_balance > 10:
                quantity = (usdt_balance * 0.99) / price
                return client.order_market_buy(
                    symbol=symbol,
                    quantity=round(quantity, 5)
                )
        else:  # Sell signal
            asset = symbol.replace('USDT', '')
            balance = client.get_asset_balance(asset=asset)
            asset_balance = float(balance['free'])
            if asset_balance > 0:
                return client.order_market_sell(
                    symbol=symbol,
                    quantity=round(asset_balance, 5)
                )
        return None
    except Exception as e:
        st.error(f"Trade error: {str(e)}")
        return None

# UI Components
def show_real_time_data():
    st.header("📈 Real-Time Market Data")
    price_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    def handle_socket_message(msg):
        if msg['e'] == 'kline':
            candle = msg['k']
            price = float(candle['c'])
            price_placeholder.metric(f"Current {symbol} Price", f"${price:,.2f}")
    
    if 'twm' not in st.session_state or not st.session_state.twm.is_alive():
        st.session_state.twm = ThreadedWebsocketManager(
            api_key=api_key,
            api_secret=api_secret,
            tld='us'
        )
        st.session_state.twm.start()
        st.session_state.twm.start_kline_socket(
            callback=handle_socket_message,
            symbol=symbol,
            interval=interval
        )

# Main app structure
st.title("🚀 DeepSeek Autonomous Crypto Trader")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Binance US API Key", type='password')
    api_secret = st.text_input("Binance US API Secret", type='password')
    symbol = st.selectbox("Trading Pair", ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])
    interval = st.selectbox("Candle Interval", ['1m', '5m', '15m', '1h'])
    
    if st.button("🔌 Connect to Exchange"):
        try:
            st.session_state.client = Client(
                api_key=api_key,
                api_secret=api_secret,
                tld='us',
                testnet=True  # Remove for live trading
            )
            st.success("Successfully connected to Binance US!")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Real-Time", "History", "Model", "Trade"])

with tab1:
    if st.session_state.client:
        show_real_time_data()
    else:
        st.warning("Please connect to Binance US in the sidebar")

with tab2:
    if st.session_state.client:
        st.header("📊 Historical Analysis")
        if st.button("Load Historical Data"):
            with st.spinner("Fetching historical data..."):
                hist_data = fetch_historical_data(
                    st.session_state.client,
                    symbol,
                    interval
                )
                if hist_data is not None:
                    st.session_state.hist_data = hist_data
                    st.success(f"Loaded {len(hist_data)} records")
                    
                    fig = go.Figure(data=[go.Candlestick(
                        x=hist_data['time'],
                        open=hist_data['open'],
                        high=hist_data['high'],
                        low=hist_data['low'],
                        close=hist_data['close']
                    )])
                    fig.update_layout(title=f"{symbol} Price History")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Connect to Binance US first")

with tab3:
    if 'hist_data' in st.session_state:
        st.header("🤖 Algorithm Training")
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                features, target = prepare_training_data(st.session_state.hist_data.copy())
                model, accuracy = train_model(features, target)
                if model:
                    st.session_state.model = model
                    st.success(f"Model trained (Accuracy: {accuracy:.2%})")
                    
                    # Backtesting visualization
                    test_data = st.session_state.hist_data.iloc[int(len(features)*0.8):]
                    predictions = model.predict(features.iloc[int(len(features)*0.8):])
                    test_data['returns'] = np.log(test_data['close'] / test_data['close'].shift(1))
                    test_data['strategy'] = test_data['returns'] * predictions[:-1]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=test_data['time'],
                        y=test_data['strategy'].cumsum().apply(np.exp),
                        name='Strategy'
                    ))
                    fig.update_layout(title="Backtest Performance")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load historical data first")

with tab4:
    if 'model' in st.session_state:
        st.header("💸 Live Trading")
        if st.session_state.trading_active:
            st.warning("Trading session in progress")
            progress = (datetime.now() - st.session_state.start_time).seconds / 3600
            st.progress(min(progress, 1.0))
            
            if st.button("🛑 Stop Session"):
                st.session_state.trading_active = False
                st.session_state.twm.stop()
                st.experimental_rerun()
        else:
            if st.button("Start 1-Hour Session"):
                st.session_state.trading_active = True
                st.session_state.start_time = datetime.now()
                
                # Trading loop
                while st.session_state.trading_active and \
                     (datetime.now() - st.session_state.start_time).seconds < 3600:
                    
                    try:
                        # Get latest data
                        klines = st.session_state.client.get_klines(
                            symbol=symbol,
                            interval=interval,
                            limit=100
                        )
                        latest_data = pd.DataFrame(klines, columns=[
                            'time', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades',
                            'taker_buy_base', 'taker_buy_quote', 'ignore'
                        ])
                        latest_data = compute_technical_indicators(latest_data)
                        
                        # Make prediction
                        features = latest_data.iloc[-1][[
                            'SMA_20', 'SMA_50', 'RSI', 'MACD', 
                            'Signal_Line', 'Bollinger_Upper', 'Bollinger_Lower'
                        ]].values.reshape(1, -1)
                        
                        prediction = st.session_state.model.predict(features)[0]
                        price = float(st.session_state.client.get_symbol_ticker(symbol=symbol)['price'])
                        
                        # Execute trade
                        execute_trade_action(
                            st.session_state.client,
                            symbol,
                            prediction,
                            price
                        )
                        
                        time.sleep(60)  # Check every minute
                        
                    except Exception as e:
                        st.error(f"Trading error: {str(e)}")
                        st.session_state.trading_active = False
                
                st.session_state.trading_active = False
                st.experimental_rerun()
    else:
        st.info("Train the model first")

st.markdown("---")
st.markdown("""
**🔒 Security Note:**  
- API keys are never stored or transmitted  
- Trading sessions automatically expire after 1 hour  
- Always test with Binance Testnet first  
- Use at your own risk with real funds
""")
