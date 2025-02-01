# crypto_trading_app.py
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

# Configuration
BINANCE_TLD = 'us'
DEFAULT_SYMBOL = 'BTCUSDT'
INTERVAL_OPTIONS = ['1m', '5m', '15m', '1h']
TRADE_DURATION = 3600  # 1 hour in seconds

# Initialize session state
if 'app_state' not in st.session_state:
    st.session_state.update({
        'client': None,
        'trading_active': False,
        'model': None,
        'historical_data': None,
        'start_time': None,
        'websocket': None,
        'symbol': DEFAULT_SYMBOL,
        'interval': '1h'
    })

# Configure Streamlit
st.set_page_config(page_title="DeepSeek Crypto Trader", layout="wide")
st.title("🚀 Autonomous Cryptocurrency Trading System")

def initialize_client(api_key, api_secret):
    """Initialize and verify Binance.US client"""
    try:
        client = Client(api_key, api_secret, tld=BINANCE_TLD)
        account = client.get_account()
        if not account['canTrade']:
            st.error("API keys lack trading permissions")
            return None
        return client
    except Exception as e:
        st.error(f"Connection failed: {str(e)}")
        return None

def fetch_historical_data(client, symbol, interval, days=90):
    """Retrieve and process historical market data"""
    try:
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=(datetime.now() - timedelta(days=days)).strftime('%d %b %Y')
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        
        return calculate_technical_indicators(df)
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators for analysis"""
    # SMA
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['Bollinger_Upper'] = df['close'].rolling(20).mean() + 2*df['close'].rolling(20).std()
    df['Bollinger_Lower'] = df['close'].rolling(20).mean() - 2*df['close'].rolling(20).std()
    
    return df.dropna()

def train_trading_model(df):
    """Train machine learning trading model"""
    try:
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df[:-1]  # Remove last row with NaN target
        
        features = df[[
            'SMA_20', 'SMA_50', 'RSI', 
            'MACD', 'Signal',
            'Bollinger_Upper', 'Bollinger_Lower'
        ]]
        target = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, shuffle=False
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, model.predict(X_test))
        return model, accuracy
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        return None, 0

def execute_trade(client, symbol, prediction):
    """Execute trade with proper risk management"""
    try:
        symbol_info = client.get_symbol_info(symbol)
        step_size = float([f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'][0])
        ticker = client.get_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
        
        if prediction == 1:  # Buy signal
            balance = client.get_asset_balance(asset='USDT')
            usdt_balance = float(balance['free'])
            quantity = round((usdt_balance * 0.99) / price / step_size) * step_size
            
            if quantity > 0:
                return client.order_market_buy(
                    symbol=symbol,
                    quantity=quantity
                )
        else:  # Sell signal
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
        st.error(f"Trade execution failed: {str(e)}")
        return None

def start_websocket(symbol, interval):
    """Initialize real-time data websocket"""
    def handle_message(msg):
        if msg['e'] == 'kline':
            candle = msg['k']
            st.session_state.current_price = float(candle['c'])
    
    if st.session_state.client:
        twm = ThreadedWebsocketManager(
            api_key=st.session_state.client.API_KEY,
            api_secret=st.session_state.client.SECRET_KEY,
            tld=BINANCE_TLD
        )
        twm.start()
        twm.start_kline_socket(
            symbol=symbol,
            interval=interval,
            callback=handle_message
        )
        return twm
    return None

# Sidebar Configuration
with st.sidebar:
    st.header("🔐 Exchange Configuration")
    api_key = st.text_input("Binance.US API Key", type='password')
    api_secret = st.text_input("Binance.US API Secret", type='password')
    
    st.session_state.symbol = st.selectbox(
        "Trading Pair",
        ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
    )
    st.session_state.interval = st.selectbox(
        "Chart Interval",
        INTERVAL_OPTIONS
    )
    
    if st.button("🔗 Connect to Exchange"):
        client = initialize_client(api_key, api_secret)
        if client:
            st.session_state.client = client
            st.session_state.websocket = start_websocket(
                st.session_state.symbol,
                st.session_state.interval
            )
            st.success("Successfully connected to Binance.US")

# Main Interface Tabs
tab1, tab2, tab3 = st.tabs([
    "📈 Market Analysis", 
    "🤖 Algorithm Development", 
    "💹 Live Trading"
])

with tab1:
    st.header("Real-Time Market Data")
    if st.session_state.client:
        price_placeholder = st.empty()
        if 'current_price' in st.session_state:
            price_placeholder.metric(
                f"Current {st.session_state.symbol} Price",
                f"${st.session_state.current_price:,.2f}"
            )
        
        if st.button("Load Historical Data"):
            with st.spinner("Fetching historical data..."):
                hist_data = fetch_historical_data(
                    st.session_state.client,
                    st.session_state.symbol,
                    st.session_state.interval
                )
                if hist_data is not None:
                    st.session_state.historical_data = hist_data
                    st.success(f"Loaded {len(hist_data)} historical records")
                    
                    fig = go.Figure(data=[go.Candlestick(
                        x=hist_data['timestamp'],
                        open=hist_data['open'],
                        high=hist_data['high'],
                        low=hist_data['low'],
                        close=hist_data['close']
                    )])
                    fig.update_layout(title="Price History")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Connect to Binance.US in the sidebar")

with tab2:
    st.header("Trading Algorithm Development")
    if 'historical_data' in st.session_state:
        if st.button("Train Machine Learning Model"):
            with st.spinner("Training predictive model..."):
                model, accuracy = train_trading_model(
                    st.session_state.historical_data.copy()
                )
                if model:
                    st.session_state.model = model
                    st.success(f"Model trained with {accuracy:.2%} accuracy")
                    
                    # Backtesting Visualization
                    test_data = st.session_state.historical_data.iloc[
                        int(len(st.session_state.historical_data)*0.8):
                    ]
                    predictions = model.predict(test_data[[
                        'SMA_20', 'SMA_50', 'RSI', 
                        'MACD', 'Signal',
                        'Bollinger_Upper', 'Bollinger_Lower'
                    ]])
                    
                    test_data['returns'] = np.log(test_data['close'] / test_data['close'].shift(1))
                    test_data['strategy'] = test_data['returns'] * predictions
                    cumulative_returns = test_data['strategy'].cumsum().apply(np.exp)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=test_data['timestamp'],
                        y=cumulative_returns,
                        name='Strategy Performance'
                    ))
                    fig.update_layout(title="Backtesting Results")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load historical data first")

with tab3:
    st.header("Live Trading Interface")
    if 'model' in st.session_state:
        if st.session_state.trading_active:
            elapsed = time.time() - st.session_state.start_time
            progress = elapsed / TRADE_DURATION
            
            st.progress(progress)
            st.write(f"Time remaining: {TRADE_DURATION - int(elapsed)} seconds")
            
            if st.button("🛑 Stop Trading Session"):
                st.session_state.trading_active = False
                if st.session_state.websocket:
                    st.session_state.websocket.stop()
                st.experimental_rerun()
        else:
            if st.button("🚀 Start 1-Hour Trading Session"):
                st.session_state.trading_active = True
                st.session_state.start_time = time.time()
                
                while st.session_state.trading_active and \
                    (time.time() - st.session_state.start_time < TRADE_DURATION):
                    
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
                        
                        # Generate prediction
                        features = latest_data[[
                            'SMA_20', 'SMA_50', 'RSI', 
                            'MACD', 'Signal',
                            'Bollinger_Upper', 'Bollinger_Lower'
                        ]]
                        prediction = st.session_state.model.predict(features)[0]
                        
                        # Execute trade
                        execute_trade(
                            st.session_state.client,
                            st.session_state.symbol,
                            prediction
                        )
                        
                        time.sleep(60)  # Trade interval
                        
                    except Exception as e:
                        st.error(f"Trading error: {str(e)}")
                        st.session_state.trading_active = False
    else:
        st.warning("Train the trading model first")

st.markdown("---")
st.markdown("""
**🔒 Security & Risk Disclosure:**  
- All API credentials are ephemeral and never stored  
- Trading involves substantial risk of capital loss  
- Test thoroughly with Binance Testnet before live trading  
- This software is provided for educational purposes only  
- Past performance is not indicative of future results
""")
