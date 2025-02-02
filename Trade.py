import streamlit as st
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="Solana Trading Bot", page_icon="📈", layout="wide")

# Initialize session states
if 'api' not in st.session_state:
    st.session_state.api = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'running' not in st.session_state:
    st.session_state.running = False

def create_features(df):
    """Create technical indicators for trading"""
    df = df.copy()
    
    # Add SMA indicators
    df['SMA20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['SMA50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    
    # Add RSI
    df['RSI'] = RSIIndicator(close=df['close']).rsi()
    
    # Add MACD
    macd = MACD(close=df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # Add Bollinger Bands
    bb = BollingerBands(close=df['close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    
    # Add price changes
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=20).std()
    
    # Fill NaN values
    df = df.fillna(method='bfill')
    
    return df

def train_model(api, start_date, end_date):
    """Train the trading model"""
    try:
        # Fetch historical data
        bars = api.get_crypto_bars(
            ['SOLUSD'],
            '1Min',
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        ).df
        
        if isinstance(bars.index, pd.MultiIndex):
            df = bars.loc['SOLUSD'].copy()
        else:
            df = bars.copy()
            
        # Create features
        df = create_features(df)
        
        # Create target (1 if price goes up, 0 if down)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Select features for training
        features = ['SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_signal', 
                   'price_change', 'volatility']
        X = df[features].dropna()
        y = df['target'].dropna()
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
        model.fit(X[-len(y):], y)
        
        return model, features
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        st.error(f"Training error: {e}")
        return None, None

def get_trading_signal(model, features, current_data):
    """Get trading signal from model"""
    try:
        prediction = model.predict(current_data[features].iloc[-1:])
        probability = model.predict_proba(current_data[features].iloc[-1:])[0]
        return prediction[0], probability
    except Exception as e:
        logger.error(f"Error getting trading signal: {e}")
        return None, None

def execute_trade(api, symbol, side, quantity):
    """Execute trade on Alpaca"""
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        return order
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return None

def main():
    st.title("🌊 Solana Trading Bot")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # API Configuration
    api_key = st.sidebar.text_input("Alpaca API Key", type="password")
    api_secret = st.sidebar.text_input("Alpaca API Secret", type="password")
    
    # Trading Parameters
    position_size = st.sidebar.number_input("Position Size (USD)", 
                                          min_value=10, 
                                          max_value=1000, 
                                          value=100)
    
    min_confidence = st.sidebar.slider("Minimum Confidence", 
                                     min_value=0.5, 
                                     max_value=0.95, 
                                     value=0.6)
    
    # Initialize API if credentials provided
    if api_key and api_secret:
        if st.session_state.api is None:
            try:
                st.session_state.api = tradeapi.REST(
                    api_key,
                    api_secret,
                    'https://paper-api.alpaca.markets'
                )
                st.success("Connected to Alpaca Paper Trading!")
            except Exception as e:
                st.error(f"API Connection Error: {e}")
    
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Training")
        if st.button("Train Model"):
            if st.session_state.api is None:
                st.error("Please enter valid API credentials first")
            else:
                with st.spinner("Training model..."):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)  # Use 7 days of data
                    model, features = train_model(st.session_state.api, 
                                               start_date, 
                                               end_date)
                    if model is not None:
                        st.session_state.model = model
                        st.session_state.features = features
                        st.success("Model trained successfully!")
    
    with col2:
        st.subheader("Live Trading")
        if st.session_state.model is None:
            st.warning("Please train the model first")
        elif st.session_state.api is None:
            st.warning("Please enter valid API credentials")
        else:
            if st.button("Start Trading" if not st.session_state.running else "Stop Trading"):
                st.session_state.running = not st.session_state.running
            
            if st.session_state.running:
                placeholder = st.empty()
                
                while st.session_state.running:
                    try:
                        # Get current market data
                        current_data = st.session_state.api.get_crypto_bars(
                            ['SOLUSD'],
                            '1Min'
                        ).df
                        
                        if isinstance(current_data.index, pd.MultiIndex):
                            current_data = current_data.loc['SOLUSD']
                        
                        # Create features
                        current_data = create_features(current_data)
                        
                        # Get trading signal
                        signal, proba = get_trading_signal(
                            st.session_state.model,
                            st.session_state.features,
                            current_data
                        )
                        
                        # Get current position
                        try:
                            position = st.session_state.api.get_position('SOLUSD')
                            has_position = True
                        except:
                            has_position = False
                        
                        # Trading logic
                        current_price = current_data['close'].iloc[-1]
                        quantity = position_size / current_price
                        
                        with placeholder.container():
                            metrics_col1, metrics_col2 = st.columns(2)
                            
                            with metrics_col1:
                                st.metric("Current Price", 
                                        f"${current_price:.2f}")
                                st.metric("Signal", 
                                        "BUY" if signal == 1 else "SELL",
                                        delta="↑" if signal == 1 else "↓")
                            
                            with metrics_col2:
                                st.metric("Confidence", 
                                        f"{max(proba[0], proba[1]):.2%}")
                                st.metric("Position", 
                                        "Yes" if has_position else "No")
                            
                            # Execute trades
                            if not has_position and signal == 1 and max(proba[0], proba[1]) >= min_confidence:
                                order = execute_trade(st.session_state.api, 
                                                    'SOLUSD', 
                                                    'buy', 
                                                    quantity)
                                if order:
                                    st.success(f"Bought {quantity:.4f} SOL at ${current_price:.2f}")
                            
                            elif has_position and signal == 0 and max(proba[0], proba[1]) >= min_confidence:
                                order = execute_trade(st.session_state.api, 
                                                    'SOLUSD', 
                                                    'sell', 
                                                    position.qty)
                                if order:
                                    st.success(f"Sold {position.qty} SOL at ${current_price:.2f}")
                        
                        time.sleep(60)  # Wait for 1 minute
                        
                    except Exception as e:
                        logger.error(f"Error in trading loop: {e}")
                        st.error(f"Trading error: {e}")
                        st.session_state.running = False
                        break

if __name__ == "__main__":
    main()
