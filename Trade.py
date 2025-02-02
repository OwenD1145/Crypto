import streamlit as st
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Page config
st.set_page_config(
    page_title="Solana Trading Bot",
    page_icon="📈",
    layout="wide"
)

# Initialize session state for API
if 'api' not in st.session_state:
    st.session_state.api = None

# Helper functions
def calculate_rsi(prices, periods=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    """Calculate MACD and Signal Line"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def create_features(df, sma_short, sma_long, rsi_period):
    """Create technical indicators and features"""
    df['SMA_short'] = df['close'].rolling(window=sma_short).mean()
    df['SMA_long'] = df['close'].rolling(window=sma_long).mean()
    df['RSI'] = calculate_rsi(df['close'], periods=rsi_period)
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=20).std()
    df['MACD'], df['MACD_signal'] = calculate_macd(df['close'])
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df.to_numpy()
    return df

def initialize_api(api_key, api_secret):
    """Initialize Alpaca API"""
    try:
        api = tradeapi.REST(api_key, api_secret, 'https://paper-api.alpaca.markets')
        # Test API connection
        api.get_account()
        return api
    except Exception as e:
        st.error(f"API initialization failed: {str(e)}")
        return None

def train_model(api, symbol, timeframe, start_date, end_date, model_params, feature_params):
    """Train the trading model with specified parameters"""
    # Fetch historical data
    historical_data = api.get_crypto_bars(
        symbol,
        timeframe,
        start=start_date,
        end=end_date
    ).df

    # Create features
    df = create_features(
        historical_data,
        feature_params['sma_short'],
        feature_params['sma_long'],
        feature_params['rsi_period']
    )
    df = df.dropna()

    # Create target variable
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Prepare features for training
    feature_columns = [
        'SMA_short', 'SMA_long', 'RSI', 'price_change', 'volatility',
        'MACD', 'MACD_signal', 'volume_ratio'
    ]
    
    X = df[feature_columns]
    y = df['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train model
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    # Generate predictions for backtesting
    df['predicted_signal'] = model.predict(X)
    df['strategy_returns'] = df['price_change'].shift(-1) * df['predicted_signal']
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    df['buy_hold_returns'] = (1 + df['price_change']).cumprod()

    return model, df, feature_columns, model.score(X_test, y_test)

# Streamlit UI
def main():
    st.title("🤖 Solana Trading Bot Dashboard")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API Configuration
    st.sidebar.subheader("API Settings")
    api_key = st.sidebar.text_input("Alpaca API Key", type="password")
    api_secret = st.sidebar.text_input("Alpaca API Secret", type="password")
    symbol = st.sidebar.selectbox(
        "Trading Pair",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"]
    )

    # Trading Parameters
    st.sidebar.subheader("Trading Parameters")
    position_size = st.sidebar.number_input("Position Size (USDT)", .01, 10.00, 1.00)
    timeframe = st.sidebar.selectbox("Trading Timeframe", 
                                   ["1Min", "5Min", "15Min", "1Hour", "1Day"])
  
    # Initialize API if credentials are provided
    if api_key and api_secret:
        if st.session_state.api is None:
            st.session_state.api = initialize_api(api_key, api_secret)
    
    # Training Parameters
    st.sidebar.subheader("Training Parameters")
    lookback_days = st.sidebar.slider("Historical Data (days)", 30, 365, 180)
    
    # Feature Parameters
    st.sidebar.subheader("Technical Indicators")
    sma_short = st.sidebar.slider("Short SMA Period", 5, 50, 20)
    sma_long = st.sidebar.slider("Long SMA Period", 20, 200, 50)
    rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)
    
    # Model Parameters
    st.sidebar.subheader("Model Parameters")
    n_estimators = st.sidebar.slider("Number of Trees", 50, 300, 100)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 100, 50)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 100, 20)
    
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.feature_columns = None
        st.session_state.running = False

    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Training and Backtesting")
        
        if st.button("Train Model"):
            if st.session_state.api is None:
                st.error("Please enter valid API credentials first")
            else:
                try:
                    # Prepare parameters
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=lookback_days)
                    
                    model_params = {
                        'n_estimators': n_estimators,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'random_state': 42
                    }
                    
                    feature_params = {
                        'sma_short': sma_short,
                        'sma_long': sma_long,
                        'rsi_period': rsi_period
                    }
                    
                    with st.spinner('Training model...'):
                        model, backtest_results, feature_columns, accuracy = train_model(
                            st.session_state.api, symbol, timeframe, 
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d'),
                            model_params, feature_params
                        )
                        
                        st.session_state.model = model
                        st.session_state.feature_columns = feature_columns
                        
                        # Display backtest results
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=backtest_results.index,
                            y=backtest_results['cumulative_returns'],
                            name='Strategy Returns'
                        ))
                        fig.add_trace(go.Scatter(
                            x=backtest_results.index,
                            y=backtest_results['buy_hold_returns'],
                            name='Buy & Hold Returns'
                        ))
                        fig.update_layout(title='Backtest Results',
                                        xaxis_title='Date',
                                        yaxis_title='Cumulative Returns')
                        st.plotly_chart(fig)

                      

                        # Display metrics
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric("Model Accuracy", f"{accuracy:.2%}")
                        with metrics_col2:
                            total_returns = backtest_results['cumulative_returns'].iloc[-1] - 1
                            st.metric("Total Returns", f"{total_returns:.2%}")
                        with metrics_col3:
                            sharpe = np.sqrt(365) * (backtest_results['strategy_returns'].mean() 
                                                   / backtest_results['strategy_returns'].std())
                            st.metric("Sharpe Ratio", f"{sharpe:.2f}")

                        backtest_results[:]
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
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
                try:
                    while st.session_state.running:
                        # Get current data
                        current_data = st.session_state.api.get_crypto_bars(symbol, timeframe).df
                        current_data = create_features(
                            current_data, 
                            sma_short,
                            sma_long,
                            rsi_period)
                            
                            
                            #     feature_params['sma_short'],
                        #     feature_params['sma_long'],
                        #     feature_params['rsi_period']
                        # )
                        # current_data = current_data.dropna()
                                           
                       
        
                        # Make prediction
                        current_features = current_data[st.session_state.feature_columns].iloc[-1].values.reshape(1, -1)
                        prediction = st.session_state.model.predict(current_features)[0]

                        if prediction == 1:  # Predicted price increase
                            try:
                                position = api.get_position(symbol)

                            except:
                                api.submit_order(
                                    symbol,
                                    qty=1,
                                    side='buy',
                                    type='market',
                                    time_in_force='gtc'
                                )
                        else:  # Predicted price decrease
                            try:
                                position = api.get_position(symbol)
                                api.submit_order(
                                    symbol,
                                    qty=1,
                                    side='sell',
                                    type='market',
                                    time_in_force='gtc'
                                )
                            except:
                                print("No position to sell...")
                        # Update display
                        with placeholder.container():
                            st.metric("Current Price", 
                                    f"${current_data['close'].iloc[-1]:.2f}")
                            st.metric("Trading Signal",
                                    (f"Buy order at {datetime.now()}") if prediction == 1 else (f"Sell order at {datetime.now()}"))
                            st.metric("Last Updated",
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        
                        time.sleep(60)  # Update every minute
                        
                except Exception as e:
                    st.error(f"Trading Error: {str(e)}")
                    st.session_state.running = False

if __name__ == "__main__":
    main()
