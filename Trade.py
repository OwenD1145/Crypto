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
import ta
import logging
from typing import Dict, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Crypto Trading Bot",
    page_icon="📈",
    layout="wide"
)

# Initialize session states
if 'api' not in st.session_state:
    st.session_state.api = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'position' not in st.session_state:
    st.session_state.position = None
if 'last_trade_time' not in st.session_state:
    st.session_state.last_trade_time = None

# Technical Analysis Helper Functions
def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, slow: int = 26, fast: int = 12, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD and Signal Line"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    middle_band = prices.rolling(window=window).mean()
    std_dev = prices.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return upper_band, middle_band, lower_band

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def create_features(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Create technical indicators and features"""
    features = pd.DataFrame(index=df.index)
    
    # Basic price features
    features['SMA_short'] = df['close'].rolling(window=params['sma_short']).mean()
    features['SMA_long'] = df['close'].rolling(window=params['sma_long']).mean()
    features['price_change'] = df['close'].pct_change()
    features['volatility'] = df['close'].rolling(window=20).std()
    
    # RSI
    features['RSI'] = calculate_rsi(df['close'], periods=params['rsi_period'])
    
    # MACD
    features['MACD'], features['MACD_signal'] = calculate_macd(df['close'])
    features['MACD_hist'] = features['MACD'] - features['MACD_signal']
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
    features['BB_upper'] = bb_upper
    features['BB_middle'] = bb_middle
    features['BB_lower'] = bb_lower
    features['BB_width'] = (bb_upper - bb_lower) / bb_middle
    
    # Volume indicators
    features['volume_ma'] = df['volume'].rolling(window=20).mean()
    features['volume_ratio'] = df['volume'] / features['volume_ma']
    features['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # Momentum indicators
    features['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
    features['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'])
    
    # Volatility indicators
    features['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    features['ATR_ratio'] = features['ATR'] / df['close']
    
    return features

def calculate_position_size(account_value: float, risk_per_trade: float, 
                          current_price: float, stop_loss_pct: float) -> float:
    """Calculate position size based on risk management rules"""
    max_loss_amount = account_value * (risk_per_trade / 100)
    stop_loss_amount = current_price * (stop_loss_pct / 100)
    position_size = max_loss_amount / stop_loss_amount
    return min(position_size, account_value / current_price)

def calculate_stop_loss(entry_price: float, position_type: str, 
                       stop_loss_pct: float, atr_multiple: float = 2) -> float:
    """Calculate stop loss price based on percentage or ATR"""
    if position_type == 'long':
        return entry_price * (1 - stop_loss_pct / 100)
    else:
        return entry_price * (1 + stop_loss_pct / 100)
def calculate_performance_metrics(returns: pd.Series) -> Dict:
    """Calculate trading performance metrics"""
    metrics = {}
    
    # Basic returns metrics
    metrics['total_return'] = (returns + 1).prod() - 1
    metrics['annual_return'] = (1 + metrics['total_return']) ** (252 / len(returns)) - 1
    metrics['daily_sharpe'] = np.sqrt(252) * returns.mean() / returns.std()
    
    # Drawdown analysis
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    metrics['max_drawdown'] = drawdowns.min()
    
    # Win rate and profit metrics
    metrics['win_rate'] = (returns > 0).sum() / len(returns)
    metrics['profit_factor'] = abs(returns[returns > 0].sum() / returns[returns < 0].sum())
    metrics['avg_win'] = returns[returns > 0].mean()
    metrics['avg_loss'] = returns[returns < 0].mean()
    
    return metrics

def train_model(api, symbol: str, timeframe: str, start_date: str, end_date: str, 
                model_params: Dict, feature_params: Dict) -> Tuple:
    """Train and backtest the trading model"""
    try:
        # Fetch historical data
        historical_data = api.get_crypto_bars(
            symbol,
            timeframe,
            start=start_date,
            end=end_date
        ).df
        
        # Create features
        features = create_features(historical_data, feature_params)
        
        # Define feature columns
        feature_columns = [col for col in features.columns 
                         if col not in ['target', 'predicted_signal', 'returns']]
        
        # Create target variable
        features['target'] = (historical_data['close'].shift(-1) > 
                            historical_data['close']).astype(int)
        
        # Remove NaN values
        features = features.dropna()
        
        # Split data
        X = features[feature_columns]
        y = features['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Train model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # Generate predictions for backtesting
        features['predicted_signal'] = model.predict(X)
        features['returns'] = (historical_data['close'].pct_change().shift(-1) * 
                             features['predicted_signal'])
        
        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(features['returns'].dropna())
        
        return model, features, feature_columns, performance_metrics
    
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        return None, None, None, None

def execute_trade(api, symbol: str, side: str, quantity: float, 
                 stop_loss: float, take_profit: float = None) -> Dict:
    """Execute trade with stop loss and take profit"""
    try:
        # Submit main order
        order = api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        
        # Wait for fill
        filled_order = api.get_order(order.id)
        fill_price = float(filled_order.filled_avg_price)
        
        # Submit stop loss order
        stop_loss_order = api.submit_order(
            symbol=symbol,
            qty=quantity,
            side='sell' if side == 'buy' else 'buy',
            type='stop',
            stop_price=stop_loss,
            time_in_force='gtc'
        )
        
        # Submit take profit order if specified
        take_profit_order = None
        if take_profit:
            take_profit_order = api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell' if side == 'buy' else 'buy',
                type='limit',
                limit_price=take_profit,
                time_in_force='gtc'
            )
        
        return {
            'main_order': order,
            'stop_loss_order': stop_loss_order,
            'take_profit_order': take_profit_order,
            'fill_price': fill_price
        }
        
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        return None

def run_trading_loop(placeholder, model, feature_columns, api, symbol: str, 
                    timeframe: str, params: Dict):
    """Main trading loop with risk management"""
    try:
        # Get current market data
        current_data = api.get_crypto_bars(symbol, timeframe).df
        
        # Create features
        features = create_features(current_data, params)
        features = features[feature_columns].copy()
        
        # Make prediction
        current_features = pd.DataFrame(features.iloc[-1:])
        prediction = model.predict(current_features)[0]
        
        # Get account information
        account = api.get_account()
        buying_power = float(account.buying_power)
        
        # Get current position
        try:
            position = api.get_position(symbol)
            current_position = {
                'side': 'long' if float(position.qty) > 0 else 'short',
                'quantity': abs(float(position.qty)),
                'entry_price': float(position.avg_entry_price)
            }
        except:
            current_position = None
        
        # Calculate trading signals and risk parameters
        current_price = current_data['close'].iloc[-1]
        stop_loss_price = calculate_stop_loss(
            current_price, 
            'long' if prediction == 1 else 'short',
            params['stop_loss_pct']
        )
        
        position_size = calculate_position_size(
            buying_power,
            params['risk_per_trade'],
            current_price,
            params['stop_loss_pct']
        )
        
        # Update display
        with placeholder.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric("Signal", "BUY" if prediction == 1 else "SELL")
                st.metric("Position Size", f"{position_size:.4f}")
            
            with col2:
                st.metric("Stop Loss", f"${stop_loss_price:.2f}")
                st.metric("Risk per Trade", f"{params['risk_per_trade']}%")
                st.metric("Buying Power", f"${buying_power:.2f}")
            
            # Display current feature values
            st.write("Current Indicators:")
            feature_df = pd.DataFrame(current_features.iloc[0]).T
            st.dataframe(feature_df)
            
            # Display current position if exists
            if current_position:
                st.write("Current Position:")
                st.json(current_position)
        
        return {
            'prediction': prediction,
            'current_price': current_price,
            'stop_loss_price': stop_loss_price,
            'position_size': position_size,
            'current_position': current_position
        }
        
    except Exception as e:
        logger.error(f"Error in trading loop: {str(e)}")
        return None
def main():
    st.title("🤖 Crypto Trading Bot Dashboard")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API Configuration
    st.sidebar.subheader("API Settings")
    api_key = st.sidebar.text_input("Alpaca API Key", type="password")
    api_secret = st.sidebar.text_input("Alpaca API Secret", type="password")
    
    # Initialize API if credentials are provided
    if api_key and api_secret:
        if st.session_state.api is None:
            try:
                st.session_state.api = tradeapi.REST(
                    api_key, 
                    api_secret, 
                    'https://paper-api.alpaca.markets'
                )
                st.success("API connection established!")
            except Exception as e:
                st.error(f"API initialization failed: {str(e)}")
    
    # Trading Parameters
    st.sidebar.subheader("Trading Parameters")
    
    # Asset selection
    symbol = str(st.sidebar.selectbox(
        "Trading Pair",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"]
    ))
    
    timeframe = st.sidebar.selectbox(
        "Trading Timeframe",
        ["1MIN", "5MIN", "15MIN", "1HOUR", "1DAY"]
    )
    
    # Risk Management Parameters
    risk_per_trade = st.sidebar.slider(
        "Risk per Trade (%)", 
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )
    
    stop_loss_pct = st.sidebar.slider(
        "Stop Loss (%)", 
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5
    )
    
    take_profit_pct = st.sidebar.slider(
        "Take Profit (%)", 
        min_value=1.0,
        max_value=20.0,
        value=4.0,
        step=0.5
    )
    
    # Technical Indicator Parameters
    st.sidebar.subheader("Technical Indicators")
    sma_short = st.sidebar.slider("Short SMA Period", 5, 50, 20)
    sma_long = st.sidebar.slider("Long SMA Period", 20, 200, 50)
    rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)
    
    # Model Parameters
    st.sidebar.subheader("Model Parameters")
    lookback_days = st.sidebar.slider("Historical Data (days)", 30, 365, 180)
    n_estimators = st.sidebar.slider("Number of Trees", 50, 300, 100)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 100, 50)
    
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
                        'random_state': 42
                    }
                    
                    feature_params = {
                        'sma_short': sma_short,
                        'sma_long': sma_long,
                        'rsi_period': rsi_period
                    }
                    
                    with st.spinner('Training model...'):
                        model, backtest_results, feature_columns, performance = train_model(
                            st.session_state.api,
                            symbol,
                            timeframe,
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d'),
                            model_params,
                            feature_params
                        )
                        
                        if model is not None:
                            st.session_state.model = model
                            st.session_state.feature_columns = feature_columns
                            
                            # Display backtest results
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=backtest_results.index,
                                y=(1 + backtest_results['returns']).cumprod(),
                                name='Strategy Returns'
                            ))
                            fig.update_layout(
                                title='Backtest Results',
                                xaxis_title='Date',
                                yaxis_title='Cumulative Returns'
                            )
                            st.plotly_chart(fig)
                            
                            # Display performance metrics
                            metrics_cols = st.columns(3)
                            with metrics_cols[0]:
                                st.metric(
                                    "Total Return",
                                    f"{performance['total_return']:.2%}"
                                )
                            with metrics_cols[1]:
                                st.metric(
                                    "Sharpe Ratio",
                                    f"{performance['daily_sharpe']:.2f}"
                                )
                            with metrics_cols[2]:
                                st.metric(
                                    "Max Drawdown",
                                    f"{performance['max_drawdown']:.2%}"
                                )
                            
                            # Additional metrics
                            st.write("Detailed Performance Metrics:")
                            metrics_df = pd.DataFrame({
                                'Metric': [
                                    'Win Rate',
                                    'Profit Factor',
                                    'Average Win',
                                    'Average Loss',
                                    'Annual Return'
                                ],
                                'Value': [
                                    f"{performance['win_rate']:.2%}",
                                    f"{performance['profit_factor']:.2f}",
                                    f"{performance['avg_win']:.2%}",
                                    f"{performance['avg_loss']:.2%}",
                                    f"{performance['annual_return']:.2%}"
                                ]
                            })
                            st.dataframe(metrics_df)
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Live Trading")
        if st.session_state.model is None:
            st.warning("Please train the model first")
        elif st.session_state.api is None:
            st.warning("Please enter valid API credentials")
        else:
            trading_params = {
                'risk_per_trade': risk_per_trade,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'sma_short': sma_short,
                'sma_long': sma_long,
                'rsi_period': rsi_period
            }
            
            if st.button("Start Trading" if not st.session_state.running else "Stop Trading"):
                st.session_state.running = not st.session_state.running
            
            if st.session_state.running:
                placeholder = st.empty()
                try:
                    while st.session_state.running:
                        trading_data = run_trading_loop(
                            placeholder,
                            st.session_state.model,
                            st.session_state.feature_columns,
                            st.session_state.api,
                            symbol,
                            timeframe,
                            trading_params
                        )
                        
                        if trading_data is None:
                            st.session_state.running = False
                            break
                        
                        # Execute trades based on signals and position
                        current_position = trading_data['current_position']
                        prediction = trading_data['prediction']
                        
                        if current_position is None and prediction == 1:
                            # Enter long position
                            trade_result = execute_trade(
                                st.session_state.api,
                                symbol,
                                'buy',
                                trading_data['position_size'],
                                trading_data['stop_loss_price'],
                                trading_data['current_price'] * (1 + take_profit_pct/100)
                            )
                            if trade_result:
                                st.success(f"Entered long position at {trade_result['fill_price']}")
                        
                        elif current_position and prediction == 0:
                            # Exit position
                            trade_result = execute_trade(
                                st.session_state.api,
                                symbol,
                                'sell',
                                current_position['quantity'],
                                None
                            )
                            if trade_result:
                                st.success(f"Exited position at {trade_result['fill_price']}")
                        
                        time.sleep(60)  # Update every minute
                        
                except Exception as e:
                    st.error(f"Trading Error: {str(e)}")
                    st.session_state.running = False

if __name__ == "__main__":
    main()
