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
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(index=prices.index, data=50)

def calculate_macd(prices: pd.Series, slow: int = 26, fast: int = 12, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD and Signal Line"""
    try:
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd.fillna(0), signal_line.fillna(0)
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        return pd.Series(index=prices.index, data=0), pd.Series(index=prices.index, data=0)

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    try:
        middle_band = prices.rolling(window=window).mean()
        std_dev = prices.rolling(window=window).std()
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        return upper_band.fillna(prices), middle_band.fillna(prices), lower_band.fillna(prices)
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        return prices, prices, prices

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    try:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.fillna(tr)
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        return pd.Series(index=high.index, data=0)

def create_features(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Create technical indicators and features"""
    try:
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
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
        features['BB_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Volume indicators
        features['volume_ma'] = df['volume'].rolling(window=20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma']
        
        # Additional indicators
        features['MFI'] = ta.volume.money_flow_index(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            fillna=True
        )
        
        features['ADX'] = ta.trend.adx(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            fillna=True
        )
        
        # ATR ratio
        atr = calculate_atr(df['high'], df['low'], df['close'])
        features['ATR_ratio'] = atr / df['close']
        
        # Fill any remaining NaN values
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        return features
    
    except Exception as e:
        logger.error(f"Error in feature creation: {str(e)}")
        st.error(f"Feature creation failed: {str(e)}")
        return pd.DataFrame()
def calculate_performance_metrics(returns: pd.Series) -> Dict:
    """Calculate trading performance metrics"""
    try:
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
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        return {}

def train_model(api, symbol: str, timeframe: str, start_date: str, end_date: str, 
                model_params: Dict, feature_params: Dict) -> Tuple:
    """Train and backtest the trading model"""
    try:
        # Convert symbol format for Alpaca (remove '/')
        alpaca_symbol = symbol.replace('/', '')
        
        st.info(f"Fetching data for {alpaca_symbol}...")
        
        # Fetch historical data
        historical_bars = api.get_crypto_bars(
            alpaca_symbol,
            timeframe,
            start=start_date,
            end=end_date
        ).df
        
        # Reset index to make timestamp a column
        historical_data = historical_bars.reset_index()
        
        if historical_data.empty:
            st.error("No data received from API")
            return None, None, None, None
            
        st.info(f"Creating features from {len(historical_data)} data points...")
        
        # Create features
        features = create_features(historical_data, feature_params)
        
        # Define feature columns
        feature_columns = [
            'SMA_short', 'SMA_long', 'RSI', 'price_change', 'volatility',
            'MACD', 'MACD_signal', 'volume_ratio', 'BB_width', 'MFI', 'ADX', 'ATR_ratio'
        ]
        
        # Ensure all feature columns exist
        missing_columns = [col for col in feature_columns if col not in features.columns]
        if missing_columns:
            st.error(f"Missing features: {missing_columns}")
            return None, None, None, None
        
        # Create target variable (1 if price goes up, 0 if down)
        features['target'] = (historical_data['close'].shift(-1) > 
                            historical_data['close']).astype(int)
        
        # Remove NaN values
        features = features.dropna()
        
        if len(features) < 100:
            st.error("Insufficient data for training")
            return None, None, None, None
        
        st.info("Training model...")
        
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
        
        # Calculate returns
        features['returns'] = historical_data['close'].pct_change().shift(-1) * features['predicted_signal']
        
        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(features['returns'].dropna())
        
        st.success("Model training completed successfully!")
        
        return model, features, feature_columns, performance_metrics
    
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        st.error(f"Model training failed: {str(e)}")
        st.exception(e)
        return None, None, None, None

def calculate_position_size(account_value: float, risk_per_trade: float, 
                          current_price: float, stop_loss_pct: float) -> float:
    """Calculate position size based on risk management rules"""
    try:
        max_loss_amount = account_value * (risk_per_trade / 100)
        stop_loss_amount = current_price * (stop_loss_pct / 100)
        position_size = max_loss_amount / stop_loss_amount
        return min(position_size, account_value / current_price)
    except Exception as e:
        logger.error(f"Error calculating position size: {str(e)}")
        return 0.0

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
        st.error(f"Trade execution failed: {str(e)}")
        return None
      def run_trading_loop(placeholder, model, feature_columns, api, symbol: str, 
                    timeframe: str, params: Dict):
    """Main trading loop with risk management"""
    try:
        # Convert symbol format for Alpaca
        alpaca_symbol = symbol.replace('/', '')
        
        # Get current market data
        current_data = api.get_crypto_bars(alpaca_symbol, timeframe).df
        
        if current_data.empty:
            st.error("Unable to fetch current market data")
            return None
            
        # Create features
        features = create_features(current_data, params)
        
        if features.empty:
            st.error("Failed to create features for current data")
            return None
            
        # Select only required feature columns
        features = features[feature_columns].copy()
        
        # Make prediction
        current_features = features.iloc[-1:].copy()
        prediction = model.predict(current_features)[0]
        
        # Get account information
        account = api.get_account()
        buying_power = float(account.buying_power)
        
        # Get current position
        try:
            position = api.get_position(alpaca_symbol)
            current_position = {
                'side': 'long' if float(position.qty) > 0 else 'short',
                'quantity': abs(float(position.qty)),
                'entry_price': float(position.avg_entry_price)
            }
        except Exception as e:
            current_position = None
        
        # Calculate trading signals and risk parameters
        current_price = current_data['close'].iloc[-1]
        stop_loss_price = current_price * (1 - params['stop_loss_pct']/100) if prediction == 1 else \
                         current_price * (1 + params['stop_loss_pct']/100)
        
        position_size = calculate_position_size(
            buying_power,
            params['risk_per_trade'],
            current_price,
            params['stop_loss_pct']
        )
        
        # Update display
        with placeholder.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric("Signal", "BUY" if prediction == 1 else "SELL",
                         delta="↑" if prediction == 1 else "↓")
            
            with col2:
                st.metric("Position Size", f"{position_size:.4f}")
                st.metric("Stop Loss", f"${stop_loss_price:.2f}")
            
            with col3:
                st.metric("Buying Power", f"${buying_power:.2f}")
                st.metric("Risk per Trade", f"{params['risk_per_trade']}%")
            
            # Display current indicators
            st.subheader("Current Indicators")
            feature_df = pd.DataFrame(current_features.iloc[0]).T
            st.dataframe(feature_df.style.format("{:.2f}"))
            
            # Display current position if exists
            if current_position:
                st.subheader("Current Position")
                position_df = pd.DataFrame([current_position])
                st.dataframe(position_df)
        
        return {
            'prediction': prediction,
            'current_price': current_price,
            'stop_loss_price': stop_loss_price,
            'position_size': position_size,
            'current_position': current_position
        }
        
    except Exception as e:
        logger.error(f"Error in trading loop: {str(e)}")
        st.error(f"Trading loop error: {str(e)}")
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
    symbol = st.sidebar.selectbox(
        "Trading Pair",
        ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "MATIC/USD"]
    )
    
    timeframe = st.sidebar.selectbox(
        "Trading Timeframe",
        ["1Min", "5Min", "15Min", "1Hour", "1Day"]
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
                            if backtest_results is not None and not backtest_results.empty:
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
                                
                                if performance:
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
                                        'Metric': ['Win Rate', 'Profit Factor', 'Avg Win', 'Avg Loss'],
                                        'Value': [
                                            f"{performance['win_rate']:.2%}",
                                            f"{performance['profit_factor']:.2f}",
                                            f"{performance['avg_win']:.2%}",
                                            f"{performance['avg_loss']:.2%}"
                                        ]
                                    })
                                    st.dataframe(metrics_df)
                        else:
                            st.error("Model training failed. Please check the errors above.")
                            
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
                    st.exception(e)
    
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
                                symbol.replace('/', ''),
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
                                symbol.replace('/', ''),
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

