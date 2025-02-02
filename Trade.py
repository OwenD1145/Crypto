import streamlit as st
import pandas as pd
import numpy as np
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import ta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="Crypto Trading Bot", page_icon="📈", layout="wide")

class TradingBot:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.historical_client = CryptoHistoricalDataClient()
        self.trading_client = None
        
    def create_features(self, df):
        """Create technical indicators for trading"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Price features
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log1p(features['returns'])
            
            # Moving averages
            for window in [5, 10, 20, 50, 100]:
                features[f'SMA_{window}'] = ta.trend.sma_indicator(df['close'], window=window)
                features[f'EMA_{window}'] = ta.trend.ema_indicator(df['close'], window=window)
            
            # RSI
            features['RSI'] = ta.momentum.rsi(df['close'])
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            features['MACD'] = macd.macd()
            features['MACD_signal'] = macd.macd_signal()
            features['MACD_diff'] = macd.macd_diff()
            
            # Volume indicators
            features['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            
            # Volatility
            features['volatility'] = df['close'].rolling(window=20).std()
            features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(window=20).mean()
            
            # Momentum
            features['momentum'] = ta.momentum.momentum_indicator(df['close'])
            features['roc'] = ta.momentum.roc(df['close'])
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            features['bb_upper'] = bb.bollinger_hband()
            features['bb_lower'] = bb.bollinger_lband()
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb.bollinger_mavg()
            
            # Fill NaN values
            features = features.fillna(method='bfill').fillna(method='ffill')
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            raise
    
    def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime):
        """Fetch historical data from Alpaca"""
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start_date,
                end=end_date
            )
            bars = self.historical_client.get_crypto_bars(request)
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.loc[symbol]
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def train_model(self, df: pd.DataFrame, features: pd.DataFrame):
        """Train the trading model"""
        try:
            # Create target variable (1 for price increase, 0 for decrease)
            target = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Remove NaN values
            features = features.dropna()
            target = target.dropna()
            
            # Align features and target
            features = features.loc[target.index]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, shuffle=False
            )
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                min_samples_split=50,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, train_pred),
                'test_accuracy': accuracy_score(y_test, test_pred),
                'feature_importance': dict(zip(features.columns, 
                                            self.model.feature_importances_))
            }
            
            return metrics, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def calculate_performance_metrics(self, df: pd.DataFrame, predictions: np.ndarray):
        """Calculate trading performance metrics"""
        try:
            # Calculate returns based on predictions
            df['strategy_returns'] = df['returns'].shift(-1) * predictions
            
            # Calculate cumulative returns
            df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
            
            # Calculate metrics
            total_return = df['cumulative_returns'].iloc[-1] - 1
            daily_returns = df['strategy_returns'].mean() * 252
            daily_vol = df['strategy_returns'].std() * np.sqrt(252)
            sharpe_ratio = daily_returns / daily_vol if daily_vol != 0 else 0
            
            # Calculate maximum drawdown
            rolling_max = df['cumulative_returns'].expanding().max()
            drawdowns = df['cumulative_returns'] / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise
    
    def plot_results(self, df: pd.DataFrame, metrics: dict):
        """Plot trading results"""
        try:
            # Create returns plot
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Scatter(
                x=df.index,
                y=df['cumulative_returns'],
                name='Strategy Returns'
            ))
            fig_returns.update_layout(
                title='Cumulative Returns',
                xaxis_title='Date',
                yaxis_title='Returns'
            )
            
            # Create feature importance plot
            importance_df = pd.DataFrame({
                'Feature': list(metrics['feature_importance'].keys()),
                'Importance': list(metrics['feature_importance'].values())
            }).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance'
            )
            
            return fig_returns, fig_importance
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            raise

def main():
    st.title("🤖 Crypto Trading Bot")
    
    # Initialize bot
    if 'bot' not in st.session_state:
        st.session_state.bot = TradingBot()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API Configuration
    api_key = st.sidebar.text_input("Alpaca API Key", type="password")
    api_secret = st.sidebar.text_input("Alpaca API Secret", type="password")
    
    if api_key and api_secret:
        st.session_state.bot.trading_client = TradingClient(api_key, api_secret, paper=True)
    
    # Trading parameters
    symbol = st.sidebar.selectbox(
        "Trading Pair",
        ["BTC/USD", "ETH/USD", "SOL/USDT"]
    )
    
    lookback_days = st.sidebar.slider(
        "Training Data (days)",
        min_value=7,
        max_value=90,
        value=30
    )
    
    position_size = st.sidebar.number_input(
        "Position Size (USD)",
        min_value=10,
        max_value=1000,
        value=100
    )
    
    # Main area
    tab1, tab2 = st.tabs(["Training", "Live Trading"])
    
    with tab1:
        if st.button("Train Model"):
            try:
                with st.spinner("Fetching historical data..."):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=lookback_days)
                    df = st.session_state.bot.fetch_historical_data(
                        symbol.replace("/", ""),
                        start_date,
                        end_date
                    )
                
                with st.spinner("Creating features..."):
                    features = st.session_state.bot.create_features(df)
                
                with st.spinner("Training model..."):
                    metrics, X_test, y_test = st.session_state.bot.train_model(df, features)
                
                with st.spinner("Calculating performance..."):
                    performance = st.session_state.bot.calculate_performance_metrics(
                        df.loc[X_test.index],
                        st.session_state.bot.model.predict(X_test)
                    )
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Return", f"{performance['total_return']:.2%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{performance['max_drawdown']:.2%}")
                
                # Display model accuracy
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Accuracy", f"{metrics['train_accuracy']:.2%}")
                with col2:
                    st.metric("Test Accuracy", f"{metrics['test_accuracy']:.2%}")
                
                # Plot results
                fig_returns, fig_importance = st.session_state.bot.plot_results(
                    df.loc[X_test.index],
                    metrics
                )
                st.plotly_chart(fig_returns, use_container_width=True)
                st.plotly_chart(fig_importance, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during training: {e}")
    
    with tab2:
        if st.session_state.bot.model is None:
            st.warning("Please train the model first")
        elif st.session_state.bot.trading_client is None:
            st.warning("Please enter valid API credentials")
        else:
            if 'running' not in st.session_state:
                st.session_state.running = False
            
            if st.button("Start Trading" if not st.session_state.running else "Stop Trading"):
                st.session_state.running = not st.session_state.running
            
            if st.session_state.running:
                placeholder = st.empty()
                
                while st.session_state.running:
                    try:
                        # Get current market data
                        current_data = st.session_state.bot.fetch_historical_data(
                            symbol.replace("/", ""),
                            datetime.now() - timedelta(hours=1),
                            datetime.now()
                        )
                        
                        # Create features
                        current_features = st.session_state.bot.create_features(current_data)
                        
                        # Get prediction
                        prediction = st.session_state.bot.model.predict(
                            current_features.iloc[-1:])
                        probability = st.session_state.bot.model.predict_proba(
                            current_features.iloc[-1:])[0]
                        
                        # Update display
                        with placeholder.container():
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    "Current Price",
                                    f"${current_data['close'].iloc[-1]:.2f}"
                                )
                                st.metric(
                                    "Signal",
                                    "BUY" if prediction[0] == 1 else "SELL",
                                    delta="↑" if prediction[0] == 1 else "↓"
                                )
                            
                            with col2:
                                st.metric(
                                    "Confidence",
                                    f"{max(probability):.2%}"
                                )
                                st.metric(
                                    "Last Update",
                                    datetime.now().strftime("%H:%M:%S")
                                )
                        
                        time.sleep(60)  # Update every minute
                        
                    except Exception as e:
                        logger.error(f"Error in trading loop: {e}")
                        st.error(f"Trading error: {e}")
                        st.session_state.running = False
                        break

if __name__ == "__main__":
    main()
