import streamlit as st
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import ta
import logging
from typing import Dict, Tuple, List
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Bot",
    page_icon="📈",
    layout="wide"
)

class TradingBot:
    def __init__(self):
        self.api = None
        self.model = None
        self.feature_columns = None
        self.scaler = None
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators for trading"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Basic price features
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log1p(features['returns'])
            
            # Moving averages
            for window in [5, 10, 20, 50, 100]:
                features[f'SMA_{window}'] = ta.trend.sma_indicator(df['close'], window=window)
                features[f'EMA_{window}'] = ta.trend.ema_indicator(df['close'], window=window)
                
                # Moving average crossovers
                if window in [10, 20, 50]:
                    features[f'SMA_cross_{window}'] = (
                        features[f'SMA_5'] > features[f'SMA_{window}']).astype(int)
            
            # RSI
            features['RSI'] = ta.momentum.rsi(df['close'])
            features['RSI_MA'] = ta.trend.sma_indicator(features['RSI'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            features['MACD'] = macd.macd()
            features['MACD_signal'] = macd.macd_signal()
            features['MACD_diff'] = macd.macd_diff()
            features['MACD_cross'] = (features['MACD'] > features['MACD_signal']).astype(int)
            
            # Volume indicators
            features['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            features['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            
            # Volatility
            features['volatility'] = df['close'].rolling(window=20).std()
            features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(window=20).mean()
            
            # Momentum
            features['ROC'] = ta.momentum.roc(df['close'])
            features['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            features['BB_upper'] = bb.bollinger_hband()
            features['BB_lower'] = bb.bollinger_lband()
            features['BB_width'] = (features['BB_upper'] - features['BB_lower']) / bb.bollinger_mavg()
            features['BB_position'] = (df['close'] - features['BB_lower']) / (features['BB_upper'] - features['BB_lower'])
            
            # Additional indicators
            features['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'])
            features['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'])
            
            # Price patterns
            features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
            
            # Fill NaN values
            features = features.fillna(method='bfill').fillna(method='ffill')
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            raise

    def fetch_historical_data(self, symbol: str, timeframe: str, 
                            start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical data from Alpaca"""
        try:
            bars = self.api.get_crypto_bars(
                symbol,
                timeframe,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            ).df
            
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.loc[symbol]
                
            return bars
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise

    def train_model(self, df: pd.DataFrame, features: pd.DataFrame) -> Tuple[Dict, pd.DataFrame, pd.Series]:
        """Train the trading model with cross-validation"""
        try:
            # Create target variable (1 for price increase, 0 for decrease)
            target = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Remove NaN values
            features = features.dropna()
            target = target.dropna()
            
            # Align features and target
            features = features.loc[target.index]
            
            # Store feature columns
            self.feature_columns = features.columns.tolist()
            
            # Perform time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            
            for train_idx, test_idx in tscv.split(features):
                X_train = features.iloc[train_idx]
                X_test = features.iloc[test_idx]
                y_train = target.iloc[train_idx]
                y_test = target.iloc[test_idx]
                
                model = RandomForestClassifier(
                    n_estimators=100,
                    min_samples_split=50,
                    max_depth=10,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                cv_scores.append(accuracy_score(y_test, model.predict(X_test)))
            
            # Train final model on all data
            self.model = RandomForestClassifier(
                n_estimators=100,
                min_samples_split=50,
                max_depth=10,
                random_state=42
            )
            self.model.fit(features, target)
            
            # Calculate metrics
            predictions = self.model.predict(features)
            metrics = {
                'accuracy': accuracy_score(target, predictions),
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'feature_importance': dict(zip(features.columns, 
                                            self.model.feature_importances_))
            }
            
            return metrics, features, target
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    def calculate_performance_metrics(self, df: pd.DataFrame, predictions: np.ndarray) -> Dict:
        """Calculate trading performance metrics"""
        try:
            # Calculate returns based on predictions
            df = df.copy()
            df['strategy_returns'] = df['returns'].shift(-1) * predictions
            df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
            
            # Calculate metrics
            total_return = df['cumulative_returns'].iloc[-1] - 1
            annual_return = (1 + total_return) ** (252 / len(df)) - 1
            
            # Risk metrics
            daily_returns = df['strategy_returns'].mean() * 252
            daily_vol = df['strategy_returns'].std() * np.sqrt(252)
            sharpe_ratio = daily_returns / daily_vol if daily_vol != 0 else 0
            
            # Drawdown analysis
            rolling_max = df['cumulative_returns'].expanding().max()
            drawdowns = df['cumulative_returns'] / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # Win rate
            winning_trades = (df['strategy_returns'] > 0).sum()
            total_trades = (~df['strategy_returns'].isna()).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit factor
            gross_profits = df['strategy_returns'][df['strategy_returns'] > 0].sum()
            gross_losses = abs(df['strategy_returns'][df['strategy_returns'] < 0].sum())
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': total_trades
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise

    def plot_results(self, df: pd.DataFrame, metrics: Dict) -> Tuple[go.Figure, go.Figure, go.Figure]:
        """Create visualization plots"""
        try:
            # Returns plot
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Scatter(
                x=df.index,
                y=df['cumulative_returns'],
                name='Strategy Returns',
                line=dict(color='blue')
            ))
            fig_returns.update_layout(
                title='Cumulative Returns',
                xaxis_title='Date',
                yaxis_title='Returns',
                hovermode='x unified'
            )
            
            # Feature importance plot
            importance_df = pd.DataFrame({
                'Feature': list(metrics['feature_importance'].keys()),
                'Importance': list(metrics['feature_importance'].values())
            }).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(
                importance_df.tail(15),  # Show top 15 features
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top Feature Importance'
            )
            
            # Drawdown plot
            rolling_max = df['cumulative_returns'].expanding().max()
            drawdowns = df['cumulative_returns'] / rolling_max - 1
            
            fig_drawdown = go.Figure()
            fig_drawdown.add_trace(go.Scatter(
                x=df.index,
                y=drawdowns,
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='red')
            ))
            fig_drawdown.update_layout(
                title='Drawdown Analysis',
                xaxis_title='Date',
                yaxis_title='Drawdown',
                hovermode='x unified'
            )
            
            return fig_returns, fig_importance, fig_drawdown
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            raise

    def calculate_position_size(self, account_value: float, confidence: float, 
                              current_price: float) -> float:
        """Calculate position size based on confidence and risk management"""
        try:
            # Base position size on account value and confidence
            base_risk = 0.02  # 2% risk per trade
            position_risk = base_risk * confidence  # Adjust risk based on confidence
            
            # Calculate maximum position size
            max_position_value = account_value * position_risk
            position_size = max_position_value / current_price
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            raise

    def execute_trade(self, symbol: str, side: str, quantity: float, 
                     stop_loss_pct: float, take_profit_pct: float) -> Dict:
        """Execute trade with stop loss and take profit orders"""
        try:
            # Get current price
            current_price = float(self.api.get_latest_trade(symbol).price)
            
            # Calculate stop loss and take profit prices
            stop_loss = current_price * (1 - stop_loss_pct) if side == 'buy' else \
                       current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct) if side == 'buy' else \
                         current_price * (1 - take_profit_pct)
            
            # Submit main order
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type='market',
                time_in_force='gtc'
            )
            
            # Wait for fill
            filled_order = self.api.get_order(order.id)
            fill_price = float(filled_order.filled_avg_price)
            
            # Submit stop loss order
            stop_loss_order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell' if side == 'buy' else 'buy',
                type='stop',
                stop_price=stop_loss,
                time_in_force='gtc'
            )
            
            # Submit take profit order
            take_profit_order = self.api.submit_order(
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
            logger.error(f"Error executing trade: {e}")
            raise

def initialize_session_state():
    """Initialize session state variables"""
    if 'bot' not in st.session_state:
        st.session_state.bot = TradingBot()
    if 'running' not in st.session_state:
        st.session_state.running = False
def main():
    st.title("🤖 Advanced Crypto Trading Bot")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API Configuration
    api_key = st.sidebar.text_input("Alpaca API Key", type="password")
    api_secret = st.sidebar.text_input("Alpaca API Secret", type="password")
    
    if api_key and api_secret:
        if st.session_state.bot.api is None:
            try:
                st.session_state.bot.api = tradeapi.REST(
                    api_key,
                    api_secret,
                    'https://paper-api.alpaca.markets',
                    api_version='v2'
                )
                st.sidebar.success("API Connected!")
            except Exception as e:
                st.sidebar.error(f"API Connection Error: {e}")
    
    # Trading Parameters
    st.sidebar.subheader("Trading Parameters")
    
    symbol = st.sidebar.selectbox(
        "Trading Pair",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    )
    
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1Min", "5Min", "15Min", "1Hour"]
    )
    
    lookback_days = st.sidebar.slider(
        "Training Data (days)",
        min_value=7,
        max_value=90,
        value=30
    )
    
    # Risk Management Parameters
    st.sidebar.subheader("Risk Management")
    
    risk_per_trade = st.sidebar.slider(
        "Risk per Trade (%)",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1
    )
    
    stop_loss_pct = st.sidebar.slider(
        "Stop Loss (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1
    )
    
    take_profit_pct = st.sidebar.slider(
        "Take Profit (%)",
        min_value=1.0,
        max_value=10.0,
        value=4.0,
        step=0.1
    )
    
    confidence_threshold = st.sidebar.slider(
        "Minimum Confidence Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.6
    )
    
    # Main area tabs
    tab1, tab2 = st.tabs(["Model Training", "Live Trading"])
    
    with tab1:
        st.subheader("Model Training and Backtesting")
        
        if st.button("Train Model"):
            if st.session_state.bot.api is None:
                st.error("Please configure API credentials first")
            else:
                try:
                    with st.spinner("Fetching historical data..."):
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=lookback_days)
                        
                        df = st.session_state.bot.fetch_historical_data(
                            symbol.replace("/", ""),
                            timeframe,
                            start_date,
                            end_date
                        )
                    
                    with st.spinner("Creating features..."):
                        features = st.session_state.bot.create_features(df)
                    
                    with st.spinner("Training model..."):
                        metrics, features_df, target = st.session_state.bot.train_model(
                            df, features
                        )
                    
                    with st.spinner("Calculating performance..."):
                        performance = st.session_state.bot.calculate_performance_metrics(
                            df,
                            st.session_state.bot.model.predict(features_df)
                        )
                    
                    # Display metrics
                    st.subheader("Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Total Return",
                            f"{performance['total_return']:.2%}"
                        )
                        st.metric(
                            "Annual Return",
                            f"{performance['annual_return']:.2%}"
                        )
                    
                    with col2:
                        st.metric(
                            "Sharpe Ratio",
                            f"{performance['sharpe_ratio']:.2f}"
                        )
                        st.metric(
                            "Win Rate",
                            f"{performance['win_rate']:.2%}"
                        )
                    
                    with col3:
                        st.metric(
                            "Max Drawdown",
                            f"{performance['max_drawdown']:.2%}"
                        )
                        st.metric(
                            "Profit Factor",
                            f"{performance['profit_factor']:.2f}"
                        )
                    
                    # Model metrics
                    st.subheader("Model Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Model Accuracy",
                            f"{metrics['accuracy']:.2%}"
                        )
                    with col2:
                        st.metric(
                            "CV Score",
                            f"{metrics['cv_mean']:.2%} ± {metrics['cv_std']:.2%}"
                        )
                    
                    # Plot results
                    fig_returns, fig_importance, fig_drawdown = \
                        st.session_state.bot.plot_results(df, metrics)
                    
                    st.plotly_chart(fig_returns, use_container_width=True)
                    st.plotly_chart(fig_drawdown, use_container_width=True)
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    logger.error(f"Training error: {e}")
    
    with tab2:
        st.subheader("Live Trading")
        
        if st.session_state.bot.model is None:
            st.warning("Please train the model first")
        elif st.session_state.bot.api is None:
            st.warning("Please configure API credentials")
        else:
            if st.button("Start Trading" if not st.session_state.running else "Stop Trading"):
                st.session_state.running = not st.session_state.running
            
            if st.session_state.running:
                placeholder = st.empty()
                
                while st.session_state.running:
                    try:
                        # Get current market data
                        current_data = st.session_state.bot.fetch_historical_data(
                            symbol.replace("/", ""),
                            timeframe,
                            datetime.now() - timedelta(hours=2),
                            datetime.now()
                        )
                        
                        # Create features
                        current_features = st.session_state.bot.create_features(current_data)
                        
                        # Get prediction and probability
                        prediction = st.session_state.bot.model.predict(
                            current_features.iloc[-1:])
                        probability = st.session_state.bot.model.predict_proba(
                            current_features.iloc[-1:])[0]
                        confidence = max(probability)
                        
                        # Get account information
                        account = st.session_state.bot.api.get_account()
                        buying_power = float(account.buying_power)
                        
                        # Get current position
                        try:
                            position = st.session_state.bot.api.get_position(
                                symbol.replace("/", ""))
                            has_position = True
                        except:
                            has_position = False
                        
                        # Update display
                        with placeholder.container():
                            col1, col2, col3 = st.columns(3)
                            
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
                                    f"{confidence:.2%}"
                                )
                                st.metric(
                                    "Buying Power",
                                    f"${buying_power:.2f}"
                                )
                            
                            with col3:
                                st.metric(
                                    "Position",
                                    "Yes" if has_position else "No"
                                )
                                st.metric(
                                    "Last Update",
                                    datetime.now().strftime("%H:%M:%S")
                                )
                            
                            # Execute trades
                            if confidence >= confidence_threshold:
                                current_price = current_data['close'].iloc[-1]
                                
                                if prediction[0] == 1 and not has_position:
                                    # Calculate position size
                                    position_size = st.session_state.bot.calculate_position_size(
                                        buying_power,
                                        confidence,
                                        current_price
                                    )
                                    
                                    # Execute buy order
                                    trade_result = st.session_state.bot.execute_trade(
                                        symbol.replace("/", ""),
                                        'buy',
                                        position_size,
                                        stop_loss_pct,
                                        take_profit_pct
                                    )
                                    
                                    if trade_result:
                                        st.success(
                                            f"Bought {position_size:.6f} {symbol} " \
                                            f"at ${trade_result['fill_price']:.2f}"
                                        )
                                
                                elif prediction[0] == 0 and has_position:
                                    # Execute sell order
                                    trade_result = st.session_state.bot.execute_trade(
                                        symbol.replace("/", ""),
                                        'sell',
                                        float(position.qty),
                                        stop_loss_pct,
                                        take_profit_pct
                                    )
                                    
                                    if trade_result:
                                        st.success(
                                            f"Sold {position.qty} {symbol} " \
                                            f"at ${trade_result['fill_price']:.2f}"
                                        )
                        
                        time.sleep(60)  # Update every minute
                        
                    except Exception as e:
                        logger.error(f"Error in trading loop: {e}")
                        st.error(f"Trading error: {e}")
                        st.session_state.running = False
                        break

if __name__ == "__main__":
    main()
