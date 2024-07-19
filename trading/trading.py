import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.database import save_trade, Trade
from tensorflow.keras.models import load_model
import pennylane as qml
from pennylane import numpy as pnp
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import ccxt

MIN_ORDER_SIZES = {
    'BTC/USD': 0.0001,
    'ETH/USD': 0.01,
    'XRP/USD': 10,
}

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize PennyLane device
dev = qml.device('default.qubit', wires=1)

def check_balance(symbol, kraken):
    """Checks balance before trading."""
    try:
        balance = kraken.fetch_balance()
        logging.debug(f"Fetched balance: {balance}")
        asset = symbol.split('/')[0]

        if 'free' in balance:
            crypto_balance = balance['free'].get(asset, 0)
            fiat_balance = balance['free'].get('USD', 0)
            return crypto_balance, fiat_balance
        else:
            logging.error(f"Free balance not found in balance response. Full balance response: {balance}")
            return 0, 0
    except Exception as e:
        logging.error(f"Error checking balance: {e}")
        return 0, 0

def is_fund_sufficient(symbol, balance, required_amount, kraken):
    try:
        ticker = kraken.fetch_ticker(symbol)
        current_price = ticker['close']
        cost = current_price * required_amount
        if cost <= balance:
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Error checking funds for {symbol}: {e}")
        return False

def execute_trade(signal, symbol, balance, kraken):
    """Executes trade based on signal."""
    min_trade_size = MIN_ORDER_SIZES.get(symbol, 10)
    position_size = max(min_trade_size, balance * 0.05)
    
    if position_size < min_trade_size:
        logging.info(f"Insufficient funds to execute trade or position size less than minimum trade size ({min_trade_size})")
        return None

    if not is_fund_sufficient(symbol, balance, position_size, kraken):
        logging.error(f"Insufficient funds to execute trade for {symbol}")
        return None

    try:
        if signal == 'buy':
            order = kraken.create_market_buy_order(symbol, position_size)
        elif signal == 'sell':
            order = kraken.create_market_sell_order(symbol, position_size)
        else:
            logging.error(f"Invalid trading signal: {signal}")
            return None

        logging.info(f'{signal.upper()} order placed: {order}')
        trade = Trade(timestamp=pd.Timestamp.now(), symbol=symbol, order_type=signal, amount=order['amount'], price=order['price'], status='completed')
        save_trade(trade)
        return order
    except Exception as e:
        logging.error(f"Error placing {signal.upper()} order: {e}")
        trade = Trade(timestamp=pd.Timestamp.now(), symbol=symbol, order_type=signal, amount=position_size, price=0, status='failed')
        save_trade(trade)
        return None

def set_trailing_stop_loss_take_profit(symbol, order, kraken, trailing_stop_pct=0.02, take_profit_pct=0.05):
    """Sets trailing stop-loss and take-profit orders."""
    if order is None:
        logging.error("Order is None, cannot set trailing stop-loss and take-profit.")
        return

    try:
        if order['side'] == 'buy':
            trailing_stop_price = order['price'] * (1 - trailing_stop_pct)
            take_profit_price = order['price'] * (1 + take_profit_pct)
        else:
            trailing_stop_price = order['price'] * (1 + trailing_stop_pct)
            take_profit_price = order['price'] * (1 - take_profit_pct)

        kraken.create_order(symbol, 'TRAILING_STOP_MARKET', 'SELL', order['amount'], trailing_stop_price)
        kraken.create_order(symbol, 'TAKE_PROFIT_MARKET', 'SELL', order['amount'], take_profit_price)
        logging.info(f'Trailing stop-loss set at {trailing_stop_price}, take-profit set at {take_profit_price}')
    except Exception as e:
        logging.error(f"Error setting trailing stop-loss and take-profit: {e}")

def predict_signal(model, data, scaler, features, pca):
    """Predicts signal using the optimized model."""
    try:
        latest_data = data[features].values[-1].reshape(1, -1)
        latest_data_scaled = scaler.transform(latest_data)
        latest_data_pca = pca.transform(latest_data_scaled)
        prediction = model.predict(np.expand_dims(latest_data_pca, axis=-1))
        return 'buy' if prediction[0][0] > 0.5 else 'sell'
    except Exception as e:
        logging.error(f"Error predicting signal: {e}")
        raise

@qml.qnode(dev)
def quantum_circuit(data):
    """Quantum circuit for computing a feature."""
    qml.RY(data, wires=0)
    return qml.expval(qml.PauliZ(0))

def predict_signal_quantum(data, features):
    """Predicts signal using a quantum model."""
    try:
        latest_data = data[features].values[-1].reshape(1, -1)
        quantum_feature = quantum_circuit(latest_data).numpy()
        return 'buy' if quantum_feature > 0.5 else 'sell'
    except Exception as e:
        logging.error(f"Error predicting signal with quantum model: {e}")
        raise

def backtest_strategy(data, model, scaler, features, pca, initial_balance=10000, quantum=False):
    """Backtests the strategy."""
    data['signal'] = 0
    data['position'] = 0
    data['strategy_returns'] = 0
    balance = initial_balance
    position = 0

    for i in range(len(data)):
        if i == 0:
            continue

        if quantum:
            signal = predict_signal_quantum(data, features)
        else:
            latest_data = data[features].values[i].reshape(1, -1)
            latest_data_scaled = scaler.transform(latest_data)
            latest_data_pca = pca.transform(latest_data_scaled)
            prediction = model.predict(np.expand_dims(latest_data_pca, axis=-1))
            signal = 'buy' if prediction[0][0] > 0.5 else 'sell'

        data.at[i, 'signal'] = signal

        if signal == 'buy' and position == 0:
            position = balance / data['close'].values[i]
            balance = 0
        elif signal == 'sell' and position > 0:
            balance = position * data['close'].values[i]
            position = 0

        data.at[i, 'position'] = position
        data.at[i, 'strategy_returns'] = (position * data['close'].values[i]) + balance

    data['strategy_cumulative_returns'] = data['strategy_returns'].pct_change().fillna(0).cumsum()
    return data

def plot_backtesting_results(data):
    """Plots backtesting results."""
    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data['strategy_cumulative_returns'], label='Strategy Returns')
    plt.plot(data['timestamp'], data['close'].pct_change().cumsum(), label='Market Returns')
    plt.legend()
    plt.show()

def evaluate_risk_management(symbol, kraken, order):
    """Evaluate risk management rules before executing trades."""
    try:
        balance = kraken.fetch_balance()
        asset_balance = balance['total'].get(symbol.split('/')[0], 0)
        position_size = order['amount']
        max_risk = 0.02 * asset_balance  # Risking only 2% of asset balance

        if position_size > max_risk:
            logging.info(f"Position size {position_size} exceeds max risk {max_risk}. Adjusting position size.")
            order['amount'] = max_risk
        return order
    except Exception as e:
        logging.error(f"Error in risk management: {e}")
        raise

def diversified_trading(client, symbol_list, risk_management_func):
    """Implement diversified trading across multiple assets."""
    loop = asyncio.get_event_loop()
    tasks = [asyncio.ensure_future(trade_symbol(client, symbol, risk_management_func)) for symbol in symbol_list]
    loop.run_until_complete(asyncio.gather(*tasks))

async def trade_symbol(client, symbol, risk_management_func):
    """Trades a single symbol asynchronously."""
    try:
        logging.info(f"Starting trading for {symbol}")
        # Fetch data and preprocess
        data = fetch_ohlcv(symbol, '1h', kraken.parse8601('2021-01-01T00:00:00Z'), 1000, kraken)
        data = check_data_quality(data)
        data = add_features(data)
        data = add_tweet_sentiment_feature(data, 'Bitcoin', 100)

        # Define features and prepare data
        features = [
            'ma50', 'ma200', 'rsi', 'volatility', 'momentum', 'stochastic_k',
            'stochastic_d', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
            'atr', 'returns', 'sentiment', 'day_of_week', 'hour_of_day', 'lag1', 'lag2'
        ]
        X_pca, y, scaler, pca = prepare_data_for_training(data, features)

        # Load model and predict signal
        model = await load_model_async(f'trained_model_{symbol.replace("/", "_")}.pkl')
        signal = predict_signal(model, data, scaler, features, pca)

        # Check balance and execute trade with risk management
        balance, usd_balance = check_balance(symbol, kraken)
        available_balance = usd_balance if 'USD' in symbol else balance
        if available_balance >= 10:
            order = execute_trade(signal, symbol, available_balance, kraken)
            order = risk_management_func(symbol, kraken, order)
            set_trailing_stop_loss_take_profit(symbol, order, kraken)
            logging.info(f"Executed trade order for {symbol}: {order}")
        else:
            logging.info(f"Insufficient funds to execute trade for {symbol}")
    except Exception as e:
        logging.error(f"Error in diversified trading for {symbol}: {e}")

async def load_model_async(model_path):
    """Loads a model asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, load_model, model_path)
