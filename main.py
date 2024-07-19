import os
import sys
import tensorflow as tf
import unittest
import schedule
import time
import logging
import ccxt
import signal
from distributed import Client
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from optuna.integration import TFKerasPruningCallback
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from pennylane import numpy as np

# Suppress TensorFlow/CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import load_config, decrypt_api_key
from data_fetch import fetch_ohlcv, check_data_quality, init_kafka_producer, fetch_ohlcv_parallel
from feature_engineering import add_features, add_tweet_sentiment_feature, add_news_sentiment_feature, add_quantum_feature
from model.advanced_models import prepare_data_for_training, tune_lstm_model, build_xgb_model, save_model, predict_signal
from trading import execute_trade, set_trailing_stop_loss_take_profit, check_balance, diversified_trading, is_fund_sufficient
from utils import setup_logging, send_error_notification, init_kafka_producer, init_kafka_consumer, train_model, train_step

# Initialize sentiment model
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_model = TFDistilBertForSequenceClassification.from_pretrained(model_name)

# Minimum order sizes for selected cryptocurrencies on Kraken
MIN_ORDER_SIZES = {
    'BTC/USD': 0.0001,
    'ETH/USD': 0.01,
    'XRP/USD': 10,
}

# List of trading pairs
TRADING_PAIRS = ['BTC/USD', 'ETH/USD', 'XRP/USD']

# Initialize Spark session
spark = SparkSession.builder.appName("CryptoTrading").getOrCreate()
ssc = StreamingContext(spark.sparkContext, 1)  # Batch interval of 1 second

# Load and decrypt API keys
config = load_config()
kraken_api_key = decrypt_api_key(config['kraken_api_key'])
kraken_secret_key = decrypt_api_key(config['kraken_secret_key'])
openai_api_key = decrypt_api_key(config['openai_api_key'])
twitter_api_key = decrypt_api_key(config['twitter_api_key'])
twitter_api_secret_key = decrypt_api_key(config['twitter_api_secret_key'])
twitter_access_token = decrypt_api_key(config['twitter_access_token'])
twitter_access_token_secret = decrypt_api_key(config['twitter_access_token_secret'])
news_api_key = decrypt_api_key(config['news_api_key'])  # Add this line

# Initialize Kraken API
kraken = ccxt.kraken({
    'apiKey': kraken_api_key,
    'secret': kraken_secret_key,
    'enableRateLimit': True,
})

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(LSTM(units=32))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy')
    return model

# Function to initialize Kafka stream
def initialize_kafka_stream(topic):
    kafka_stream = init_kafka_consumer(topic)
    return kafka_stream

# Function for monitoring and adjusting strategy
def monitor_and_adjust_strategy():
    # Implement monitoring and adjustment logic here
    pass

# Main job function that performs the trading tasks
def job(client):
    logging.info("Starting job execution")
    successful_pairs = []
    try:
        logging.info("Loading configuration")
        logging.debug(f"Config loaded: {config}")

        logging.info("Decrypting API keys")
        try:
            logging.debug("API keys decrypted successfully")
        except Exception as e:
            logging.error(f"Error decrypting API keys: {e}")
            raise

        logging.info("Initializing Kraken API")
        try:
            logging.debug("Kraken API initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing Kraken API: {e}")
            raise

        for symbol in TRADING_PAIRS:
            logging.info(f"Fetching OHLCV data for {symbol}")
            try:
                data = fetch_ohlcv(symbol, '1h', kraken.parse8601('2021-01-01T00:00:00Z'), 1000, kraken)
                if data.empty:
                    logging.error(f"No data fetched for {symbol}. Skipping this symbol.")
                    continue
                logging.debug(f"Fetched data for {symbol}: {data.head()}")
            except Exception as e:
                logging.error(f"Error fetching OHLCV data for {symbol}: {e}")
                continue

            logging.info("Checking data quality")
            try:
                data = check_data_quality(data)
                logging.debug(f"Data after quality check for {symbol}: {data.head()}")
            except Exception as e:
                logging.error(f"Error checking data quality for {symbol}: {e}")
                continue

            logging.info("Adding features to data")
            try:
                if hasattr(data, 'compute'):
                    data = data.compute()  # If it's a Dask DataFrame, compute it
                data = add_features(data)
                logging.debug(f"Data after adding features for {symbol}: {data.head()}")
            except Exception as e:
                logging.error(f"Error adding features to data for {symbol}: {e}")
                continue

            logging.info("Adding news sentiment feature")
            try:
                keywords = ['Bitcoin', 'Ethereum', 'XRP']
                data = add_news_sentiment_feature(data, keywords, news_api_key)  # Pass the news API key and keywords
                logging.debug(f"Data after adding news sentiment feature for {symbol}: {data.head()}")
            except Exception as e:
                logging.error(f"Error adding news sentiment feature for {symbol}: {e}")
                continue

            logging.info("Adding quantum feature")
            try:
                data = add_quantum_feature(data)
                logging.debug(f"Data after adding quantum feature for {symbol}: {data.head()}")
            except Exception as e:
                logging.error(f"Error adding quantum feature for {symbol}: {e}")
                continue

            logging.info("Adding tweet sentiment feature")
            try:
                data = add_tweet_sentiment_feature(data, 'Bitcoin', 100)
                logging.debug(f"Data after adding tweet sentiment feature for {symbol}: {data.head()}")
            except Exception as e:
                logging.error(f"Error adding tweet sentiment feature for {symbol}: {e}")
                continue

            # Define features for model training
            features = [
                'ma50', 'ma200', 'rsi', 'volatility', 'momentum', 'stochastic_k',
                'stochastic_d', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
                'atr', 'returns', 'sentiment', 'day_of_week', 'hour_of_day', 'lag1', 'lag2',
                'quantum_feature'  # Include quantum feature
            ]

            logging.info("Preparing data for training")
            try:
                X_pca, y, scaler, pca = prepare_data_for_training(data, features)
                logging.debug(f"Prepared data for training for {symbol}: X_pca shape={X_pca.shape}, y shape={y.shape}")
            except Exception as e:
                logging.error(f"Error preparing data for training for {symbol}: {e}")
                continue

            logging.info("Training model")
            try:
                # Hyperparameter tuning with Optuna
                def objective(trial):
                    model = create_lstm_model((X_pca.shape[1], 1))
                    batch_size = trial.suggest_categorical('batch_size', [16, 32])
                    epochs = trial.suggest_int('epochs', 10, 30)
                    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
                    
                    model.compile(optimizer=optimizer, loss='binary_crossentropy')
                    
                    early_stopping = EarlyStopping(monitor='loss', patience=3)
                    history = model.fit(X_pca, y, batch_size=batch_size, epochs=epochs,
                                        callbacks=[early_stopping, TFKerasPruningCallback(trial, 'loss')],
                                        validation_split=0.2, verbose=0)
                    
                    return history.history['val_loss'][-1]

                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=20)  # Reduce the number of trials for faster tuning

                best_params = study.best_params
                logging.info(f"Best hyperparameters: {best_params}")

                # Train final model with best parameters
                final_model = create_lstm_model((X_pca.shape[1], 1))
                final_model.compile(optimizer=best_params['optimizer'], loss='binary_crossentropy')
                final_model.fit(X_pca, y, batch_size=best_params['batch_size'], epochs=best_params['epochs'],
                                callbacks=[EarlyStopping(monitor='loss', patience=3)], validation_split=0.2, verbose=1)

                best_model = final_model
            except Exception as e:
                logging.error(f"Error training model for {symbol}: {e}")
                continue

            logging.info("Saving the trained model")
            try:
                save_model(best_model, f'trained_model_{symbol.replace("/", "_")}.pkl')
            except Exception as e:
                logging.error(f"Error saving the trained model for {symbol}: {e}")
                continue

            logging.info("Predicting trading signal")
            try:
                signal = predict_signal(best_model, data, scaler, features, pca)
                logging.debug(f"Predicted trading signal for {symbol}: {signal}")
            except Exception as e:
                logging.error(f"Error predicting trading signal for {symbol}: {e}")
                continue

            logging.info("Checking balance and executing trade")
            balance, usd_balance = check_balance(symbol, kraken)
            available_balance = usd_balance if 'USD' in symbol else balance
            logging.debug(f"Available balance for {symbol}: {available_balance}")

            if available_balance >= MIN_ORDER_SIZES[symbol]:
                try:
                    if is_fund_sufficient(available_balance, MIN_ORDER_SIZES[symbol]):
                        order = execute_trade(signal, symbol, available_balance, kraken)
                        set_trailing_stop_loss_take_profit(symbol, order, kraken=kraken)
                        logging.info(f"Executed trade order for {symbol}: {order}")
                        successful_pairs.append(symbol)
                    else:
                        logging.info(f"Insufficient funds to execute trade for {symbol}")
                except Exception as e:
                    logging.error(f"Error executing trade for {symbol}: {e}")
                    continue
            else:
                logging.info(f'Insufficient funds to execute trade for {symbol}')
    except Exception as e:
        logging.error(f"Error in job execution: {e}")
        send_error_notification("Trading Bot Error", str(e), 'your_email@example.com')
    finally:
        logging.info("Job execution finished")
        logging.info(f"Successful trading pairs: {successful_pairs}")

def stop_job(signum, frame):
    logging.info("Stopping all scheduled jobs")
    schedule.clear()  # Clears all scheduled jobs

# Main execution block
if __name__ == '__main__':
    setup_logging()

    logging.info("Initializing Dask client")
    try:
        client = Client(processes=False)  # Initialize Dask client without specifying the port and without starting additional processes
        logging.info("Dask client initialized")

        # Run unit tests
        logging.info("Running unit tests")
        unittest.main(module='test_trading_bot', exit=False)

        # Schedule the job to run every 10 minutes
        logging.info("Scheduling jobs")
        schedule.every(10).minutes.do(job, client)
        logging.debug("Job has been scheduled to run every 10 minutes")

        # Manually trigger the job once for debugging
        logging.info("Manually triggering the job for debugging")
        job(client)  # Just call the function without assigning its result

        # Register signal handlers
        signal.signal(signal.SIGTERM, stop_job)
        signal.signal(signal.SIGINT, stop_job)

        logging.info("Entering scheduler loop")
        # Run the scheduler
        while True:
            schedule.run_pending()
            logging.debug("Scheduler is running")
            time.sleep(1)
    except Exception as e:
        logging.error(f"Failed to initialize Dask client or run the scheduler: {e}")
        send_error_notification("Initialization Error", str(e), 'your_email@example.com')
    finally:
        if 'client' in locals():
            logging.info("Closing Dask client")
            client.close()
