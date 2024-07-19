import logging
import pyspark
from pyspark.sql import SparkSession
from distributed import Client
from kafka import KafkaProducer
import json
import ccxt
import pandas as pd
from textblob import TextBlob
import pennylane as qml
from pennylane import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import talib as ta
import requests

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize PennyLane device
try:
    dev = qml.device('default.qubit', wires=1)
except Exception as e:
    logger.error(f"Error initializing PennyLane device: {e}")
    dev = None

@qml.qnode(dev)
def quantum_circuit(data, params):
    qml.RY(data, wires=0)
    return qml.expval(qml.PauliZ(0))

@lru_cache(maxsize=128)
def cached_quantum_circuit(data, param0):
    try:
        return quantum_circuit(data, np.array([param0]))
    except Exception as e:
        logger.error(f"Error in cached_quantum_circuit: {e}")
        return None

def compute_quantum_feature(data):
    try:
        param0 = 0.1  # Example parameter
        data_float = float(data)  # Ensure data is a float
        result = cached_quantum_circuit(data_float, param0)
        if result is None:
            raise ValueError("Quantum circuit computation failed.")
        return float(result)
    except Exception as e:
        logger.error(f"Error computing quantum feature: {e}")
        return None

def parallel_process(data_list, function, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(function, data_list))
    return results

def init_kafka_producer(bootstrap_servers='localhost:9092'):
    """Initializes Kafka producer."""
    try:
        producer = KafkaProducer(bootstrap_servers=bootstrap_servers, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        logging.info("Kafka producer initialized successfully")
        return producer
    except Exception as e:
        logging.error(f"Error initializing Kafka producer: {e}")
        return None

def fetch_ohlcv(symbol, timeframe, since, limit, kraken, producer=None, topic=None):
    """Fetches OHLCV data from Kraken."""
    try:
        logger.info(f"Fetching OHLCV data for {symbol}")
        ohlcv = kraken.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Validate data order
        for entry in ohlcv:
            if len(entry) != 6:
                raise ValueError("All measurements must be returned in the order they are measured.")
        
        # Add quantum feature
        df = add_quantum_feature(df)
        
        # Add additional features
        df = add_features(df)
        
        # Send data to Kafka if producer and topic are provided
        if producer and topic:
            producer.send(topic, df.to_dict(orient='records'))
            logging.info(f"Sent OHLCV data for {symbol} to Kafka topic {topic}")
        
        logger.debug(f"Fetched data for {symbol}: {df}")
        return df
    except ccxt.BaseError as e:
        logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
    except ValueError as e:
        logger.error(e)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
    return pd.DataFrame()

def fetch_ohlcv_parallel(symbols, timeframe, since, limit, kraken, producer=None, topic=None):
    """Fetches OHLCV data for multiple symbols in parallel."""
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_ohlcv, symbol, timeframe, since, limit, kraken, producer, topic) for symbol in symbols]
        results = [future.result() for future in as_completed(futures)]
    return results

def fetch_real_time_data(symbol, timeframe, kraken):
    """Fetches real-time OHLCV data from Kraken."""
    try:
        ohlcv = kraken.fetch_ohlcv(symbol, timeframe, limit=1)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"Error fetching real-time data for {symbol}: {e}")
        return pd.DataFrame()

def check_data_quality(df):
    """Checks and cleans the data for missing values and duplicates."""
    try:
        if df.isnull().values.any():
            logging.warning("Data contains missing values. Filling missing values with the previous value.")
            df.fillna(method='ffill', inplace=True)
        
        if df['timestamp'].duplicated().any():
            logging.warning("Data contains duplicated timestamps. Removing duplicates.")
            df = df.drop_duplicates(subset='timestamp')

        return df
    except Exception as e:
        logging.error(f"Error checking data quality: {e}")
        return pd.DataFrame()

def fetch_news(keyword, api_key):
    """Fetches news articles related to the keyword using NewsAPI."""
    url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        news = [article['title'] + " " + article['description'] for article in articles if article['title'] and article['description']]
        return news
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching news: {e}")
        return []

def add_news_sentiment_feature(data, keyword, api_key):
    """Adds a news sentiment feature to the data."""
    news = fetch_news(keyword, api_key)
    sentiment = []
    for article in news:
        blob = TextBlob(article)
        sentiment.append(blob.sentiment.polarity)
    data['news_sentiment'] = pd.Series(sentiment)
    return data

def add_quantum_feature(df):
    """Adds a quantum feature to the data."""
    try:
        df['quantum_feature'] = parallel_process(df['close'].tolist(), compute_quantum_feature)
        return df
    except Exception as e:
        logger.error(f"Error adding quantum feature: {e}")
        df['quantum_feature'] = None
        return df

def add_features(df):
    """Adds technical indicators as features to the data."""
    try:
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)

        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        df['rsi'] = ta.RSI(df['close'].values, timeperiod=14)
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=10).std()
        df['momentum'] = ta.MOM(df['close'].values, timeperiod=10)
        df[['stochastic_k', 'stochastic_d']] = pd.DataFrame(ta.STOCH(df['high'].values, df['low'].values, df['close'].values)).T
        df[['macd', 'macd_signal', '_']] = pd.DataFrame(ta.MACD(df['close'].values)).T
        df[['bollinger_upper', 'bollinger_middle', 'bollinger_lower']] = pd.DataFrame(ta.BBANDS(df['close'].values, timeperiod=20)).T
        df['atr'] = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['lag1'] = df['close'].shift(1)
        df['lag2'] = df['close'].shift(2)
        df['lag7'] = df['close'].shift(7)
        df['lag30'] = df['close'].shift(30)

        # Adding additional features
        df['ema50'] = ta.EMA(df['close'], timeperiod=50)
        df['ema200'] = ta.EMA(df['close'], timeperiod=200)
        df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['obv'] = ta.OBV(df['close'], df['volume'])

        return df.dropna()
    except Exception as e:
        logging.error(f"Error adding features to data: {e}")
        return df

if __name__ == '__main__':
    # Initialize Spark session
    spark = SparkSession.builder.appName("CryptoTrading").getOrCreate()

    # Initialize Dask client
    try:
        client = Client()
        logging.info("Dask client initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing Dask client: {e}")
