import ccxt
import pandas as pd
import logging

def fetch_real_time_data(symbol, timeframe, kraken):
    """Fetches real-time OHLCV data from Kraken."""
    try:
        ohlcv = kraken.fetch_ohlcv(symbol, timeframe, limit=1)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Validate data order
        for entry in ohlcv:
            if len(entry) != 6:
                raise ValueError("All measurements must be returned in the order they are measured.")

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

def init_kafka_producer(bootstrap_servers='localhost:9092'):
    """Initializes Kafka producer."""
    from kafka import KafkaProducer
    import json

    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logging.info("Kafka producer initialized successfully")
        return producer
    except Exception as e:
        logging.error(f"Error initializing Kafka producer: {e}")
        return None
