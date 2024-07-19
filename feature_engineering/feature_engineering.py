import pandas as pd
import talib as ta
import asyncio
import aiohttp
import logging
from transformers import pipeline as hf_pipeline
import tensorflow as tf
import numpy as np
from kafka import KafkaProducer
import json
from textblob import TextBlob
import pennylane as qml
from concurrent.futures import ThreadPoolExecutor
import requests  # Add this import

# Limit TensorFlow memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

sentiment_model = hf_pipeline('sentiment-analysis', model='distilbert-base-uncased')

# Define a PennyLane device
dev = qml.device('default.qubit', wires=1)

@qml.qnode(dev)
def quantum_circuit(data):
    qml.RY(data, wires=0)
    return qml.expval(qml.PauliZ(0))

def compute_quantum_feature(data):
    """Compute a quantum feature based on the input data."""
    data = float(data)  # Ensure data is a float
    result = quantum_circuit(data)
    return result

def add_quantum_feature(df):
    """Adds a quantum feature to the data."""
    try:
        df['quantum_feature'] = df['close'].apply(lambda x: compute_quantum_feature(x))
    except Exception as e:
        logging.error(f"Error adding quantum feature: {e}")
        df['quantum_feature'] = np.nan
    return df

def init_kafka_producer(bootstrap_servers='localhost:9092'):
    """Initializes Kafka producer."""
    try:
        producer = KafkaProducer(bootstrap_servers=bootstrap_servers, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        logging.info("Kafka producer initialized successfully")
        return producer
    except Exception as e:
        logging.error(f"Error initializing Kafka producer: {e}")
        return None

def add_features(df):
    """Adds technical indicators as features to the data."""
    try:
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)

        features = {
            'ma50': df['close'].rolling(window=50).mean(),
            'ma200': df['close'].rolling(window=200).mean(),
            'rsi': ta.RSI(df['close'].values, timeperiod=14),
            'returns': df['close'].pct_change(),
            'volatility': df['returns'].rolling(window=10).std(),
            'momentum': ta.MOM(df['close'].values, timeperiod=10),
            'stochastic_k': ta.STOCH(df['high'].values, df['low'].values, df['close'].values)[0],
            'stochastic_d': ta.STOCH(df['high'].values, df['low'].values, df['close'].values)[1],
            'macd': ta.MACD(df['close'].values)[0],
            'macd_signal': ta.MACD(df['close'].values)[1],
            'bollinger_upper': ta.BBANDS(df['close'].values, timeperiod=20)[0],
            'bollinger_middle': ta.BBANDS(df['close'].values, timeperiod=20)[1],
            'bollinger_lower': ta.BBANDS(df['close'].values, timeperiod=20)[2],
            'atr': ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14),
            'day_of_week': df['timestamp'].dt.dayofweek,
            'hour_of_day': df['timestamp'].dt.hour,
            'lag1': df['close'].shift(1),
            'lag2': df['close'].shift(2),
            'lag7': df['close'].shift(7),
            'lag30': df['close'].shift(30),
            'ema50': ta.EMA(df['close'], timeperiod=50),
            'ema200': ta.EMA(df['close'], timeperiod=200),
            'adx': ta.ADX(df['high'], df['low'], df['close'], timeperiod=14),
            'cci': ta.CCI(df['high'], df['low'], df['close'], timeperiod=14),
            'obv': ta.OBV(df['close'], df['volume'])
        }

        feature_df = pd.DataFrame(features)
        df = pd.concat([df, feature_df], axis=1)
        df = df.dropna()

    except Exception as e:
        logging.error(f"Error adding features: {e}")
    return df

def fetch_sentiment_sync(close_prices):
    """Synchronous wrapper for fetch_sentiment."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(fetch_sentiment(close_prices))

async def fetch_sentiment(close_prices):
    """Batch process sentiment analysis with retries and exponential backoff."""
    sentiments = []
    batch_size = 10
    async with aiohttp.ClientSession() as session:
        futures = []
        for i in range(0, len(close_prices), batch_size):
            batch = close_prices[i:i + batch_size]
            futures.append(asyncio.ensure_future(process_batch(session, batch)))
        for future in asyncio.as_completed(futures):
            try:
                sentiments.extend(await future)
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
    return sentiments

async def process_batch(session, batch):
    """Process batch of sentiment analysis requests with retries and exponential backoff."""
    responses = []
    for price in batch:
        attempt = 0
        while attempt < 5:
            try:
                sentiment = sentiment_model(price)
                responses.append(sentiment[0]['label'])
                break
            except Exception as e:
                attempt += 1
                logging.error(f"Attempt {attempt}: Error processing price {price} - {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    return responses

def predict_signal(df):
    """Predict buy/sell/hold signal based on engineered features."""
    try:
        if df['ma50'].iloc[-1] > df['ma200'].iloc[-1] and df['rsi'].iloc[-1] < 30:
            return 'buy'
        elif df['ma50'].iloc[-1] < df['ma200'].iloc[-1] and df['rsi'].iloc[-1] > 70:
            return 'sell'
        else:
            return 'hold'
    except Exception as e:
        logging.error(f"Error predicting signal: {e}")
        return 'hold'

def sentiment_analysis_function(tweet):
    # Mock sentiment analysis function for testing
    return 0.5  # Dummy sentiment score

def add_tweet_sentiment_feature(df, keyword, num_tweets):
    try:
        sentiment_scores = [sentiment_analysis_function(str(tweet)) for tweet in range(num_tweets)]
        
        # Ensure the length of sentiment_scores matches the length of df
        if len(sentiment_scores) < len(df):
            sentiment_scores = np.pad(sentiment_scores, (0, len(df) - len(sentiment_scores)), 'constant')
        else:
            sentiment_scores = sentiment_scores[:len(df)]
        
        df['sentiment'] = sentiment_scores
    except Exception as e:
        logging.error(f"Error adding tweet sentiment feature: {e}")
        df['sentiment'] = np.nan
    return df

def add_news_sentiment_feature(data, keywords, news_api_key):
    """Adds a news sentiment feature to the data."""
    try:
        sentiment = np.zeros(len(data))
        for keyword in keywords:
            news = fetch_news(keyword, news_api_key)
            keyword_sentiment = []
            for article in news:
                blob = TextBlob(article)
                keyword_sentiment.append(blob.sentiment.polarity)
            
            # Ensure the length of sentiment matches the length of data
            if len(keyword_sentiment) < len(data):
                keyword_sentiment = np.pad(keyword_sentiment, (0, len(data) - len(keyword_sentiment)), 'constant', constant_values=np.nan)
            else:
                keyword_sentiment = keyword_sentiment[:len(data)]
            
            sentiment += keyword_sentiment
        
        data['news_sentiment'] = sentiment / len(keywords)
    except Exception as e:
        logging.error(f"Error adding news sentiment feature: {e}")
        data['news_sentiment'] = np.nan
    return data

def fetch_news(keyword, news_api_key):
    """Fetches news articles related to the keyword."""
    url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey={news_api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    news = [article['title'] for article in articles]
    return news

# Function to stream data to Kafka
def stream_data_to_kafka(df, producer, topic):
    """Streams processed data to Kafka."""
    try:
        producer.send(topic, df.to_dict(orient='records'))
        logging.info(f"Sent data to Kafka topic {topic}")
    except Exception as e:
        logging.error(f"Error sending data to Kafka: {e}")
