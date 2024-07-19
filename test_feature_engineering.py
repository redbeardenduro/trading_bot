# test_feature_engineering.py

import unittest
from feature_engineering.feature_engineering import add_features, add_tweet_sentiment_feature
import pandas as pd

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # Setup code to create a sample DataFrame for testing
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='h'),
            'close': [i * 10 for i in range(100)],
            'high': [i * 10 + 5 for i in range(100)],
            'low': [i * 10 - 5 for i in range(100)],
        }
        self.df = pd.DataFrame(data)

    def test_add_features(self):
        result = add_features(self.df)
        self.assertIn('ma50', result.columns)
        self.assertIn('rsi', result.columns)

    def test_add_tweet_sentiment_feature(self):
        tweets = ["Tweet about Bitcoin"] * 100  # Mock tweets
        result = add_tweet_sentiment_feature(self.df, tweets, 100)
        self.assertIn('sentiment', result.columns)

if __name__ == '__main__':
    unittest.main()

