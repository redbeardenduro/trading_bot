import unittest
import pandas as pd
from feature_engineering.feature_engineering import add_features, predict_signal

class TestFeatureEngineering(unittest.TestCase):

    def test_add_features(self):
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=300, freq='h'),  # Increased periods to 300
            'close': [i for i in range(300)],
            'high': [i + 1 for i in range(300)],
            'low': [i - 1 for i in range(300)]
        }
        df = pd.DataFrame(data)
        result = add_features(df)
        self.assertIn('ma50', result.columns)
        self.assertIn('rsi', result.columns)
        self.assertIn('momentum', result.columns)
        self.assertIn('bollinger_upper', result.columns)
        self.assertIn('sentiment', result.columns)

    def test_predict_signal(self):
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=300, freq='h'),  # Increased periods to 300
            'close': [i for i in range(300)],
            'high': [i + 1 for i in range(300)],
            'low': [i - 1 for i in range(300)]
        }
        df = pd.DataFrame(data)
        df = add_features(df)
        signal = predict_signal(df)
        self.assertIn(signal, ['buy', 'sell', 'hold'])

if __name__ == '__main__':
    unittest.main()

