import sys
import os
import unittest
import logging
import pandas as pd

# Set the module path to include the project's utils directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the modules
import utils
import config
import database  # Adjusting this import based on the correct path
from feature_engineering import feature_engineering
from trading import trading
from model import model

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Test Utilities
class TestUtils(unittest.TestCase):
    def test_add_numbers(self):
        self.assertEqual(utils.add_numbers(2, 3), 5)
        self.assertEqual(utils.add_numbers(-1, 1), 0)
        self.assertEqual(utils.add_numbers(0, 0), 0)

    def test_init_cleanup_dask_client(self):
        client = utils.init_dask_client()
        self.assertIsNotNone(client)
        utils.cleanup_dask_client(client)
        self.assertTrue(client.status == 'closed')

    def test_setup_logging(self):
        utils.setup_logging()
        logger = logging.getLogger()
        self.assertEqual(logger.level, logging.DEBUG)

# Test Feature Engineering
class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'close': [i * 10 for i in range(100)],
            'high': [i * 10 + 5 for i in range(100)],
            'low': [i * 10 - 5 for i in range(100)],
        }
        self.df = pd.DataFrame(data)

    def test_add_features(self):
        result = feature_engineering.add_features(self.df)
        self.assertIn('ma50', result.columns)
        self.assertIn('rsi', result.columns)

# Test Config
class TestConfig(unittest.TestCase):
    def test_generate_encryption_key(self):
        key = config.generate_encryption_key()
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 44)  # Fernet keys are 32 bytes base64-encoded

    def test_encrypt_decrypt_api_key(self):
        original_key = "test_api_key"
        encrypted_key = config.encrypt_api_key(original_key)
        decrypted_key = config.decrypt_api_key(encrypted_key)
        self.assertEqual(original_key, decrypted_key)

# Test Database
class TestDatabase(unittest.TestCase):
    def test_save_get_trades(self):
        trade = database.Trade(timestamp=pd.Timestamp.now(), symbol="BTC/USD", order_type="buy", amount=1.0, price=50000.0, status="completed")
        database.save_trade(trade)
        trades = database.get_trades()
        self.assertGreater(len(trades), 0)

# Test Trading
class TestTrading(unittest.TestCase):
    def test_check_balance(self):
        # This requires mocking the Kraken API response
        pass

    def test_execute_trade(self):
        # This requires mocking the Kraken API response
        pass

# Test Model
class TestModel(unittest.TestCase):
    def test_prepare_data_for_training(self):
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'close': [i * 10 for i in range(100)],
            'high': [i * 10 + 5 for i in range(100)],
            'low': [i * 10 - 5 for i in range(100)],
            'ma50': [i for i in range(100)],
            'rsi': [i for i in range(100)],
            'volatility': [i for i in range(100)],
            'momentum': [i for i in range(100)],
            'stochastic_k': [i for i in range(100)],
            'stochastic_d': [i for i in range(100)],
            'macd': [i for i in range(100)],
            'macd_signal': [i for i in range(100)],
            'bollinger_upper': [i for i in range(100)],
            'bollinger_lower': [i for i in range(100)],
            'atr': [i for i in range(100)],
            'returns': [i for i in range(100)],
            'sentiment': [i for i in range(100)],
            'day_of_week': [i for i in range(100)],
            'hour_of_day': [i for i in range(100)],
            'lag1': [i for i in range(100)],
            'lag2': [i for i in range(100)]
        }
        df = pd.DataFrame(data)
        features = [
            'ma50', 'ma200', 'rsi', 'volatility', 'momentum', 'stochastic_k',
            'stochastic_d', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
            'atr', 'returns', 'sentiment', 'day_of_week', 'hour_of_day', 'lag1', 'lag2'
        ]
        X_pca, y, scaler, pca = model.prepare_data_for_training(df, features)
        self.assertEqual(X_pca.shape[1], 10)  # PCA components
        self.assertEqual(len(y), 100)  # Number of samples

# Run the tests
if __name__ == '__main__':
    unittest.main()

