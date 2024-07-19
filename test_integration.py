import unittest
from unittest.mock import patch, mock_open
from main import job
from trading.trading import check_balance, execute_trade, predict_signal
from data_fetch.data_fetch import fetch_ohlcv
from model.model import prepare_data_for_training, build_xgb_model
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, filename='trading_bot.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

class TestIntegration(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data='{"kraken_api_key": "encrypted_key", "kraken_secret_key": "encrypted_secret"}')
    @patch('data_fetch.data_fetch.fetch_ohlcv')
    @patch('trading.trading.check_balance')
    @patch('trading.trading.execute_trade')
    @patch('trading.trading.predict_signal')
    @patch('main.decrypt_api_key')
    @patch('main.load_config')
    def test_job_execution(self, mock_load_config, mock_decrypt_api_key, mock_predict_signal, mock_execute_trade, mock_check_balance, mock_fetch_ohlcv, mock_open):
        logging.info("Starting test_job_execution")

        # Mock responses
        mock_load_config.return_value = {'kraken_api_key': 'encrypted_key', 'kraken_secret_key': 'encrypted_secret'}
        mock_decrypt_api_key.return_value = b'decrypted_key'
        logging.debug("Config and API keys mocked")

        mock_fetch_ohlcv.return_value = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'open': [1]*100,
            'high': [2]*100,
            'low': [0.5]*100,
            'close': [1.5]*100,
            'volume': [100]*100
        })
        logging.debug("fetch_ohlcv mocked data set")

        mock_check_balance.return_value = 1000
        logging.debug("check_balance mock return value set")

        mock_execute_trade.return_value = {'id': 'mock_order_id'}
        logging.debug("execute_trade mock return value set")

        mock_predict_signal.return_value = True
        logging.debug("predict_signal mock return value set")

        # Call the job function
        client = None  # Mocking the client as well, if necessary
        job(client)
        logging.info("job function executed")

        # Assert fetch_ohlcv was called
        mock_fetch_ohlcv.assert_called_once()
        logging.info("fetch_ohlcv call asserted")

        # Assert check_balance was called
        mock_check_balance.assert_called_once()
        logging.info("check_balance call asserted")

        # Assert execute_trade was called
        mock_execute_trade.assert_called_once()
        logging.info("execute_trade call asserted")

        # Assert predict_signal was called
        mock_predict_signal.assert_called_once()
        logging.info("predict_signal call asserted")

if __name__ == '__main__':
    logging.info("Starting integration tests")
    unittest.main()
    logging.info("Integration tests completed")

