import ccxt
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# API keys from the config (replace with your decrypted keys)
kraken_api_key = "AiLIFjIFz89Hm6vO7G70cw4OO8IWwM7jRzmQHWKjZplSHTG1ETb0ymg4"
kraken_secret_key = "ge0iVTu8NqKpqW1bzaMEToJvKNdlPP0Al3Se5T+u7WZdvF2+F4IgYf3O0ar1S04OIwF7tUtLAAxQy3NaOMczPA=="

# Initialize Kraken client
kraken = ccxt.kraken({
    'apiKey': kraken_api_key,
    'secret': kraken_secret_key,
    'enableRateLimit': True,
})

# Test function for Kraken API keys
def test_kraken_api():
    try:
        # Fetch balance to test API key
        balance = kraken.fetch_balance()
        logging.debug(f"Raw API response: {balance}")
        print(f"Raw API response: {balance}")
        
        if not balance.get('total'):
            logging.error("The 'total' field in the balance response is empty or missing.")
        else:
            logging.info(f"Fetched balance: {balance['total']}")
            print(f"Fetched balance: {balance['total']}")
    except Exception as e:
        logging.error(f"Error fetching balance: {e}")
        print(f"Error fetching balance: {e}")

# Run the test
test_kraken_api()

