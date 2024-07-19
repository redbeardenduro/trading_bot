import os
import json
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import logging

load_dotenv()  # Load environment variables from .env file

def generate_encryption_key():
    """Generates a new Fernet encryption key."""
    return Fernet.generate_key().decode()

def encrypt_api_key(key):
    """Encrypts API key using Fernet encryption."""
    try:
        cipher_suite = Fernet(os.getenv('ENCRYPTION_KEY'))
        encrypted_key = cipher_suite.encrypt(key.encode())
        return encrypted_key.decode()
    except Exception as e:
        logging.error(f"Error encrypting API key: {e}")
        raise

def decrypt_api_key(encrypted_key):
    """Decrypts API key using Fernet encryption."""
    try:
        cipher_suite = Fernet(os.getenv('ENCRYPTION_KEY'))
        decrypted_key = cipher_suite.decrypt(encrypted_key.encode()).decode()
        return decrypted_key
    except Exception as e:
        logging.error(f"Error decrypting API key: {e}")
        raise

def load_config():
    """Loads configuration from file or environment variables and encrypts API keys if necessary."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        if os.path.exists(config_path):
            with open(config_path) as config_file:
                config = json.load(config_file)
        else:
            config = {
                'kraken_api_key': encrypt_api_key(os.getenv('KRAKEN_API_KEY')),
                'kraken_secret_key': encrypt_api_key(os.getenv('KRAKEN_SECRET_KEY')),
                'openai_api_key': encrypt_api_key(os.getenv('OPENAI_API_KEY')),
                'twitter_api_key': encrypt_api_key(os.getenv('TWITTER_API_KEY')),
                'twitter_api_secret_key': encrypt_api_key(os.getenv('TWITTER_API_SECRET_KEY')),
                'twitter_access_token': encrypt_api_key(os.getenv('TWITTER_ACCESS_TOKEN')),
                'twitter_access_token_secret': encrypt_api_key(os.getenv('TWITTER_ACCESS_TOKEN_SECRET')),
                'news_api_key': encrypt_api_key(os.getenv('NEWS_API_KEY'))
            }
            with open(config_path, 'w') as config_file:
                json.dump(config, config_file)
        logging.info("Configuration loaded successfully")
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise
    return config
