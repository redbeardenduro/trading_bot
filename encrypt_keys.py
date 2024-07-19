# encrypt_keys.py

from cryptography.fernet import Fernet
import os

def generate_encryption_key():
    """Generates a new Fernet encryption key."""
    return Fernet.generate_key()

def encrypt_api_key(key, encryption_key):
    """Encrypts API key using Fernet encryption."""
    cipher_suite = Fernet(encryption_key)
    encrypted_key = cipher_suite.encrypt(key.encode())
    return encrypted_key

# Generate encryption key
encryption_key = generate_encryption_key()
print(f"Encryption Key: {encryption_key.decode()}")

# Replace these with your actual API keys
api_keys = {
    'kraken_api_key': os.getenv('KRAKEN_API_KEY'),
    'kraken_secret_key': os.getenv('KRAKEN_SECRET_KEY'),
    'openai_api_key': os.getenv('OPENAI_API_KEY'),
    'twitter_api_key': os.getenv('TWITTER_API_KEY'),
    'twitter_api_secret_key': os.getenv('TWITTER_API_SECRET_KEY'),
    'twitter_access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
    'twitter_access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
    'news_api_key': os.getenv('NEWS_API_KEY')
}

# Encrypt API keys
encrypted_keys = {k: encrypt_api_key(v, encryption_key).decode() for k, v in api_keys.items()}

# Print encrypted keys
for k, v in encrypted_keys.items():
    print(f"{k}: {v}")
