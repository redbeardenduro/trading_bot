# test_config.py

from config.config import load_config, decrypt_api_key

if __name__ == "__main__":
    config = load_config()
    print("Decrypted Kraken API Key:", decrypt_api_key(config['kraken_api_key']))
    print("Decrypted Kraken Secret Key:", decrypt_api_key(config['kraken_secret_key']))
