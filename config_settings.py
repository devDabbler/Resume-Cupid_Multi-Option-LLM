import os
from dotenv import load_dotenv
import logging

environment = os.getenv('ENVIRONMENT', 'development')
env_file = '.env.production' if environment == 'production' else '.env.development'
env_path = os.path.join(os.path.dirname(__file__), env_file)

load_dotenv(dotenv_path=env_path)

# Set up logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load API keys
llama_api_key = os.getenv('LLAMA_API_KEY')
if llama_api_key:
    logger.debug(f"Llama API Key (first 5 chars): {llama_api_key[:5]}")
    logger.debug(f"Llama API Key length: {len(llama_api_key)}")
else:
    logger.error("LLAMA_API_KEY not set in environment variables")

# Debugging: Log each variable individually
smtp_server = os.getenv('SMTP_SERVER')
smtp_port = os.getenv('SMTP_PORT')
smtp_username = os.getenv('SMTP_USERNAME')
smtp_password = os.getenv('SMTP_PASSWORD')
from_email = os.getenv('FROM_EMAIL')

if smtp_server is None:
    logger.error("SMTP_SERVER not found in environment variables.")
else:
    logger.debug(f"SMTP_SERVER: {smtp_server}")

if smtp_port is None:
    logger.error("SMTP_PORT not found in environment variables.")
else:
    logger.debug(f"SMTP_PORT: {smtp_port}")

if smtp_username is None:
    logger.error("SMTP_USERNAME not found in environment variables.")
else:
    logger.debug(f"SMTP_USERNAME: {smtp_username}")

if smtp_password is None:
    logger.error("SMTP_PASSWORD not found in environment variables.")
else:
    logger.debug(f"SMTP_PASSWORD: {smtp_password}")

if from_email is None:
    logger.error("FROM_EMAIL not found in environment variables.")
else:
    logger.debug(f"FROM_EMAIL: {from_email}")

if llama_api_key is None:
    logger.error("LLAMA_API_KEY not found in environment variables.")
else:
    logger.debug(f"LLAMA_API_KEY: {llama_api_key}")

if llama_api_key is None:
    logger.error("LLAMA_API_KEY not found in environment variables.")
else:
    logger.debug(f"LLAMA_API_KEY: {llama_api_key}")

# Config class with fallback values
class Config:
    API_URL = os.getenv('API_URL', 'http://localhost:8501')
    BASE_URL = os.getenv('BASE_URL', 'http://localhost:8501')
    SMTP_SERVER = smtp_server if smtp_server else 'default_smtp_server'
    SMTP_PORT = int(smtp_port) if smtp_port else 587
    SMTP_USER = smtp_username if smtp_username else 'default_username'
    SMTP_PASSWORD = smtp_password if smtp_password else 'default_password'
    FROM_EMAIL = from_email if from_email else 'default_email@example.com'
    DB_PATH = os.getenv('SQLITE_DB_PATH', './resume_cupid.db')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG')
    LOG_FILE = os.getenv('LOG_FILE', './logs/app.log')
    BERT_MODEL_NAME = os.getenv('BERT_MODEL_NAME', 'all-MiniLM-L6-v2')
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    LLAMA_API_KEY = llama_api_key if llama_api_key else 'default_llama_api_key'

    @staticmethod
    def get_smtp_config():
        """Return the SMTP configuration as a dictionary."""
        return {
            'server': Config.SMTP_SERVER,
            'port': Config.SMTP_PORT,
            'username': Config.SMTP_USER,
            'password': Config.SMTP_PASSWORD,
            'from_email': Config.FROM_EMAIL
        }

# Log the final SMTP configuration
smtp_config = Config.get_smtp_config()
logger.debug(f"SMTP Configuration: SERVER={smtp_config['server']}, PORT={smtp_config['port']}, USERNAME={smtp_config['username']}, FROM_EMAIL={smtp_config['from_email']}")


