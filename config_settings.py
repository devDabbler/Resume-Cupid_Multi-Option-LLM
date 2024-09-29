import os
from dotenv import load_dotenv
import logging

# Load environment-specific .env file
environment = os.getenv('ENVIRONMENT', 'development')
env_file = '.env.production' if environment == 'production' else '.env.development'
env_path = os.path.join(os.path.dirname(__file__), env_file)
load_dotenv(dotenv_path=env_path)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    API_URL = os.getenv('API_URL', 'http://localhost:8501')
    BASE_URL = os.getenv('BASE_URL', 'http://localhost:8501')
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'default_smtp_server')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME', 'default_username')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', 'default_password')
    FROM_EMAIL = os.getenv('FROM_EMAIL', 'default_email@example.com')
    DB_PATH = os.getenv('SQLITE_DB_PATH', './resume_cupid.db')
    ENVIRONMENT = environment
    LLAMA_API_KEY = os.getenv('LLAMA_API_KEY', 'default_llama_api_key')

    @staticmethod
    def get_smtp_config():
        return {
            'server': Config.SMTP_SERVER,
            'port': Config.SMTP_PORT,
            'username': Config.SMTP_USERNAME,
            'password': Config.SMTP_PASSWORD,
            'from_email': Config.FROM_EMAIL
        }

# Log the SMTP configuration
smtp_config = Config.get_smtp_config()
logger.debug(f"SMTP Configuration: SERVER={smtp_config['server']}, PORT={smtp_config['port']}, USERNAME={smtp_config['username']}, FROM_EMAIL={smtp_config['from_email']}")