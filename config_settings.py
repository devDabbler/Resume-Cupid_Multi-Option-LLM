import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

# Load environment-specific .env file
environment = os.getenv('ENVIRONMENT', 'development')
env_file = '.env.production' if environment == 'production' else '.env.development'
env_path = os.path.join(os.path.dirname(__file__), env_file)
load_dotenv(dotenv_path=env_path)

# Set up logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = 'app.log'

# Create a rotating file handler
file_handler = RotatingFileHandler(log_file, mode='a', maxBytes=5*1024*1024, backupCount=2, encoding=None, delay=0)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Create a stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO)

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler]
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
        config = {
            'server': Config.SMTP_SERVER,
            'port': Config.SMTP_PORT,
            'username': Config.SMTP_USERNAME,
            'password': Config.SMTP_PASSWORD,
            'from_email': Config.FROM_EMAIL
        }
        
        # Check for default values
        default_values = [
            ('SMTP_SERVER', 'default_smtp_server'),
            ('SMTP_USERNAME', 'default_username'),
            ('SMTP_PASSWORD', 'default_password'),
            ('FROM_EMAIL', 'default_email@example.com')
        ]
        
        for env_var, default_value in default_values:
            if getattr(Config, env_var) == default_value:
                logger.warning(f"{env_var} is set to its default value. Please update it in the .env file.")
        
        return config

# Log the SMTP configuration
smtp_config = Config.get_smtp_config()
logger.info(f"SMTP Configuration: SERVER={smtp_config['server']}, PORT={smtp_config['port']}, USERNAME={smtp_config['username']}, FROM_EMAIL={smtp_config['from_email']}")

# Log the environment
logger.info(f"Current environment: {Config.ENVIRONMENT}")

# Check database path
if not os.path.exists(Config.DB_PATH):
    logger.warning(f"Database file not found at {Config.DB_PATH}. It will be created when the application runs.")

# Validate LLAMA API key
if Config.LLAMA_API_KEY == 'default_llama_api_key':
    logger.warning("LLAMA_API_KEY is set to its default value. Please update it in the .env file.")

# Additional environment checks
if Config.ENVIRONMENT == 'production':
    if 'localhost' in Config.BASE_URL:
        logger.warning("BASE_URL contains 'localhost' in production environment. Please update it to the actual domain.")
    if not Config.SMTP_SERVER.endswith(('.com', '.net', '.org', '.edu')):
        logger.warning("SMTP_SERVER might not be correctly set for production. Please verify the SMTP server address.")