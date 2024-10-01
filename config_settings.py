import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
from typing import Dict, Any

# Load environment-specific .env file
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
env_file = '.env.production' if ENVIRONMENT == 'production' else '.env.development'
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

# Set watchdog logging level to INFO
watchdog_logger = logging.getLogger('watchdog')
watchdog_logger.setLevel(logging.INFO)

class Config:
    API_URL: str = os.getenv('API_URL', 'http://localhost:8501')
    BASE_URL: str = os.getenv('BASE_URL', 'http://localhost:8501')
    SMTP_SERVER: str = os.getenv('SMTP_SERVER', 'default_smtp_server')
    SMTP_PORT: int = int(os.getenv('SMTP_PORT', 587))
    SMTP_USERNAME: str = os.getenv('SMTP_USERNAME', 'default_username')
    SMTP_PASSWORD: str = os.getenv('SMTP_PASSWORD', 'default_password')
    FROM_EMAIL: str = os.getenv('FROM_EMAIL', 'default_email@example.com')
    DB_PATH: str = os.getenv('SQLITE_DB_PATH', './resume_cupid.db')
    ENVIRONMENT: str = ENVIRONMENT
    LLAMA_API_KEY: str = os.getenv('LLAMA_API_KEY', 'default_llama_api_key')
    
    # New configurations for authentication
    COOKIE_NAME: str = os.getenv('COOKIE_NAME', 'resume_cupid_auth')
    COOKIE_KEY: str = os.getenv('COOKIE_KEY', 'default_cookie_key')
    COOKIE_EXPIRY_DAYS: int = int(os.getenv('COOKIE_EXPIRY_DAYS', 30))

    @classmethod
    def get_smtp_config(cls) -> Dict[str, Any]:
        config = {
            'server': cls.SMTP_SERVER,
            'port': cls.SMTP_PORT,
            'username': cls.SMTP_USERNAME,
            'password': cls.SMTP_PASSWORD,
            'from_email': cls.FROM_EMAIL
        }
        
        # Check for default values
        default_values = [
            ('SMTP_SERVER', 'default_smtp_server'),
            ('SMTP_USERNAME', 'default_username'),
            ('SMTP_PASSWORD', 'default_password'),
            ('FROM_EMAIL', 'default_email@example.com')
        ]
        
        for env_var, default_value in default_values:
            if getattr(cls, env_var) == default_value:
                logger.warning(f"{env_var} is set to its default value. Please update it in the .env file.")
        
        return config

    @classmethod
    def validate_config(cls) -> None:
        # Log the SMTP configuration
        smtp_config = cls.get_smtp_config()
        logger.info(f"SMTP Configuration: SERVER={smtp_config['server']}, PORT={smtp_config['port']}, USERNAME={smtp_config['username']}, FROM_EMAIL={smtp_config['from_email']}")

        # Log the environment
        logger.info(f"Current environment: {cls.ENVIRONMENT}")

        # Check database path
        if not os.path.exists(cls.DB_PATH):
            logger.warning(f"Database file not found at {cls.DB_PATH}. It will be created when the application runs.")

        # Validate LLAMA API key
        if cls.LLAMA_API_KEY == 'default_llama_api_key':
            logger.warning("LLAMA_API_KEY is set to its default value. Please update it in the .env file.")

        # Additional environment checks
        if cls.ENVIRONMENT == 'production':
            if 'localhost' in cls.BASE_URL:
                logger.warning("BASE_URL contains 'localhost' in production environment. Please update it to the actual domain.")
            if not cls.SMTP_SERVER.endswith(('.com', '.net', '.org', '.edu')):
                logger.warning("SMTP_SERVER might not be correctly set for production. Please verify the SMTP server address.")

        # Validate authentication settings
        if cls.COOKIE_KEY == 'default_cookie_key':
            logger.warning("COOKIE_KEY is set to its default value. Please update it in the .env file for security.")

# Run validation on import
Config.validate_config()

# Example of setting up watchdog observer
if __name__ == "__main__":
    event_handler = LoggingEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()