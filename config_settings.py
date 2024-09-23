import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    # App settings
    APP_NAME = "Resume Cupid"
    VERSION = "1.0.0"

    # Environment
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    DEBUG = ENVIRONMENT == 'development'

    # Llama API
    LLAMA_API_KEY = os.getenv('LLAMA_API_KEY')
    if not LLAMA_API_KEY:
        logger.error("LLAMA_API_KEY not set in environment variables")

    # Database
    DB_PATH = os.getenv('SQLITE_DB_PATH', './resume_cupid.db')

    # Streamlit
    BASE_URL = os.getenv('BASE_URL', 'http://localhost:8501')

    # SMTP (for email functionality)
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.example.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME', 'your_username')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', 'your_password')
    FROM_EMAIL = os.getenv('FROM_EMAIL', 'noreply@example.com')

    @classmethod
    def get_smtp_config(cls):
        return {
            'server': cls.SMTP_SERVER,
            'port': cls.SMTP_PORT,
            'username': cls.SMTP_USERNAME,
            'password': cls.SMTP_PASSWORD,
            'from_email': cls.FROM_EMAIL
        }

# Log non-sensitive configuration details
logger.info(f"Environment: {Config.ENVIRONMENT}")
logger.info(f"Debug mode: {Config.DEBUG}")
logger.info(f"Database path: {Config.DB_PATH}")
logger.info(f"Base URL: {Config.BASE_URL}")