import os
import sys
import logging
from dotenv import load_dotenv
import streamlit as st
from main_app import main_app
from auth import auth_main, handle_password_reset, login_page, verify_email
from database import init_db, debug_db_contents

# Streamlit configuration
st.set_page_config(page_title="Resume Cupid", page_icon="💘", layout="centered")

# Logging configuration
def setup_logging():
    log_directory = "/home/rezcupid2024/Resume_Cupid_Multi_LLM/logs"
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, "app.log")

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)

logger = setup_logging()
logger.info("Application started")

# Environment configuration
def load_environment():
    environment = os.getenv('ENVIRONMENT', 'development')
    env_file = '.env.production' if environment == 'production' else '.env.development'
    env_path = os.path.join(os.path.dirname(__file__), env_file)

    try:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    except Exception as e:
        logger.error(f"Failed to load {env_path} file: {e}")

    return environment

ENVIRONMENT = load_environment()

# Configuration import and logging
try:
    from config_settings import Config

    logger.debug(f"Current environment: {Config.ENVIRONMENT}")
    logger.debug(f"Base URL: {Config.BASE_URL}")

    smtp_config = Config.get_smtp_config()
    logger.debug(f"SMTP Configuration from Config: {smtp_config}")

    logger.debug("All environment variables:")
    for key, value in os.environ.items():
        if not key.endswith(('PASSWORD', 'SECRET', 'KEY')):
            logger.debug(f"{key}: {value}")
except Exception as e:
    logger.error(f"Failed to import or use Config: {e}")

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    load_css()
    try:
        logger.debug("Entering main function")
        logger.debug(f"Current working directory: {os.getcwd()}")
        
        logger.info(f"Attempting to connect to database at: {Config.DB_PATH}")
        
        try:
            init_db()
            logger.info("Database initialized successfully")
            debug_db_contents()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
        
        query_params = st.query_params
        logger.debug(f"Query parameters: {query_params}")

        if 'action' in query_params and 'token' in query_params:
            action = query_params['action']
            token = query_params['token']
            logger.debug(f"Detected action: {action}, token: {token}")
            
            if action == 'reset':
                logger.debug("Entering password reset mode")
                st.session_state.password_reset_mode = True
                st.session_state.reset_token = token
                st.query_params.clear()
                logger.debug("Cleared query parameters")
            elif action == 'verify':
                logger.debug("Handling email verification")
                verify_email(token)
                st.query_params.clear()
                logger.debug("Cleared query parameters")

        logger.debug(f"Session state after query param processing: {st.session_state}")

        if st.session_state.get('password_reset_mode', False):
            logger.debug("Handling password reset")
            reset_successful = handle_password_reset(st.session_state.get('reset_token'))
            if reset_successful:
                logger.debug("Password reset completed successfully")
                st.session_state.password_reset_mode = False
                st.session_state.reset_token = None
                st.success("Password reset successful. You can now log in with your new password.")
                st.rerun()
            elif reset_successful is False:  # Not None, which means the form was submitted
                st.error("Failed to reset password. Please try again.")
        elif 'logged_in' not in st.session_state or not st.session_state['logged_in']:
            logger.debug("User not logged in, showing auth main")
            auth_main()
        else:
            logger.debug("User logged in, showing main app")
            main_app()

    except Exception as e:
        logger.error(f"An unexpected error occurred in main function: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred. Please try again later. Error details: {str(e)}")
    finally:
        logger.debug("Exiting main function")
        logger.debug(f"Final session state: {st.session_state}")

if __name__ == "__main__":
    main()