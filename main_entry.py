import streamlit as st
import os
from main_app import main_app
from auth import auth_main, handle_password_reset, login_page, verify_email
import logging
from dotenv import load_dotenv
from database import init_db
import sys

st.set_page_config(page_title="Resume Cupid", page_icon="💘", layout="centered")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the .env.development file
try:
    load_dotenv(dotenv_path=r'C:\Users\SEAN COLLINS\Resume_Cupid_Multi_LLM\.env.development')
    logger.info("Loaded environment variables from .env.development")
except Exception as e:
    logger.error(f"Failed to load .env.development file: {e}")

# Now import Config after loading environment variables
try:
    from config_settings import Config

    # Log environment and configuration details
    logger.debug(f"Current environment: {Config.ENVIRONMENT}")
    logger.debug(f"Base URL: {Config.BASE_URL}")

    smtp_config = Config.get_smtp_config()
    logger.debug(f"SMTP Configuration from Config: {smtp_config}")

    # Log all environment variables (excluding sensitive ones)
    logger.debug("All environment variables:")
    for key, value in os.environ.items():
        if not key.endswith(('PASSWORD', 'SECRET', 'KEY')):
            logger.debug(f"{key}: {value}")
except Exception as e:
    logger.error(f"Failed to import or use Config: {e}")
    
    st.set_page_config(page_title="Resume Cupid", page_icon="💘")

def main():
    try:
        logger.debug("Entering main function")
        logger.debug(f"Current working directory: {os.getcwd()}")
        
        # Log the database path
        logger.info(f"Attempting to connect to database at: {Config.DB_PATH}")
        
        # Initialize the database
        try:
            init_db()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # Optionally exit the application if database initialization fails
            # st.error("Failed to initialize the database. Please contact support.")
            # sys.exit(1)
        
        query_params = st.query_params
        logger.debug(f"Query parameters: {query_params}")

        if 'action' in query_params and 'token' in query_params:
            action = query_params['action']
            token = query_params['token']
            logger.debug(f"Detected action: {action}, token: {token}")
            
            if action == 'verify':
                logger.debug("Handling email verification")
                verify_email(token)
                st.query_params.clear()
                logger.debug("Cleared query parameters")
                st.rerun()
            elif action == 'reset':
                logger.debug("Entering password reset mode")
                st.session_state.password_reset_mode = True
                st.session_state.reset_token = token
                st.query_params.clear()
                logger.debug("Cleared query parameters")
                st.rerun()

        logger.debug(f"Session state: {st.session_state}")

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
        logger.error(f"An error occurred in the main function: {e}", exc_info=True)
        st.error(f"An unexpected error occurred. Please try again later. Error details: {str(e)}")

if __name__ == "__main__":
    main()