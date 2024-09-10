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
        
        # Check for password reset token
        reset_token = st.query_params.get("reset_token")
        if reset_token:
            logger.debug(f"Detected password reset token: {reset_token}")
            st.session_state.password_reset_mode = True
            st.session_state.reset_token = reset_token
            # Clear the query parameters
            st.query_params.clear()

        # Check for email verification token
        verify_token = st.query_params.get("verify_token")
        if verify_token:
            logger.debug(f"Detected email verification token: {verify_token}")
            verification_result = verify_email(verify_token)
            if verification_result:
                logger.info(f"Email verified successfully with token: {verify_token}")
                st.success("Your email has been successfully verified! You can now log in.")
            else:
                logger.warning(f"Email verification failed with token: {verify_token}")
                st.error("Email verification failed. Please try again or contact support.")
            st.query_params.clear()

        logger.debug(f"Session state: {st.session_state}")

        if st.session_state.get('password_reset_mode', False):
            logger.debug("Handling password reset")
            auth_main()  # This will now handle the password reset
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