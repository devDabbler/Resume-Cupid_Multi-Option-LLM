import streamlit as st
import os
import logging
from auth import auth_main, handle_password_reset, login_page, verify_email
from config_settings import Config

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

def main():
    logger.debug("Entering main function")
    logger.debug(f"Initial session state: {st.session_state}")

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

    try:
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

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred. Please try again later. Error details: {str(e)}")

    logger.debug(f"Final session state: {st.session_state}")

if __name__ == "__main__":
    main()