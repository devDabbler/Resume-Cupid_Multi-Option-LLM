import streamlit as st
import bcrypt
import secrets
from database import get_user, get_user_by_email, register_user, update_user_password
from config_settings import Config
import logging
from email_service import email_service
import re

logger = logging.getLogger(__name__)

def generate_verification_token() -> str:
    return secrets.token_urlsafe(32)

def init_auth_state():
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'login_success' not in st.session_state:
        st.session_state.login_success = False

def login_user(username: str, password: str, user_type: str) -> bool:
    try:
        user = get_user(username, user_type)
        if user:
            stored_password = user['password_hash']
            if isinstance(stored_password, str):
                stored_password = stored_password.encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                st.session_state.user = user
                st.session_state['logged_in_user_type'] = user_type
                st.session_state.login_success = True
                logger.info(f"{user_type.capitalize()} logged in: {username}")
                return True
        logger.warning(f"Failed login attempt for {user_type}: {username}")
        return False
    except Exception as e:
        logger.error(f"Error during login for {user_type} {username}: {str(e)}")
        st.error("An unexpected error occurred during login. Please try again later.")
        return False

def is_valid_email(email: str) -> bool:
    """
    Validate an email address using a simple regex pattern.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def logout_user():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.user = None
    st.session_state.login_success = False

def register_new_user(username: str, email: str, password: str, user_type: str) -> bool:
    logger.info(f"Attempting to register new {user_type}: {username}")
    
    if not username or not email or not password:
        logger.warning("Registration failed: All fields are required.")
        st.error("All fields are required.")
        return False

    if not is_valid_email(email):
        logger.warning(f"Registration failed: Invalid email address - {email}")
        st.error("Please enter a valid email address.")
        return False

    if len(password) < 8:
        logger.warning("Registration failed: Password too short.")
        st.error("Password must be at least 8 characters long.")
        return False

    try:
        # Check if username exists
        if get_user(username, user_type):
            logger.warning(f"Registration failed: {user_type.capitalize()} username already exists - {username}")
            st.error(f"{user_type.capitalize()} username already exists.")
            return False

        # Check if email exists
        if get_user_by_email(email, user_type):
            logger.warning(f"Registration failed: {user_type.capitalize()} email already exists - {email}")
            st.error(f"{user_type.capitalize()} email already exists.")
            return False

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        verification_token = generate_verification_token()

        if register_user(username, email, hashed_password, user_type, verification_token):
            logger.info(f"New {user_type} registered: {username}")
            
            success_message = "Registration successful! "
            
            if Config.ENVIRONMENT == 'development':
                verification_link = email_service.get_verification_link(verification_token)
                success_message += f"""
                Since this is a development environment, we're displaying the verification link here:
                
                {verification_link}
                
                In a production environment, this link would be sent to {email}.
                Click the link above to verify your account.
                """
                logger.info(f"Development mode: Verification link generated for {email}")
            else:
                try:
                    if email_service.send_verification_email(email, verification_token):
                        success_message += f"""
                        An email has been sent to {email} with a verification link. 
                        Please check your email (including spam folder) and click the link to verify your account.
                        
                        After verifying your email, you can return here to log in.
                        """
                        logger.info(f"Verification email sent successfully to {email}")
                    else:
                        success_message += """
                        We couldn't send the verification email. 
                        Please contact support to verify your account.
                        """
                        logger.error(f"Failed to send verification email to {email}")
                except Exception as e:
                    success_message += """
                    We couldn't send the verification email. 
                    Please contact support to verify your account.
                    """
                    logger.error(f"Exception occurred while sending verification email to {email}: {str(e)}")
            
            st.success(success_message)
            st.rerun()
            return True
        else:
            logger.error(f"Failed to register user in database: {username}")
            st.error("An error occurred during registration. Please try again.")
            return False
    except Exception as e:
        logger.error(f"Unexpected error during registration for {user_type} {username}: {str(e)}")
        st.error("An unexpected error occurred during registration. Please try again later.")
        return False

def reset_password(email: str, user_type: str) -> bool:
    try:
        user = get_user_by_email(email, user_type)
        if user:
            reset_token = generate_verification_token()
            if update_user_reset_token(user['id'], reset_token):
                if email_service.send_password_reset_email(email, reset_token):
                    logger.info(f"Password reset email sent to {email}")
                    st.success("Password reset instructions have been sent to your email.")
                    return True
                else:
                    logger.error(f"Failed to send password reset email to {email}")
                    st.error("Failed to send password reset email. Please try again.")
            else:
                logger.error(f"Failed to update reset token for user {email}")
                st.error("Failed to initiate password reset. Please try again.")
        else:
            logger.warning(f"{user_type.capitalize()} email not found: {email}")
            st.error(f"{user_type.capitalize()} email not found")
        return False
    except Exception as e:
        logger.error(f"Error resetting password for {user_type} {email}: {str(e)}")
        st.error("An unexpected error occurred during password reset. Please try again later.")
        return False

def check_db_connection() -> bool:
    try:
        conn = get_db_connection()
        if conn is None:
            logger.error("Failed to establish database connection")
            return False
        cur = conn.cursor()
        cur.execute("SELECT 1")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        return False

# ... (rest of the code remains unchanged)

def require_auth(func):
    def wrapper(*args, **kwargs):
        init_auth_state()
        if not st.session_state.user:
            auth_page()
        else:
            if not check_db_connection():
                logger.error("Database connection error during authentication check")
                st.error("Database connection error. Please try logging in again.")
                logout_user()
                auth_page()
            else:
                return func(*args, **kwargs)
    return wrapper
