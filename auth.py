import streamlit as st
import bcrypt
import secrets
from database import get_user, get_user_by_email, register_user, update_user_password, execute_with_retry, get_db_connection, update_user_reset_token, get_user_by_reset_token, clear_reset_token
from config_settings import Config
import logging
from email_service import email_service
from datetime import datetime, timedelta
from typing import Optional
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
        if get_user(username, user_type):
            logger.warning(f"Registration failed: {user_type.capitalize()} username already exists - {username}")
            st.error(f"{user_type.capitalize()} username already exists.")
            return False

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
                    return True
                else:
                    logger.error(f"Failed to send password reset email to {email}")
        else:
            logger.warning(f"{user_type.capitalize()} email not found: {email}")
        return False
    except Exception as e:
        logger.error(f"Error resetting password for {user_type} {email}: {str(e)}")
        return False

def get_user_by_reset_token(reset_token: str) -> Optional[dict]:
    try:
        def _get_user(conn):
            cur = conn.cursor()
            cur.execute('''
                SELECT id, username, email
                FROM users
                WHERE reset_token = ? AND reset_token_expiration > ?
            ''', (reset_token, datetime.utcnow()))
            user = cur.fetchone()
            return dict(user) if user else None

        user = execute_with_retry(_get_user)

        if user:
            logger.info(f"User {user['id']} retrieved by reset token")
        else:
            logger.info("No valid user found for the given reset token")

        return user
    except Exception as e:
        logger.error(f"Error retrieving user by reset token: {str(e)}")
        return None

def clear_reset_token(user_id: int) -> bool:
    try:
        def _clear_token(conn):
            cur = conn.cursor()
            cur.execute('''
                UPDATE users
                SET reset_token = NULL, reset_token_expiration = NULL
                WHERE id = ?
            ''', (user_id,))
            conn.commit()
            return cur.rowcount > 0

        success = execute_with_retry(_clear_token)

        if success:
            logger.info(f"Reset token cleared for user {user_id}")
        else:
            logger.warning(f"No user found with ID {user_id} when clearing reset token")

        return success
    except Exception as e:
        logger.error(f"Error clearing reset token for user {user_id}: {str(e)}")
        return False

def complete_password_reset(reset_token: str, new_password: str) -> bool:
    user = get_user_by_reset_token(reset_token)
    if user:
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        if update_user_password(user['id'], hashed_password):
            clear_reset_token(user['id'])
            logger.info(f"Password reset completed for user {user['id']}")
            return True
    logger.warning("Invalid or expired reset token")
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

def auth_page():
    if 'auth_message' in st.session_state:
        message_type = st.session_state.auth_message['type']
        message_content = st.session_state.auth_message['content']
        if message_type == 'success':
            st.success(message_content)
        elif message_type == 'error':
            st.error(message_content)
        del st.session_state.auth_message

    # Rest of the function remains unchanged

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

# Explicitly export the functions
__all__ = ['require_auth', 'init_auth_state', 'auth_page', 'logout_user', 'login_user', 'register_new_user', 'reset_password']
