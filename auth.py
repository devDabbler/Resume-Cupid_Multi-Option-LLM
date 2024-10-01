import bcrypt
import logging
import streamlit as st
import uuid
from datetime import datetime, timedelta
from typing import Optional
import re
from database import get_user, get_user_by_email, register_user, update_user_password, update_user_reset_token, execute_with_retry, get_db_connection
from email_service import email_service
from config_settings import Config
import streamlit as st
from main_app import custom_notification

logger = logging.getLogger(__name__)

class Authenticator:
    def __init__(self):
        self.cookie_name = Config.COOKIE_NAME
        self.cookie_key = Config.COOKIE_KEY
        self.cookie_expiry_days = Config.COOKIE_EXPIRY_DAYS

    def login(self, username: str, password: str, user_type: str) -> bool:
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

    def logout(self):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.user = None
        st.session_state.login_success = False

    def register(self, username: str, email: str, password: str, user_type: str) -> bool:
        if not self.is_valid_email(email):
            logger.error(f"Invalid email format: {email}")
            return False

        if len(password) < 8:  # Example minimum password length
            logger.error("Password is too short")
            return False

        existing_user = get_user(username, user_type)
        if existing_user:
            logger.error(f"Username already exists: {username}")
            return False

        existing_email = get_user_by_email(email, user_type)
        if existing_email:
            logger.error(f"Email already registered: {email}")
            return False

        # If we've passed all checks, proceed with registration
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        verification_token = str(uuid.uuid4())
    
        if register_user(username, email, password_hash, user_type, verification_token):
            # Send verification email
            if email_service.send_verification_email(email, verification_token):
                logger.info(f"Verification email sent to {email}")
                return True
            else:
                logger.error(f"Failed to send verification email to {email}")
                return False
        else:
            logger.error(f"Failed to register user: {username}")
            return False

    def initiate_password_reset(self, email: str, user_type: str = "job_seeker") -> bool:
        try:
            user = get_user_by_email(email, user_type)
            if user:
                reset_token = str(uuid.uuid4())
                expiration_time = datetime.utcnow() + timedelta(hours=24)
                if update_user_reset_token(user['id'], reset_token, expiration_time):
                    email_sent = email_service.send_password_reset_email(email, reset_token)
                    if email_sent:
                        logger.info(f"Password reset email sent to {email}")
                        st.success("Password reset email sent successfully. Please check your email.")
                        return True
                    else:
                        logger.error(f"Failed to send password reset email to {email}")
                        st.error("Failed to send password reset email. Please try again later.")
                else:
                    logger.error(f"Failed to update reset token for {email}")
                    st.error("Failed to process password reset. Please try again later.")
            else:
                logger.warning(f"{user_type.capitalize()} email not found: {email}")
                st.error("No user found with this email address.")
            return False
        except Exception as e:
            logger.error(f"Error initiating password reset for {user_type} {email}: {str(e)}", exc_info=True)
            st.error("An unexpected error occurred while processing the password reset. Please try again later.")
            return False

    def verify_email(self, token: str) -> bool:
        try:
            def _verify(conn):
                cur = conn.cursor()
                cur.execute('''
                    UPDATE users
                    SET is_verified = 1, verification_token = NULL
                    WHERE verification_token = ?
                ''', (token,))
                conn.commit()
                return cur.rowcount > 0

            result = execute_with_retry(_verify)
            if result:
                logger.info(f"Email verified successfully with token: {token}")
                return True
            else:
                logger.warning(f"Email verification failed for token: {token}")
                return False
        except Exception as e:
            logger.error(f"Error during email verification: {str(e)}")
            return False

    def complete_password_reset(self, reset_token: str, new_password: str) -> bool:
        user = self.get_user_by_reset_token(reset_token)
        if user:
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            if update_user_password(user['id'], hashed_password):
                self.clear_reset_token(user['id'])
                logger.info(f"Password reset completed for user {user['id']}")
                return True
        logger.warning("Invalid or expired reset token")
        return False

    @staticmethod
    def is_valid_email(email: str) -> bool:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    @staticmethod
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

    @staticmethod
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

# Create a global instance of the Authenticator
authenticator = Authenticator()

def init_auth_state():
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'login_success' not in st.session_state:
        st.session_state.login_success = False

def require_auth(func):
    def wrapper(*args, **kwargs):
        init_auth_state()
        if not st.session_state.user:
            auth_page()
        else:
            if not check_db_connection():
                logger.error("Database connection error during authentication check")
                st.error("Database connection error. Please try logging in again.")
                authenticator.logout()
                auth_page()
            else:
                return func(*args, **kwargs)
    return wrapper

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
        
        # Use both standard Streamlit notifications and custom notifications
        if message_type == 'success':
            st.success(message_content)
        elif message_type == 'error':
            st.error(message_content)
        
        # Add custom notification (assuming custom_notification is defined in main_app.py)
        if 'custom_notification' in globals():
            custom_notification(message_content, message_type)
        
        del st.session_state.auth_message

    # Your existing CSS styles here (unchanged)
    st.markdown("""
    <style>
    /* ... (keep your existing CSS styles) ... */
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Your existing welcome section and features grid here (unchanged)
    st.markdown("""
    <div class="welcome-section">
        <h1 class="main-title">Welcome to Resume Cupid</h1>
        <p class="subtitle">Redefining recruitment by simplifying the hiring process for employers and helping job seekers distinguish themselves. Utilizing advanced Large Language Models and Machine Learning algorithms to drive innovation in talent acquisition and career growth.</p>
    </div>
    <div class="features-grid">
        <!-- ... (keep your existing feature items) ... -->
    </div>
    """, unsafe_allow_html=True)

    # User type selection title and radio button (unchanged)
    st.markdown("""
    <h2 style="text-align:center; color:#1e3a8a; margin-bottom: 1rem;">Begin Your Personalized AI Journey</h2>
    """, unsafe_allow_html=True)

    st.markdown('<div class="user-type-selection">', unsafe_allow_html=True)
    user_type = st.radio("I am a:", ("Job Seeker", "Employer"), horizontal=True, key="user_type_selection")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="login-section">', unsafe_allow_html=True)

    if user_type == "Job Seeker":
        st.markdown('<h2 class="login-title">Job Seeker Access</h2>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Login", "Register", "Reset Password"])

        with tab1:
            username = st.text_input("Username", key="login_username_seeker")
            password = st.text_input("Password", type="password", key="login_password_seeker")
            if st.button("Login", key="login_button_seeker", type="primary", use_container_width=True):
                if authenticator.login(username, password, "job_seeker"):
                    st.session_state.auth_message = {'type': 'success', 'content': "Logged in successfully!"}
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab2:
            new_username = st.text_input("Username", key="register_username_seeker")
            new_email = st.text_input("Email", key="register_email_seeker")
            new_password = st.text_input("Password", type="password", key="register_password_seeker")
            if st.button("Register", key="register_button_seeker", type="primary", use_container_width=True):
                if authenticator.register(new_username, new_email, new_password, "job_seeker"):
                    st.session_state.auth_message = {'type': 'success', 'content': "Registered successfully! Please check your email to verify your account."}
                    st.rerun()

        with tab3:
            reset_email = st.text_input("Email", key="reset_email_seeker")
            if st.button("Reset Password", key="reset_password_button_seeker", type="primary", use_container_width=True):
                if authenticator.initiate_password_reset(reset_email, "job_seeker"):
                    st.session_state.auth_message = {'type': 'success', 'content': "Password reset instructions have been sent to your email."}
                    st.rerun()

    else:  # Employer
        st.markdown('<h2 class="login-title">Employer Access</h2>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Login", "Register", "Reset Password"])

        with tab1:
            username = st.text_input("Username", key="login_username_employer")
            password = st.text_input("Password", type="password", key="login_password_employer")
            if st.button("Login", key="login_button_employer", type="primary", use_container_width=True):
                if authenticator.login(username, password, "employer"):
                    st.session_state.auth_message = {'type': 'success', 'content': "Logged in successfully!"}
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab2:
            new_username = st.text_input("Username", key="register_username_employer")
            new_email = st.text_input("Email", key="register_email_employer")
            new_password = st.text_input("Password", type="password", key="register_password_employer")
            if st.button("Register Your Company", key="register_button_employer", type="primary", use_container_width=True):
                if authenticator.register(new_username, new_email, new_password, "employer"):
                    st.session_state.auth_message = {'type': 'success', 'content': "Company registered successfully! Please check your email to verify your account."}
                    st.rerun()

        with tab3:
            reset_email = st.text_input("Email", key="reset_email_employer")
            if st.button("Reset Password", key="reset_password_button_employer", type="primary", use_container_width=True):
                if authenticator.initiate_password_reset(reset_email, "employer"):
                    st.session_state.auth_message = {'type': 'success', 'content': "Password reset instructions have been sent to your email."}
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <footer class="footer" style="text-align:center; padding: 1rem 0; color: #6b7280; margin-top: 1.5rem;">
        <p>&copy; 2024 Resume Cupid. All rights reserved.</p>
        <p>Contact us: hello@resumecupid.ai</p>
    </footer>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

__all__ = ['Authenticator', 'authenticator', 'require_auth', 'init_auth_state', 'auth_page', 'reset_password_page']