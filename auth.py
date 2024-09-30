import streamlit as st
import bcrypt
import secrets
from database import get_user, get_user_by_email, register_user, update_user_password, execute_with_retry, get_db_connection, update_user_reset_token
from config_settings import Config
import logging
from email_service import email_service
from datetime import datetime, timedelta
from typing import Optional
import re
import uuid

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

def register_user(username: str, email: str, password_hash: bytes, user_type: str) -> bool:
    verification_token = str(uuid.uuid4())  # Generate a verification token

    def _register(conn):
        cur = conn.cursor()
        try:
            cur.execute('''
            INSERT INTO users (username, email, password_hash, user_type, profile, verification_token, is_verified)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, user_type, json.dumps({}), verification_token, False))
            conn.commit()
            logger.info(f"{user_type.capitalize()} registered successfully: {username}")

            # Call the email service to send verification email
            email_sent = email_service.send_verification_email(email, verification_token)
            if email_sent:
                logger.info(f"Verification email sent to {email}")
                st.success("Registration successful. Please check your email to verify your account.")
            else:
                logger.error(f"Failed to send verification email to {email}")
                st.error("Failed to send verification email. Please try again later.")
            
            return True
        except sqlite3.IntegrityError as e:
            logger.error(f"IntegrityError while registering {user_type} {username}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while registering {user_type} {username}: {str(e)}")
            return False

    try:
        return execute_with_retry(_register)
    except Exception as e:
        logger.error(f"Error registering {user_type} {username}: {str(e)}")
        return False

def initiate_password_reset(email: str, user_type: str = "job_seeker") -> bool:
    try:
        # Retrieve user by email and user_type
        user = get_user_by_email(email, user_type)
        
        if user:
            # Generate reset token
            reset_token = str(uuid.uuid4())
            
            # Update the reset token in the database
            if update_user_reset_token(user['id'], reset_token):
                # Send the password reset email
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
    
def generate_verification_token() -> str:
    return secrets.token_urlsafe(32)

def auth_page():
    if 'auth_message' in st.session_state:
        message_type = st.session_state.auth_message['type']
        message_content = st.session_state.auth_message['content']
        if message_type == 'success':
            st.success(message_content)
        elif message_type == 'error':
            st.error(message_content)
        del st.session_state.auth_message
    st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        padding: 0;
    }
    .main-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        padding: 0rem;
    }
    .welcome-section {
        width: 100%;
        max-width: 1200px;
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .main-title {
        font-size: 2.5rem;
        color: #3366cc;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .user-type-selection {
        display: flex;
        justify-content: center;
        margin-bottom: 0rem;
    }
    .login-section {
        width: 100%;
        max-width: 400px;
        text-align: center;
        padding: 0rem;
        border-radius: 10px;
        background-color: #ffffff;
        margin-bottom: 1rem;
    }
    .login-title {
        font-size: 1.5rem;
        color: #3366cc;
        margin-bottom: 1rem;
    }
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-bottom: 1.5rem;
        width: 100%;
        max-width: 1200px;
    }
    .feature-item {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .feature-item h3 {
        color: #1e3a8a;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    .feature-item p {
        color: #4b5563;
        font-size: 1rem;
    }
    .powered-by {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .subtitle, .login-title {
            font-size: 1rem;
        }
        .features-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-section">
        <h1 class="main-title">Welcome to Resume Cupid</h1>
        <p class="subtitle">Redefining recruitment by simplifying the hiring process for employers and helping job seekers distinguish themselves. Utilizing advanced Large Language Models and Machine Learning algorithms to drive innovation in talent acquisition and career growth.</p>
    </div>
    <div class="features-grid">
        <div class="feature-item">
            <h3>AI-Powered Analysis</h3>
            <p>Cutting-edge resume evaluation using advanced AI algorithms</p>
        </div>
        <div class="feature-item">
            <h3>Instant Matching</h3>
            <p>Seamless alignment of candidate profiles with job requirements</p>
        </div>
        <div class="feature-item">
            <h3>Deep Insights</h3>
            <p>Comprehensive candidate assessments and tailored recommendations</p>
        </div>
        <div class="feature-item">
            <h3>Time-Saving Automation</h3>
            <p>Streamlined hiring process with automated evaluations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # User type selection title
    st.markdown("""
    <h2 style="text-align:center; color:#1e3a8a; margin-bottom: 1rem;">Begin Your Personalized AI Journey</h2>
    """, unsafe_allow_html=True)

    # User type selection
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
                if login_user(username, password, "job_seeker"):
                    st.session_state.auth_message = {'type': 'success', 'content': "Logged in successfully!"}
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab2:
            new_username = st.text_input("Username", key="register_username_seeker")
            new_email = st.text_input("Email", key="register_email_seeker")
            new_password = st.text_input("Password", type="password", key="register_password_seeker")
            if st.button("Register", key="register_button_seeker", type="primary", use_container_width=True):
                if register_new_user(new_username, new_email, new_password, "job_seeker"):
                    st.session_state.auth_message = {'type': 'success', 'content': "Registered successfully! Please check your email to verify your account."}
                    st.rerun()

        with tab3:
            reset_email = st.text_input("Email", key="reset_email_seeker")
            if st.button("Reset Password", key="reset_password_button_seeker", type="primary", use_container_width=True):
                if initiate_password_reset(reset_email, "job_seeker"):
                    st.session_state.auth_message = {'type': 'success', 'content': "Password reset instructions have been sent to your email."}
                    st.rerun()

    else:  # Employer
        st.markdown('<h2 class="login-title">Employer Access</h2>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Login", "Register", "Reset Password"])

        with tab1:
            username = st.text_input("Username", key="login_username_employer")
            password = st.text_input("Password", type="password", key="login_password_employer")
            if st.button("Login", key="login_button_employer", type="primary", use_container_width=True):
                if login_user(username, password, "employer"):
                    st.session_state.auth_message = {'type': 'success', 'content': "Logged in successfully!"}
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab2:
            new_username = st.text_input("Username", key="register_username_employer")
            new_email = st.text_input("Email", key="register_email_employer")
            new_password = st.text_input("Password", type="password", key="register_password_employer")
            if st.button("Register Your Company", key="register_button_employer", type="primary", use_container_width=True):
                if register_new_user(new_username, new_email, new_password, "employer"):
                    st.session_state.auth_message = {'type': 'success', 'content': "Company registered successfully! Please check your email to verify your account."}
                    st.rerun()

        with tab3:
            reset_email = st.text_input("Email", key="reset_email_employer")
            if st.button("Reset Password", key="reset_password_button_employer", type="primary", use_container_width=True):
                if initiate_password_reset(reset_email, "employer"):
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

def verify_email(token: str) -> bool:
    try:
        if execute_with_retry(lambda conn: conn.execute('''
            UPDATE users
            SET is_verified = 1, verification_token = NULL
            WHERE verification_token = ?
        ''', (token,))):
            logger.info(f"Email verified successfully with token: {token}")
            return True
        else:
            logger.warning(f"Email verification failed for token: {token}")
            return False
    except Exception as e:
        logger.error(f"Error during email verification: {str(e)}")
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

# Explicitly export the functions
__all__ = ['require_auth', 'init_auth_state', 'auth_page', 'logout_user', 'login_user', 'register_new_user', 'initiate_password_reset', 'verify_email', 'reset_password_page']
            