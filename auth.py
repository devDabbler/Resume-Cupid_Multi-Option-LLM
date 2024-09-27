import streamlit as st
import bcrypt
from database import get_user, get_user_by_email, register_user, update_user_password
from config_settings import Config
import logging
from utils import is_valid_email

logger = logging.getLogger(__name__)

def init_auth_state():
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'login_success' not in st.session_state:
        st.session_state.login_success = False

def login_user(username: str, password: str) -> bool:
    try:
        user = get_user(username)
        if user:
            stored_password = user['password_hash']
            if isinstance(stored_password, str):
                stored_password = stored_password.encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                st.session_state.user = user
                st.session_state.login_success = True
                logger.info(f"User logged in: {username}")
                return True
        logger.warning(f"Failed login attempt for user: {username}")
        return False
    except Exception as e:
        logger.error(f"Error during login for user {username}: {str(e)}")
        st.error("An unexpected error occurred during login. Please try again later.")
        return False

def logout_user():
    st.session_state.user = None
    st.session_state.login_success = False
    logger.info("User logged out")

from database import get_user, get_user_by_email, register_user  # Add these imports at the top of auth.py

def register_new_user(username: str, email: str, password: str) -> bool:
    if not username or not email or not password:
        st.error("All fields are required.")
        return False

    if not is_valid_email(email):
        st.error("Please enter a valid email address.")
        return False

    if len(password) < 8:
        st.error("Password must be at least 8 characters long.")
        return False

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    try:
        # Check if username exists
        if get_user(username):
            st.error("Username already exists.")
            return False

        # Check if email exists
        if get_user_by_email(email):
            st.error("Email already exists.")
            return False

        if register_user(username, email, hashed_password):
            logger.info(f"New user registered: {username}")
            st.success("Registration successful! You can now log in with your new credentials.")
            return True
        else:
            st.error("An error occurred during registration. Please try again.")
            return False
    except Exception as e:
        logger.error(f"Error registering user {username}: {str(e)}")
        st.error("An unexpected error occurred during registration. Please try again later.")
        return False

def reset_password(username_or_email: str, old_password: str, new_password: str) -> bool:
    if len(new_password) < 8:
        st.error("New password must be at least 8 characters long.")
        return False

    try:
        user = get_user(username_or_email)
        if not user:
            user = get_user_by_email(username_or_email)
        if user:
            stored_password = user['password_hash']
            if isinstance(stored_password, str):
                stored_password = stored_password.encode('utf-8')
            if bcrypt.checkpw(old_password.encode('utf-8'), stored_password):
                new_hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
                # You'll need to implement an update_user_password function in database.py
                if update_user_password(user['id'], new_hashed_password):
                    logger.info(f"Password reset successful for user: {user['username']}")
                    st.success("Password reset successfully!")
                    return True
                else:
                    st.error("Failed to reset password. Please try again.")
                    return False
            else:
                st.error("Invalid old password")
                return False
        else:
            st.error("Username or email not found")
            return False
    except Exception as e:
        logger.error(f"Error resetting password for user {username_or_email}: {str(e)}")
        st.error("An unexpected error occurred during password reset. Please try again later.")
        return False

def check_db_connection() -> bool:
    try:
        conn = get_db_connection()
        if conn is None:
            return False
        cur = conn.cursor()
        cur.execute("SELECT 1")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        return False

def auth_page():
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
        padding: 2rem;
    }
    .welcome-section {
        width: 100%;
        max-width: 1200px;
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        text-align: center;
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
    .features-title {
        font-size: 1.5rem;
        color: #1f4e79;
        margin-bottom: 1rem;
    }
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        text-align: left;
    }
    .feature-item {
        display: flex;
        align-items: center;
    }
    .feature-item:before {
        content: "✓";
        color: #3366cc;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .powered-by {
        font-size: 0.9rem;
        color: #666;
        margin-top: 1.5rem;
    }
    .login-section {
        width: 100%;
        max-width: 400px;
        text-align: center;
    }
    .login-title {
        font-size: 1.5rem;
        color: #3366cc;
        margin-bottom: 1rem;
    }
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .subtitle, .features-title, .login-title {
            font-size: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="welcome-section">
        <h1 class="main-title">Welcome to Resume Cupid</h1>
        <p class="subtitle">Revolutionize your hiring process with our cutting-edge AI-powered resume evaluation tool.</p>
        <h2 class="features-title">Key Features:</h2>
        <div class="features-grid">
            <div class="feature-item">Advanced AI analysis of resumes</div>
            <div class="feature-item">Instant matching against job descriptions</div>
            <div class="feature-item">Detailed candidate insights and recommendations</div>
            <div class="feature-item">Time-saving automated evaluations</div>
        </div>
        <p class="powered-by">Powered by the latest Large Language Models and Machine Learning algorithms.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="login-title">Login / Register / Reset Password</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Login", "Register", "Reset Password"])
    
    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_button", type="primary", use_container_width=True):
            if login_user(username, password):
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        new_username = st.text_input("Username", key="register_username")
        new_email = st.text_input("Email", key="register_email")
        new_password = st.text_input("Password", type="password", key="register_password")
        if st.button("Register", key="register_button", type="primary", use_container_width=True):
            if register_new_user(new_username, new_email, new_password):
                st.success("Registered successfully! You can now log in.")
                st.rerun()

    with tab3:
        username_or_email = st.text_input("Username or Email", key="reset_username_email")
        old_password = st.text_input("Old Password", type="password", key="reset_old_password")
        new_password = st.text_input("New Password", type="password", key="reset_new_password")
        if st.button("Reset Password", key="reset_password_button", type="primary", use_container_width=True):
            if reset_password(username_or_email, old_password, new_password):
                st.success("Password reset successful! You can now log in with your new password.")
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def require_auth(func):
    def wrapper(*args, **kwargs):
        init_auth_state()
        if not st.session_state.user:
            auth_page()
        else:
            if not check_db_connection():
                st.error("Database connection error. Please try logging in again.")
                logout_user()
                auth_page()
            else:
                return func(*args, **kwargs)
    return wrapper
