import streamlit as st
import bcrypt
from database import register_user, get_user, get_db_connection
from config_settings import Config
import logging
from utils import is_valid_email

logger = logging.getLogger(__name__)

def init_auth_state():
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'login_success' not in st.session_state:
        st.session_state.login_success = False
    if 'db_connection' not in st.session_state:
        st.session_state.db_connection = None

def login_user(username: str, password: str) -> bool:
    conn = get_db_connection()
    if conn is None:
        logger.error("Failed to establish database connection")
        return False
    
    try:
        user = get_user(conn, username)
        if user:
            stored_password = user['password_hash']
            if isinstance(stored_password, str):
                stored_password = stored_password.encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                st.session_state.user = user
                st.session_state.login_success = True
                st.session_state.db_connection = conn  # Store the connection in session state
                logger.info(f"User logged in: {username}")
                return True
        logger.warning(f"Failed login attempt for user: {username}")
        return False
    finally:
        if not st.session_state.login_success:
            conn.close()

def logout_user():
    if st.session_state.db_connection:
        st.session_state.db_connection.close()
    st.session_state.user = None
    st.session_state.login_success = False
    st.session_state.db_connection = None
    logger.info("User logged out")

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
    conn = get_db_connection()
    if conn is None:
        logger.error("Failed to establish database connection")
        return False
    
    try:
        if register_user(conn, username, email, hashed_password):
            logger.info(f"New user registered: {username}")
            return True
        else:
            st.error("Username or email already exists.")
            return False
    finally:
        conn.close()

def auth_page():
    st.title("Resume Cupid - Authentication")

    init_auth_state()

    if st.session_state.user:
        st.write(f"Welcome, {st.session_state.user['username']}!")
        if st.button("Logout"):
            logout_user()
            st.rerun()  # Ensure this is available in your Streamlit version
    else:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.header("Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                if login_user(username, password):
                    st.success("Logged in successfully!")
                    st.rerun()  # Ensure this is available in your Streamlit version
                else:
                    st.error("Invalid username or password")

        with tab2:
            st.header("Register")
            new_username = st.text_input("Username", key="register_username")
            new_email = st.text_input("Email", key="register_email")
            new_password = st.text_input("Password", type="password", key="register_password")
            if st.button("Register"):
                if register_new_user(new_username, new_email, new_password):
                    st.success("Registered successfully! You can now log in.")
                    st.rerun()  # Ensure this is available in your Streamlit version

def check_db_connection():
    if st.session_state.db_connection is None or st.session_state.db_connection.total_changes == -1:
        logger.info("Refreshing database connection")
        st.session_state.db_connection = get_db_connection()
    return st.session_state.db_connection is not None

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