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
        text-align: center; /* Center text inside the section */
    }
    .main-title {
        font-size: 2.5rem; /* Adjusted for responsiveness */
        color: #3366cc;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem; /* Adjusted for responsiveness */
        color: #555;
        margin-bottom: 1.5rem;
    }
    .features-title {
        font-size: 1.5rem; /* Adjusted for responsiveness */
        color: #1f4e79;
        margin-bottom: 1rem;
    }
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        text-align: left; /* Align text to the left inside the grid */
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
        font-size: 0.9rem; /* Adjusted for responsiveness */
        color: #666;
        margin-top: 1.5rem;
    }
    .login-section {
        width: 100%;
        max-width: 400px;
        text-align: center; /* Center text inside the login section */
    }
    .login-title {
        font-size: 1.5rem; /* Adjusted for responsiveness */
        color: #3366cc;
        margin-bottom: 1rem;
    }
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem; /* Smaller font size for mobile */
        }
        .subtitle, .features-title, .login-title {
            font-size: 1rem; /* Smaller font size for mobile */
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
    st.markdown('<h2 class="login-title">Login / Register</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
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

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

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
