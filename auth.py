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

def auth_page():
    if 'registration_success' in st.session_state:
        st.success(st.session_state.registration_success)
        del st.session_state.registration_success
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
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab2:
            new_username = st.text_input("Username", key="register_username_seeker")
            new_email = st.text_input("Email", key="register_email_seeker")
            new_password = st.text_input("Password", type="password", key="register_password_seeker")
            if st.button("Register", key="register_button_seeker", type="primary", use_container_width=True):
                if register_new_user(new_username, new_email, new_password, "job_seeker"):
                    st.success("Registered successfully! You can now log in.")
                    st.rerun()

        with tab3:
            username_or_email = st.text_input("Username or Email", key="reset_username_email_seeker")
            old_password = st.text_input("Old Password", type="password", key="reset_old_password_seeker")
            new_password = st.text_input("New Password", type="password", key="reset_new_password_seeker")
            if st.button("Reset Password", key="reset_password_button_seeker", type="primary", use_container_width=True):
                if reset_password(username_or_email, old_password, new_password, "job_seeker"):
                    st.success("Password reset successful! You can now log in with your new password.")
                    st.rerun()

    else:  # Employer
        st.markdown('<h2 class="login-title">Employer Access</h2>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Login", "Register", "Reset Password"])

        with tab1:
            username = st.text_input("Company Username", key="login_username_employer")
            password = st.text_input("Password", type="password", key="login_password_employer")
            if st.button("Login", key="login_button_employer", type="primary", use_container_width=True):
                if login_user(username, password, "employer"):
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab2:
            new_username = st.text_input("Company Username", key="register_username_employer")
            new_email = st.text_input("Company Email", key="register_email_employer")
            new_password = st.text_input("Password", type="password", key="register_password_employer")
            if st.button("Register Your Company", key="register_button_employer", type="primary", use_container_width=True):
                if register_new_user(new_username, new_email, new_password, "employer"):
                    st.success("Company registered successfully! You can now log in.")
                    st.rerun()

        with tab3:
            username_or_email = st.text_input("Company Username or Email", key="reset_username_email_employer")
            old_password = st.text_input("Old Password", type="password", key="reset_old_password_employer")
            new_password = st.text_input("New Password", type="password", key="reset_new_password_employer")
            if st.button("Reset Password", key="reset_password_button_employer", type="primary", use_container_width=True):
                if reset_password(username_or_email, old_password, new_password, "employer"):
                    st.success("Password reset successful! You can now log in with your new password.")
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
