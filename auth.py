import streamlit as st
import bcrypt
import logging
from database import register_user, authenticate_user, set_reset_token, reset_password, set_verification_token, verify_user, is_user_verified
from email_utils import send_verification_email, send_password_reset_email

logger = logging.getLogger(__name__)

# Custom CSS for branding and UI enhancement
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f2f6;
        margin: 0;
        padding: 0;
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    .main-title {
        font-size: 3.5em;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
        margin: 0.5rem 0 0.25rem;
    }
    
    .subtitle {
        font-size: 1.2em;
        color: #34495E;
        text-align: center;
        margin: 0 0 0.5rem;
    }
    
    .login-form {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 100%;
        max-width: 350px;
        margin: 0.5rem auto 0;
    }
    
    .login-button {
        background-color: #3498DB;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5em 1em;
        margin-top: 1rem;
        width: 100%;
    }
    
    .footer-text {
        text-align: center;
        width: 100%;
        max-width: 800px;
        margin: 1rem auto;
    }
    
    h3 {
        margin-bottom: 1rem !important;
    }
    
    .stButton>button {
        width: 100%;
    }

    .stTabs [role="tabpanel"] {
        padding: 0 !important;
        margin: 0 !important;
    }

    .stForm {
        margin-top: -1rem !important;
    }
    
    .login-form {
        padding-top: 1rem !important;
    }
    
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
    }

    .element-container {
        margin-bottom: 0.5rem !important;
    }
</style>
"""

def main_auth_page():
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown("<h1 class='main-title'>Resume Cupid 💘</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Welcome to Resume Cupid - Your AI-powered resume evaluation tool</p>", unsafe_allow_html=True)

    st.markdown('<div style="margin-top: -2rem;"></div>', unsafe_allow_html=True)

    if st.session_state.get('password_reset_mode', False):
        handle_password_reset(st.session_state.get('reset_token'))
    else:
        tab1, tab2, tab3 = st.tabs(["Login", "Register", "Reset Password"])

    with tab1:
        login_page()
    
    with tab2:
        register_page()
    
    with tab3:
        reset_password_page()

    st.markdown('<style>footer {visibility: hidden;}</style>', unsafe_allow_html=True)

def login_page():
    st.markdown('<div class="login-form">', unsafe_allow_html=True)
    with st.form(key='login_form'):
        st.markdown("<h2>Login</h2>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            user = authenticate_user(username, password)
            if user:
                verified = is_user_verified(username)
                logger.debug(f"User {username} verification status: {verified}")
                if verified:
                    logger.info(f"User {username} logged in successfully")
                    st.success("Login successful! Redirecting to the main app...")
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    logger.warning(f"User {username} not verified")
                    st.error("Please verify your email before logging in.")
            else:
                logger.warning(f"Failed login attempt for user {username}")
                st.error("Invalid username or password.")
    st.markdown('</div>', unsafe_allow_html=True)

def register_page():
    st.markdown('<div class="login-form">', unsafe_allow_html=True)
    with st.form(key='register_form'):
        st.markdown("<h2>Register</h2>", unsafe_allow_html=True)
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        register_button = st.form_submit_button("Register")

        if register_button:
            if password != confirm_password:
                st.error("Passwords do not match.")
            elif register_user(username, email, password):
                verification_token = set_verification_token(email)
                if send_verification_email(email, verification_token):
                    st.success("Registration successful! Please check your email for verification.")
                else:
                    st.warning("Registration successful, but failed to send verification email.")
            else:
                st.error("Registration failed. Username or email may already be in use.")
    st.markdown('</div>', unsafe_allow_html=True)

def reset_password_page():
    logger.debug(f"Session state: {st.session_state}")
    
    st.markdown('<div class="login-form">', unsafe_allow_html=True)
    with st.form(key='reset_password_form'):
        st.markdown("<h2>Reset Password</h2>", unsafe_allow_html=True)
        reset_email = st.text_input("Email")
        reset_button = st.form_submit_button("Reset Password")

        if reset_button:
            reset_token = set_reset_token(reset_email)
            if reset_token:
                if send_password_reset_email(reset_email, reset_token):
                    st.success("Password reset link sent to your email.")
                else:
                    st.error("Failed to send password reset email.")
            else:
                st.error("Email not found.")
    st.markdown('</div>', unsafe_allow_html=True)

def verify_email(token):
    logger.debug(f"Attempting to verify email with token: {token}")
    if verify_user(token):
        logger.info(f"Email verified successfully with token: {token}")
        st.success("Your email has been successfully verified! You can now log in.")
        st.session_state['email_verified'] = True
    else:
        logger.warning(f"Failed to verify email with token: {token}")
        st.error("Invalid or expired verification token.")

def handle_password_reset(token):
    logger.debug(f"Handling password reset for token: {token}")
    
    st.markdown('<div class="login-form">', unsafe_allow_html=True)
    st.markdown("<h2>Set New Password</h2>", unsafe_allow_html=True)

    with st.form(key='new_password_form'):
        new_password = st.text_input("New Password", type="password")
        confirm_new_password = st.text_input("Confirm New Password", type="password")
        submit_button = st.form_submit_button("Set New Password")

        if submit_button:
            if new_password != confirm_new_password:
                st.error("Passwords do not match.")
                logger.error("Passwords do not match")
                return None
            elif reset_password(token, new_password):
                logger.info("Password reset successful")
                st.success("Password reset successful. You can now log in with your new password.")
                st.session_state.password_reset_mode = False
                st.session_state.reset_token = None
                return "SUCCESS"
            else:
                logger.error("Failed to reset password")
                st.error("Failed to reset password. The token may be invalid or expired.")
                return False

    st.markdown('</div>', unsafe_allow_html=True)
    return None

def auth_main():
    logger.debug(f"Auth main called. Session state: {st.session_state}")
    
    if st.session_state.get('password_reset_mode', False):
        handle_password_reset(st.session_state.get('reset_token'))
    else:
        main_auth_page()
    
    query_params = st.query_params
    logger.debug(f"Query parameters: {query_params}")
    
    if 'password_reset_complete' in st.session_state:
        st.info("Your password has been reset. Please log in with your new password.")
        del st.session_state['password_reset_complete']
        st.rerun()
    
    if 'action' in query_params and 'token' in query_params:
        action = query_params['action'][0]
        token = query_params['token'][0]
        
        if action == 'verify':
            verify_email(token)
            return
        elif action == 'reset':
            st.session_state.password_reset_mode = True
            st.session_state.reset_token = token
            return

    if st.session_state.get('password_reset_mode', False):
        result = handle_password_reset(st.session_state.get('reset_token'))
        if result == "SUCCESS":
            st.success("Password reset successful. You can now log in with your new password.")
            st.session_state.password_reset_mode = False
            st.session_state.reset_token = None
            st.session_state.show_login = True
            st.rerun()
        return
    
    main_auth_page()