import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Email configuration
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
FROM_EMAIL = os.getenv('FROM_EMAIL')

def send_email(to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}. Error: {str(e)}")
        return False

def send_verification_email(to_email, verification_token):
    subject = "Verify Your Email for Resume Cupid"
    verification_link = f"http://localhost:8501/verify?token={verification_token}"
    body = f"""
    Hello,

    Thank you for registering with Resume Cupid. Please click on the link below to verify your email address:

    {verification_link}

    If you didn't register for Resume Cupid, please ignore this email.

    Best regards,
    The Resume Cupid Team
    """
    return send_email(to_email, subject, body)

def send_password_reset_email(to_email, reset_token):
    subject = "Reset Your Password for Resume Cupid"
    reset_link = f"http://localhost:8501/reset_password?token={reset_token}"
    body = f"""
    Hello,

    You have requested to reset your password for Resume Cupid. Please click on the link below to set a new password:

    {reset_link}

    If you didn't request a password reset, please ignore this email.

    Best regards,
    The Resume Cupid Team
    """
    return send_email(to_email, subject, body)