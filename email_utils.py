import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from utils import get_logger
import logging
from config_settings import Config
import time
import socket

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info(f"Using BASE_URL: {Config.BASE_URL}")

# Get SMTP configuration
smtp_config = Config.get_smtp_config()

logger.debug(f"SMTP Configuration: SERVER={smtp_config['server']}, PORT={smtp_config['port']}, USERNAME={smtp_config['username']}, FROM_EMAIL={smtp_config['from_email']}")

def verify_smtp_settings():
    try:
        logger.info(f"Verifying SMTP settings: {smtp_config['server']}:{smtp_config['port']}")
        logger.info(f"Attempting DNS resolution of {smtp_config['server']}")
        ip_address = socket.gethostbyname(smtp_config['server'])
        logger.info(f"DNS resolution successful. IP: {ip_address}")
        socket.create_connection((smtp_config['server'], smtp_config['port']), timeout=10)
        logger.info("SMTP server is reachable")
        return True
    except socket.gaierror as e:
        logger.error(f"DNS resolution failed for {smtp_config['server']}: {e}")
    except Exception as e:
        logger.error(f"Failed to connect to SMTP server: {e}")
    return False

def send_email(to_email, subject, body, max_retries=3, retry_delay=5):
    if not all([smtp_config['server'], smtp_config['port'], smtp_config['username'], smtp_config['password'], smtp_config['from_email']]):
        logger.error("SMTP configuration is incomplete. Please check your environment variables.")
        return False

    if not verify_smtp_settings():
        logger.error("SMTP settings verification failed. Cannot send email.")
        return False

    for attempt in range(max_retries):
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = to_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_config['server'], smtp_config['port']) as server:
                logger.info(f"Attempting to connect to SMTP server: {smtp_config['server']}:{smtp_config['port']}")
                server.ehlo()
                logger.info("EHLO sent")
                server.starttls(context=context)
                logger.info("TLS started")
                server.ehlo()
                logger.info("Second EHLO sent")
                server.login(smtp_config['username'], smtp_config['password'])
                logger.info("Logged in successfully")
                server.send_message(msg)
                logger.info("Email sent")

            logger.info(f"Email sent successfully to {to_email}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP Authentication failed on attempt {attempt + 1}: {e}")
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error on attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            logger.error(f"Failed to send email after {max_retries} attempts.")
    
    return False

def send_verification_email(to_email, verification_token):
    subject = "Verify Your Email for Resume Cupid"
    verification_link = f"{Config.BASE_URL}/verify?token={verification_token}"
    body = f"""
    Hello,

    Thank you for registering with Resume Cupid. Please click on the link below to verify your email address:

    {verification_link}

    If you didn't register for Resume Cupid, please ignore this email.

    Best regards,
    The Resume Cupid Team
    """
    logger.info(f"Sending verification email to {to_email}")
    return send_email(to_email, subject, body)

def send_password_reset_email(to_email, reset_token):
    subject = "Reset Your Password for Resume Cupid"
    reset_link = f"{Config.BASE_URL}/reset_password?token={reset_token}"
    body = f"""
    Hello,

    You have requested to reset your password for Resume Cupid. Please click on the link below to set a new password:

    {reset_link}

    If you didn't request a password reset, please ignore this email.

    Best regards,
    The Resume Cupid Team
    """
    logger.info(f"Sending password reset email to {to_email}")
    return send_email(to_email, subject, body)
