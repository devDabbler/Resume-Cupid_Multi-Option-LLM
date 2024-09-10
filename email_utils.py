import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from utils import get_logger
import logging
from config_settings import Config

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info(f"Using BASE_URL: {Config.BASE_URL}")

# Get SMTP configuration
smtp_config = Config.get_smtp_config()

logger.debug(f"SMTP Configuration: SERVER={smtp_config['server']}, PORT={smtp_config['port']}, USERNAME={smtp_config['username']}, FROM_EMAIL={smtp_config['from_email']}")

def send_email(to_email, subject, body):
    if not all([smtp_config['server'], smtp_config['port'], smtp_config['username'], smtp_config['password'], smtp_config['from_email']]):
        logger.error("SMTP configuration is incomplete. Please check your environment variables.")
        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_config['from_email']
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(smtp_config['server'], smtp_config['port']) as server:
            logger.info(f"Attempting to connect to SMTP server: {smtp_config['server']}:{smtp_config['port']}")
            server.ehlo()
            logger.info("EHLO sent")
            server.starttls()
            logger.info("TLS started")
            server.ehlo()
            logger.info("Second EHLO sent")
            server.login(smtp_config['username'], smtp_config['password'])
            logger.info("Logged in successfully")
            server.send_message(msg)
            logger.info("Email sent")

        logger.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}. Error: {str(e)}")
        return False

def send_verification_email(recipient_email, verification_token):
    from config_settings import Config  # Import the config settings

    sender_email = Config.FROM_EMAIL
    subject = "Verify Your Email"
    verification_link = f"{Config.BASE_URL}?action=verify&token={verification_token}"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject

    body = f"Please verify your email by clicking the link: {verification_link}"
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT) as server:
            server.starttls()
            server.login(Config.SMTP_USER, Config.SMTP_PASSWORD)
            server.sendmail(sender_email, recipient_email, message.as_string())
        logger.info(f"Verification email sent to {recipient_email}")
    except Exception as e:
        logger.error(f"Failed to send verification email to {recipient_email}: {e}")

def send_password_reset_email(recipient_email, reset_token):
    from config_settings import Config  # Import the config settings

    sender_email = Config.FROM_EMAIL
    subject = "Reset Your Password"
    reset_link = f"{Config.BASE_URL}?action=reset&token={reset_token}"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject

    body = f"Please reset your password by clicking the link: {reset_link}"
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT) as server:
            server.starttls()
            server.login(Config.SMTP_USER, Config.SMTP_PASSWORD)
            server.sendmail(sender_email, recipient_email, message.as_string())
        logger.info(f"Password reset email sent to {recipient_email}")
        return True  # Return True if email is sent successfully
    except Exception as e:
        logger.error(f"Failed to send password reset email to {recipient_email}: {e}")
        return False  # Return False if email sending fails
