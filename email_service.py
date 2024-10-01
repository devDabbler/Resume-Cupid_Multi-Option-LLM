import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import time
import socket
from config_settings import Config

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.smtp_config = Config.get_smtp_config()
        self.is_development = Config.ENVIRONMENT == 'development'

    def verify_smtp_settings(self):
        try:
            logger.info(f"Verifying SMTP settings: {self.smtp_config['server']}:{self.smtp_config['port']}")
            logger.info(f"Attempting DNS resolution of {self.smtp_config['server']}")
            ip_address = socket.gethostbyname(self.smtp_config['server'])
            logger.info(f"DNS resolution successful. IP: {ip_address}")
            socket.create_connection((self.smtp_config['server'], self.smtp_config['port']), timeout=10)
            logger.info("SMTP server is reachable")
            return True
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed for {self.smtp_config['server']}: {e}")
        except Exception as e:
            logger.error(f"Failed to connect to SMTP server: {e}")
        return False

    def send_email(self, to_email: str, subject: str, body: str, max_retries=3, retry_delay=5) -> bool:
        if self.is_development:
            logger.info(f"Development mode: Email would be sent to {to_email}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Body: {body}")
            return True

        if not self.verify_smtp_settings():
            logger.error("SMTP settings verification failed. Cannot send email.")
            return False

        for attempt in range(max_retries):
            try:
                msg = MIMEMultipart()
                msg['From'] = self.smtp_config['from_email']
                msg['To'] = to_email
                msg['Subject'] = subject
                msg.attach(MIMEText(body, 'plain'))

                context = ssl.create_default_context()
                with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                    logger.info(f"Attempting to connect to SMTP server: {self.smtp_config['server']}:{self.smtp_config['port']}")
                    server.ehlo()
                    server.starttls(context=context)
                    server.ehlo()
                    server.login(self.smtp_config['username'], self.smtp_config['password'])
                    server.send_message(msg)
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

    def send_verification_email(self, to_email: str, verification_token: str) -> bool:
        subject = "Verify your Resume Cupid account"
        verification_link = self.get_verification_link(verification_token)
        body = f"""
        Welcome to Resume Cupid!

        Please verify your email by clicking on this link:
        {verification_link}

        If you didn't register for Resume Cupid, please ignore this email.

        Best regards,
        The Resume Cupid Team
        """
        success = self.send_email(to_email, subject, body)
        if success:
            logger.info(f"Verification email sent successfully to {to_email}")
        else:
            logger.error(f"Failed to send verification email to {to_email}")
        return success

    def send_password_reset_email(self, to_email: str, reset_token: str) -> bool:
        subject = "Reset your Resume Cupid password"
        reset_link = f"{Config.BASE_URL}/reset_password?token={reset_token}"
        body = f"""
        Hello,

        You have requested to reset your password for Resume Cupid. Please click on the link below to set a new password:

        {reset_link}

        If you didn't request a password reset, please ignore this email.

        Best regards,
        The Resume Cupid Team
        """
        success = self.send_email(to_email, subject, body)
        if success:
            logger.info(f"Password reset email sent successfully to {to_email}")
        else:
            logger.error(f"Failed to send password reset email to {to_email}")
        return success

    def get_verification_link(self, verification_token: str) -> str:
        return f"{Config.BASE_URL}/verify_email?token={verification_token}"

email_service = EmailService()
