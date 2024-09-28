import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import re
from config_settings import Config

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.smtp_config = Config.get_smtp_config()
        self.is_development = Config.ENVIRONMENT == 'development'

    def send_email(self, to_email: str, subject: str, body: str) -> bool:
        if self.is_development:
            logger.info(f"Development mode: Email would be sent to {to_email}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Body: {body}")
            return True

        message = MIMEMultipart()
        message['From'] = self.smtp_config['from_email']
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        try:
            with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                server.starttls()
                server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(message)
            logger.info(f"Email sent successfully to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False

    @staticmethod
    def is_valid_email(email: str) -> bool:
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_regex, email) is not None

    def send_verification_email(self, to_email: str, verification_token: str) -> bool:
        subject = "Verify your Resume Cupid account"
        verification_link = self.get_verification_link(verification_token)
        body = f"Welcome to Resume Cupid! Please verify your email by clicking on this link: {verification_link}"
        return self.send_email(to_email, subject, body)

    def send_password_reset_email(self, to_email: str, reset_token: str) -> bool:
        subject = "Reset your Resume Cupid password"
        reset_link = f"{Config.BASE_URL}/reset_password?token={reset_token}"
        body = f"Click the following link to reset your password: {reset_link}"
        return self.send_email(to_email, subject, body)

    def get_verification_link(self, verification_token: str) -> str:
        return f"{Config.BASE_URL}/verify_email?token={verification_token}"

email_service = EmailService()