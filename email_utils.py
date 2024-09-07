import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def send_verification_email(to_email, verification_token):
    # Email configuration
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT'))
    smtp_username = os.getenv('SMTP_USERNAME')
    smtp_password = os.getenv('SMTP_PASSWORD')
    from_email = os.getenv('FROM_EMAIL')
    subject = 'Email Verification'
    production_url = os.getenv('PRODUCTION_URL')  # Add this environment variable to your .env file
    
    # Email content
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = subject
    
    body = f"""
    Please verify your email by clicking the link below:
    {production_url}/verify?token={verification_token}
    """
    message.attach(MIMEText(body, 'plain'))
    
    try:
        # Connect to the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        
        # Send the email
        server.sendmail(from_email, to_email, message.as_string())
        server.quit()
        print("Verification email sent successfully")
    except Exception as e:
        print(f"Failed to send verification email: {e}")

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

# Example usage
to_email = 'user@example.com'
verification_token = 'your_verification_token'
send_verification_email(to_email, verification_token)
