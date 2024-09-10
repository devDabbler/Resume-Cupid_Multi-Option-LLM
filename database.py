import sqlite3
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime, timedelta
from typing import Optional
from logger import get_logger
import bcrypt
from config_settings import Config

logger = get_logger(__name__)

DB_PATH = Config.DB_PATH

def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Create users table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_verified BOOLEAN DEFAULT 0,
            verification_token TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_admin INTEGER DEFAULT 0
        )
        ''')
        
        # Create reset_tokens table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL
        )
        ''')
        
        # Create saved_roles table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS saved_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role_name TEXT NOT NULL,
            client TEXT,
            job_description TEXT NOT NULL,
            job_description_link TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create feedback table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            resume_id TEXT NOT NULL,
            accuracy_rating INTEGER,
            content_rating INTEGER,
            suggestions TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create run_logs table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS run_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            event TEXT NOT NULL,
            details TEXT
        )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        if conn:
            conn.close()

def register_user(username: str, email: str, password: str) -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        verification_token = str(uuid.uuid4())
        logger.debug(f"Attempting to register user: {username}, email: {email}")
        cur.execute('''
        INSERT INTO users (username, email, password_hash, verification_token)
        VALUES (?, ?, ?, ?)
        ''', (username, email, hashed_password, verification_token))
        conn.commit()
        logger.info(f"User registered successfully: {username}")
        
        # Verify the user was actually inserted
        cur.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cur.fetchone()
        if user:
            logger.debug(f"User found in database after registration: {user['username']}")
        else:
            logger.error(f"User not found in database after registration: {username}")
        
        return True
    except sqlite3.IntegrityError as ie:
        logger.error(f"IntegrityError registering user {username}: {str(ie)}")
        return False
    except sqlite3.Error as e:
        logger.error(f"Error registering user {username}: {str(e)}")
        return False
    finally:
        conn.close()

def verify_user(verification_token: str) -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
        UPDATE users SET is_verified = 1, verification_token = NULL
        WHERE verification_token = ?
        ''', (verification_token,))
        conn.commit()
        affected_rows = cur.rowcount
        logger.debug(f"Rows affected by verify_user: {affected_rows}")
        
        if affected_rows > 0:
            # Check if the update was successful
            cur.execute('SELECT username FROM users WHERE is_verified = 1 AND verification_token IS NULL')
            verified_user = cur.fetchone()
            if verified_user:
                logger.info(f"User {verified_user['username']} successfully verified")
            else:
                logger.error("User verification failed: No user found with updated status")
        
        return affected_rows > 0
    except sqlite3.Error as e:
        logger.error(f"Error verifying user: {e}")
        return False
    finally:
        conn.close()

def is_user_verified(username: str) -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT is_verified FROM users WHERE username = ?', (username,))
        result = cur.fetchone()
        verified = bool(result['is_verified']) if result else False
        logger.debug(f"User {username} verification status: {verified}")
        return verified
    except sqlite3.Error as e:
        logger.error(f"Error checking user verification status: {e}")
        return False
    finally:
        conn.close()

def authenticate_user(username: str, password: str) -> Optional[dict]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        logger.info(f"Executing query to fetch user: {username}")
        cur.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cur.fetchone()
        if user:
            logger.info(f"User found: {user['username']}")
        else:
            logger.info("User not found")
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
            logger.info("Password match successful")
            return dict(user)
        else:
            logger.info("Password match failed")
        return None
    except sqlite3.Error as e:
        logger.error(f"Error authenticating user: {e}")
        return None
    finally:
        conn.close()

def set_reset_token(email: str) -> Optional[str]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        reset_token = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=1)  # Token expires in 1 hour
        cur.execute('''
        INSERT INTO reset_tokens (email, token, expires_at)
        VALUES (?, ?, ?)
        ''', (email, reset_token, expires_at))
        conn.commit()
        return reset_token
    except sqlite3.Error as e:
        logger.error(f"Error setting reset token: {e}")
        return None
    finally:
        conn.close()

def reset_password(token: str, new_password: str) -> bool:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        logger.debug(f"Attempting to reset password with token: {token}")

        # Verify the token
        cursor.execute("SELECT email FROM reset_tokens WHERE token = ? AND expires_at > datetime('now')", (token,))
        result = cursor.fetchone()

        if not result:
            logger.error(f"Invalid or expired token: {token}")
            return False

        email = result['email']
        logger.debug(f"Valid token found for email: {email}")

        # Hash the new password
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

        # Update the user's password
        cursor.execute("UPDATE users SET password_hash = ? WHERE email = ?", (hashed_password, email))
        
        if cursor.rowcount == 0:
            logger.error(f"No user found with email: {email}")
            return False

        # Delete the used token
        cursor.execute("DELETE FROM reset_tokens WHERE token = ?", (token,))
        
        conn.commit()
        logger.info(f"Password reset successful for email: {email}")

        # Verify the password was actually updated
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        if user:
            logger.debug(f"User found after password reset: {user['username']}")
        else:
            logger.error(f"User not found after password reset for email: {email}")

        return True
    except Exception as e:
        logger.error(f"Failed to reset password: {str(e)}")
        return False
    finally:
        conn.close()

def save_role(role_name: str, client: str, job_description: str, job_description_link: str) -> bool:
    if not job_description:
        raise ValueError("Job description is required to save a role.")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO saved_roles (role_name, client, job_description, job_description_link)
        VALUES (?, ?, ?, ?)
        ''', (role_name, client, job_description, job_description_link))
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Error saving role: {e}")
        return False
    finally:
        conn.close()

def get_saved_roles():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM saved_roles')
        rows = cur.fetchall()
        
        saved_roles = []
        for row in rows:
            role = dict(row)
            saved_roles.append(role)
        return saved_roles
    except sqlite3.Error as e:
        logger.error(f"Error getting saved roles: {e}")
        raise
    finally:
        conn.close()

def delete_saved_role(role_name: str) -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM saved_roles WHERE role_name = ?', (role_name,))
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error as e:
        logger.error(f"Error deleting saved role: {e}")
        return False
    finally:
        conn.close()

def save_feedback(run_id: str, resume_id: str, accuracy_rating: int, content_rating: int, suggestions: str) -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO feedback (run_id, resume_id, accuracy_rating, content_rating, suggestions)
        VALUES (?, ?, ?, ?, ?)
        ''', (run_id, resume_id, accuracy_rating, content_rating, suggestions))
        conn.commit()
        logger.info(f"Feedback saved for run_id: {run_id}, resume_id: {resume_id}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Error saving feedback: {e}")
        return False
    finally:
        if conn:
            conn.close()

def insert_run_log(run_id: str, event: str, details: str = None) -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        timestamp = datetime.now().isoformat()
        cur.execute('''
        INSERT INTO run_logs (run_id, timestamp, event, details)
        VALUES (?, ?, ?, ?)
        ''', (run_id, timestamp, event, details))
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Error inserting run log: {e}")
        return False
    finally:
        conn.close()

def set_verification_token(email: str) -> Optional[str]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        verification_token = str(uuid.uuid4())
        cur.execute('''
        UPDATE users SET verification_token = ?
        WHERE email = ?
        ''', (verification_token, email))
        conn.commit()
        if cur.rowcount > 0:
            return verification_token
        return None
    except sqlite3.Error as e:
        logger.error(f"Error setting verification token: {e}")
        return None
    finally:
        conn.close()

def debug_db_contents():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check users table
        cur.execute("SELECT * FROM users")
        users = cur.fetchall()
        logger.debug(f"Users in database: {len(users)}")
        for user in users:
            logger.debug(f"User: {user['username']}, Email: {user['email']}, Verified: {user['is_verified']}")

        # Check reset_tokens table
        cur.execute("SELECT * FROM reset_tokens")
        tokens = cur.fetchall()
        logger.debug(f"Reset tokens in database: {len(tokens)}")
        for token in tokens:
            logger.debug(f"Token: {token['token']}, Email: {token['email']}, Expires: {token['expires_at']}")

    except sqlite3.Error as e:
        logger.error(f"Error checking database contents: {str(e)}")
    finally:
        conn.close()

# Add this to the end of the file
if __name__ == "__main__":
    print(f"Initializing database at: {DB_PATH}")
    init_db()  # Ensure tables are created
    print("Database initialization complete.")
    debug_db_contents()  # Check database contents after initialization