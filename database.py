import sqlite3
import threading
import json
import logging
import bcrypt  # Import bcrypt for password hashing
from typing import Any, Dict, List, Optional
from queue import Queue, Empty
from contextlib import contextmanager
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self.connection_queue = Queue(maxsize=max_connections)
        self.lock = threading.Lock()

    def get_connection(self):
        try:
            connection = self.connection_queue.get(block=False)
            if self._is_connection_valid(connection):
                return connection
            else:
                return self._create_connection()
        except Empty:
            if self.connection_queue.qsize() < self.max_connections:
                return self._create_connection()
            return self.connection_queue.get()

    def release_connection(self, connection):
        if self._is_connection_valid(connection):
            self.connection_queue.put(connection)
        else:
            connection.close()
            self.connection_queue.put(self._create_connection())

    def _create_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _is_connection_valid(self, connection):
        try:
            connection.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

db_pool = DatabaseConnectionPool('resume_cupid.db')

@contextmanager
def get_db_connection():
    connection = db_pool.get_connection()
    try:
        yield connection
    finally:
        db_pool.release_connection(connection)

def execute_with_retry(func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            with get_db_connection() as conn:
                return func(conn, *args, **kwargs)
        except sqlite3.Error as e:
            logger.error(f"Database error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                raise

# Define the reset_user_password function
def reset_user_password(reset_token: str, new_password: str) -> bool:
    def _reset_password(conn):
        cur = conn.cursor()
        current_time = datetime.utcnow()
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        cur.execute('''
        UPDATE users
        SET password_hash = ?, reset_token = NULL, reset_token_expiration = NULL
        WHERE reset_token = ? AND reset_token_expiration > ?
        ''', (hashed_password, reset_token, current_time))
        conn.commit()
        if cur.rowcount > 0:
            logger.info(f"Password updated successfully for reset token: {reset_token}")
            return True
        else:
            logger.warning(f"No user found or token expired for reset token: {reset_token}")
            return False

    try:
        return execute_with_retry(_reset_password)
    except Exception as e:
        logger.error(f"Error resetting password with token: {str(e)}")
        return False

def init_db(conn):
    try:
        cur = conn.cursor()
        
        # Users table (unchanged)
        cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            user_type TEXT NOT NULL,
            is_verified BOOLEAN DEFAULT 0,
            verification_token TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            profile TEXT,
            UNIQUE(username, user_type),
            UNIQUE(email, user_type)
        )
        ''')

        # Updated Saved roles table to include client field
        cur.execute('''
        CREATE TABLE IF NOT EXISTS saved_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role_name TEXT NOT NULL,
            client TEXT NOT NULL,
            job_description TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Evaluation results table (unchanged)
        cur.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_file_name TEXT NOT NULL,
            job_role_id INTEGER,
            match_score INTEGER,
            recommendation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_role_id) REFERENCES saved_roles (id)
        )
        ''')

        # User resumes table (unchanged)
        cur.execute('''
        CREATE TABLE IF NOT EXISTS user_resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')

        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def register_user(username: str, email: str, password_hash: bytes, user_type: str, verification_token: str) -> bool:
    def _register(conn):
        cur = conn.cursor()
        try:
            cur.execute('''
            INSERT INTO users (username, email, password_hash, user_type, profile, verification_token, is_verified)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, user_type, json.dumps({}), verification_token, False))
            conn.commit()
            logger.info(f"{user_type.capitalize()} registered successfully: {username}")
            return True
        except sqlite3.IntegrityError as e:
            logger.error(f"IntegrityError while registering {user_type} {username}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while registering {user_type} {username}: {str(e)}")
            return False

    try:
        return execute_with_retry(_register)
    except Exception as e:
        logger.error(f"Error registering {user_type} {username}: {str(e)}")
        return False

def update_user_reset_token(user_id: int, reset_token: str, expiration_time: datetime) -> bool:
    def _update_token(conn):
        cur = conn.cursor()
        cur.execute('''
        UPDATE users
        SET reset_token = ?, reset_token_expiration = ?
        WHERE id = ?
        ''', (reset_token, expiration_time, user_id))
        conn.commit()
        return cur.rowcount > 0

    try:
        return execute_with_retry(_update_token)
    except Exception as e:
        logger.error(f"Error updating reset token for user {user_id}: {str(e)}")
        return False

def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    logger.info(f"Attempting to retrieve user with token: {token}")
    def _get_user(conn):
        cur = conn.cursor()
        cur.execute('''
        SELECT * FROM users
        WHERE verification_token = ?
        ''', (token,))
        user = cur.fetchone()
        if user:
            logger.info(f"User found with token: {token}")
            return dict(zip([column[0] for column in cur.description], user))
        logger.warning(f"No user found with token: {token}")
        return None

    try:
        return execute_with_retry(_get_user)
    except Exception as e:
        logger.error(f"Error retrieving user by token: {str(e)}", exc_info=True)
        return None
    
def get_all_verification_tokens() -> List[str]:
    def _get_tokens(conn):
        cur = conn.cursor()
        cur.execute('SELECT verification_token FROM users WHERE verification_token IS NOT NULL')
        tokens = cur.fetchall()
        return [token[0] for token in tokens if token[0]]

    try:
        return execute_with_retry(_get_tokens)
    except Exception as e:
        logger.error(f"Error retrieving verification tokens: {str(e)}", exc_info=True)
        return []
    
def get_user_by_reset_token(reset_token: str) -> Optional[dict]:
    def _get_user(conn):
        cur = conn.cursor()
        current_time = datetime.utcnow()
        cur.execute('''
            SELECT id, username, email
            FROM users
            WHERE reset_token = ? AND reset_token_expiration > ?
        ''', (reset_token, current_time))
        user = cur.fetchone()
        if user:
            logger.info(f"Found user with ID {user['id']} for reset token")
            return dict(user)
        else:
            logger.warning(f"No user found for reset token {reset_token}")
            return None

    try:
        return execute_with_retry(_get_user)
    except Exception as e:
        logger.error(f"Error retrieving user by reset token: {str(e)}")
        return None

def clear_reset_token(user_id: int) -> bool:
    try:
        def _clear_token(conn):
            cur = conn.cursor()
            cur.execute('''
                UPDATE users
                SET reset_token = NULL, reset_token_expiration = NULL
                WHERE id = ?
            ''', (user_id,))
            conn.commit()
            return cur.rowcount > 0

        success = execute_with_retry(_clear_token)

        if success:
            logger.info(f"Reset token cleared for user {user_id}")
        else:
            logger.warning(f"No user found with ID {user_id} when clearing reset token")

        return success
    except Exception as e:
        logger.error(f"Error clearing reset token for user {user_id}: {str(e)}")
        return False

def get_user(username: str, user_type: str) -> Optional[Dict[str, Any]]:
    def _get_user(conn):
        cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE username = ? AND user_type = ?', (username, user_type))
        user = cur.fetchone()
        if user:
            return dict(zip([column[0] for column in cur.description], user))
        return None

    try:
        return execute_with_retry(_get_user)
    except Exception as e:
        logger.error(f"Error retrieving {user_type} {username}: {str(e)}")
        return None

def get_user_by_email(email: str, user_type: str) -> Optional[Dict[str, Any]]:
    def _get_user_by_email(conn):
        cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE email = ? AND user_type = ?', (email, user_type))
        user = cur.fetchone()
        if user:
            return dict(zip([column[0] for column in cur.description], user))
        return None

    try:
        return execute_with_retry(_get_user_by_email)
    except Exception as e:
        logger.error(f"Error retrieving {user_type} by email {email}: {str(e)}")
        return None

def get_user_profile(user_id: int) -> Optional[Dict[str, Any]]:
    def _get_user_profile(conn):
        cur = conn.cursor()
        cur.execute('''
        SELECT profile
        FROM users
        WHERE id = ?
        ''', (user_id,))
        result = cur.fetchone()
        if result and result[0]:
            return json.loads(result[0])
        return None

    try:
        return execute_with_retry(_get_user_profile)
    except Exception as e:
        logger.error(f"Error retrieving user profile for user {user_id}: {str(e)}")
        return None

def update_user_profile(user_id: int, profile: Dict[str, Any]) -> bool:
    def _update_user_profile(conn):
        cur = conn.cursor()
        profile_json = json.dumps(profile)
        cur.execute('''
        UPDATE users
        SET profile = ?
        WHERE id = ?
        ''', (profile_json, user_id))
        conn.commit()
        return cur.rowcount > 0

    try:
        return execute_with_retry(_update_user_profile)
    except Exception as e:
        logger.error(f"Error updating user profile for user {user_id}: {str(e)}")
        return False

def save_user_resume(user_id: int, file_name: str, content: str) -> bool:
    def _save_resume(conn):
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO user_resumes (user_id, file_name, content)
        VALUES (?, ?, ?)
        ''', (user_id, file_name, content))
        conn.commit()
        return cur.rowcount > 0

    try:
        return execute_with_retry(_save_resume)
    except Exception as e:
        logger.error(f"Error saving resume for user {user_id}: {str(e)}")
        return False

def get_user_resumes(user_id: int) -> List[Dict[str, Any]]:
    def _get_resumes(conn):
        cur = conn.cursor()
        cur.execute('''
        SELECT id, file_name, content, created_at
        FROM user_resumes
        WHERE user_id = ?
        ORDER BY created_at DESC
        ''', (user_id,))
        resumes = cur.fetchall()
        return [dict(zip([column[0] for column in cur.description], resume)) for resume in resumes]

    try:
        return execute_with_retry(_get_resumes)
    except Exception as e:
        logger.error(f"Error retrieving resumes for user {user_id}: {str(e)}")
        return []

def get_job_recommendations(user_id: int) -> List[Dict[str, Any]]:
    def _get_job_recommendations(conn):
        cur = conn.cursor()
        # This is a placeholder implementation. In a real-world scenario,
        # you would implement a more sophisticated recommendation algorithm.
        cur.execute('''
        SELECT id, role_name as title, 'Company Name' as company, job_description as description,
               '0-100000' as salary_range, 'Remote' as location, 80 as match_score
        FROM saved_roles
        LIMIT 5
        ''')
        jobs = cur.fetchall()
        return [dict(zip([column[0] for column in cur.description], job)) for job in jobs]

    try:
        recommendations = execute_with_retry(_get_job_recommendations)
        for job in recommendations:
            job['required_skills'] = ['Skill 1', 'Skill 2', 'Skill 3']  # Placeholder
        return recommendations
    except Exception as e:
        logger.error(f"Error retrieving job recommendations for user {user_id}: {str(e)}")
        return []

def get_latest_evaluation(user_id: int) -> Optional[Dict[str, Any]]:
    def _get_latest_evaluation(conn):
        cur = conn.cursor()
        cur.execute('''
        SELECT er.id, er.resume_file_name, er.match_score, er.recommendation, er.created_at,
               er.skills_gap, er.areas_for_improvement
        FROM evaluation_results er
        JOIN users u ON u.id = ?
        ORDER BY er.created_at DESC
        LIMIT 1
        ''', (user_id,))
        result = cur.fetchone()
        if result:
            return {
                'id': result[0],
                'resume_file_name': result[1],
                'match_score': result[2],
                'recommendation': result[3],
                'created_at': result[4],
                'skills_gap': json.loads(result[5]) if result[5] else [],
                'areas_for_improvement': json.loads(result[6]) if result[6] else []
            }
        return None

    try:
        return execute_with_retry(_get_latest_evaluation)
    except Exception as e:
        logger.error(f"Error retrieving latest evaluation for user {user_id}: {str(e)}")
        return None

def verify_user_email(token: str) -> bool:
    logger.info(f"Attempting to verify email with token: {token}")
    all_tokens = get_all_verification_tokens()
    logger.info(f"All verification tokens in database: {all_tokens}")
    logger.info(f"Attempting to verify email with token: {token}")
    try:
        user = get_user_by_token(token)
        if user:
            logger.info(f"User found for token: {token}")
            # Update user record to mark the email as verified
            user['email_verified'] = True
            save_user(user)
            logger.info(f"Email verified successfully for user ID: {user['id']}")
            return True
        else:
            logger.warning(f"No user found for token: {token}")
            return False
    except Exception as e:
        logger.error(f"Error verifying email for token {token}: {e}", exc_info=True)
        return False

def is_user_email_verified(user_id: int) -> bool:
    def _check_verification(conn):
        cur = conn.cursor()
        cur.execute('SELECT is_verified FROM users WHERE id = ?', (user_id,))
        result = cur.fetchone()
        return result[0] if result else False

    try:
        return execute_with_retry(_check_verification)
    except Exception as e:
        logger.error(f"Error checking email verification status for user {user_id}: {str(e)}")
        return False
    
def update_password_with_token(reset_token: str, new_password: str) -> bool:
    def _update_password(conn):
        cur = conn.cursor()
        current_time = datetime.utcnow()
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        cur.execute('''
        UPDATE users
        SET password_hash = ?, reset_token = NULL, reset_token_expiration = NULL
        WHERE reset_token = ? AND reset_token_expiration > ?
        ''', (hashed_password, reset_token, current_time))
        conn.commit()
        if cur.rowcount > 0:
            logger.info(f"Password updated successfully for token {reset_token}")
            return True
        else:
            logger.warning(f"No user found or token expired for reset token {reset_token}")
            return False

    try:
        return execute_with_retry(_update_password)
    except Exception as e:
        logger.error(f"Error updating password with token: {str(e)}")
        return False
     
def update_user_password(user_id: int, new_password_hash: bytes, user_type: str) -> bool:
    def _update_password(conn):
        cur = conn.cursor()
        cur.execute('''
        UPDATE users SET password_hash = ? WHERE id = ? AND user_type = ?
        ''', (new_password_hash, user_id, user_type))
        conn.commit()
        return cur.rowcount > 0

    try:
        return execute_with_retry(_update_password)
    except Exception as e:
        logger.error(f"Error updating password for {user_type} {user_id}: {str(e)}")
        return False

def get_users_by_type(user_type: str) -> List[Dict[str, Any]]:
    def _get_users_by_type(conn):
        cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE user_type = ?', (user_type,))
        users = cur.fetchall()
        return [dict(zip([column[0] for column in cur.description], user)) for user in users]

    try:
        return execute_with_retry(_get_users_by_type)
    except Exception as e:
        logger.error(f"Error retrieving {user_type}s: {str(e)}")
        return []
    
def save_role(role_name: str, client: str, job_description: str) -> bool:
    def _save_role(conn):
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO saved_roles (role_name, client, job_description)
        VALUES (?, ?, ?)
        ''', (role_name, client, job_description))
        conn.commit()
        logger.info(f"Role saved successfully: {role_name} for client: {client}")
        return True

    try:
        return execute_with_retry(_save_role)
    except Exception as e:
        logger.error(f"Error saving role: {e}")
        return False

def get_saved_roles() -> List[Dict[str, Any]]:
    def _get_saved_roles(conn):
        cur = conn.cursor()
        cur.execute('SELECT * FROM saved_roles')
        roles = cur.fetchall()
        return [dict(zip([column[0] for column in cur.description], role)) for role in roles]

    try:
        return execute_with_retry(_get_saved_roles)
    except Exception as e:
        logger.error(f"Error retrieving saved roles: {e}")
        return []
    
def delete_role(role_id: int) -> bool:
    def _delete_role(conn):
        cur = conn.cursor()
        cur.execute('''
        DELETE FROM saved_roles
        WHERE id = ?
        ''', (role_id,))
        conn.commit()
        return cur.rowcount > 0

    try:
        return execute_with_retry(_delete_role)
    except Exception as e:
        logger.error(f"Error deleting role with ID {role_id}: {e}")
        return False

def save_evaluation_result(resume_file_name: str, job_role_id: int, match_score: int, recommendation: str) -> bool:
    def _save_evaluation_result(conn):
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO evaluation_results (resume_file_name, job_role_id, match_score, recommendation)
        VALUES (?, ?, ?, ?)
        ''', (resume_file_name, job_role_id, match_score, recommendation))
        conn.commit()
        logger.info(f"Evaluation result saved successfully for resume: {resume_file_name}")
        return True

    try:
        return execute_with_retry(_save_evaluation_result)
    except Exception as e:
        logger.error(f"Error saving evaluation result: {e}")
        return False

def get_evaluation_results(job_role_id: int) -> List[Dict[str, Any]]:
    def _get_evaluation_results(conn):
        cur = conn.cursor()
        cur.execute('''
        SELECT id, resume_file_name, match_score, recommendation, created_at
        FROM evaluation_results
        WHERE job_role_id = ?
        ORDER BY created_at DESC
        ''', (job_role_id,))
        results = cur.fetchall()
        return [dict(zip([column[0] for column in cur.description], result)) for result in results]

    try:
        return execute_with_retry(_get_evaluation_results)
    except Exception as e:
        logger.error(f"Error retrieving evaluation results: {e}")
        return []

def ensure_db_initialized():
    def _init(conn):
        init_db(conn)

    try:
        execute_with_retry(_init)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")

ensure_db_initialized()