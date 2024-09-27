import sqlite3
import threading
import logging
from typing import Any, Dict, List, Optional
from queue import Queue, Empty
from contextlib import contextmanager

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

def init_db(conn):
    try:
        cur = conn.cursor()
        
        # Create users table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_verified BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create saved_roles table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS saved_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role_name TEXT NOT NULL,
            job_description TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create evaluation_results table
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
        
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def register_user(username: str, email: str, password_hash: bytes) -> bool:
    def _register(conn):
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO users (username, email, password_hash)
        VALUES (?, ?, ?)
        ''', (username, email, password_hash))
        conn.commit()
        logger.info(f"User registered successfully: {username}")
        return True

    try:
        return execute_with_retry(_register)
    except Exception as e:
        logger.error(f"Error registering user {username}: {str(e)}")
        return False

def get_user(username: str) -> Optional[Dict[str, Any]]:
    def _get_user(conn):
        cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cur.fetchone()
        if user:
            return dict(zip([column[0] for column in cur.description], user))
        return None

    try:
        return execute_with_retry(_get_user)
    except Exception as e:
        logger.error(f"Error retrieving user {username}: {str(e)}")
        return None

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    def _get_user_by_email(conn):
        cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cur.fetchone()
        if user:
            return dict(zip([column[0] for column in cur.description], user))
        return None

    try:
        return execute_with_retry(_get_user_by_email)
    except Exception as e:
        logger.error(f"Error retrieving user by email {email}: {str(e)}")
        return None

def save_role(role_name: str, job_description: str) -> bool:
    def _save_role(conn):
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO saved_roles (role_name, job_description)
        VALUES (?, ?)
        ''', (role_name, job_description))
        conn.commit()
        logger.info(f"Role saved successfully: {role_name}")
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