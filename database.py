import sqlite3
import threading
import logging
from typing import Any, Dict, List, Optional
from queue import Queue, Empty

logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self.connection_queue = Queue(maxsize=max_connections)
        self.lock = threading.Lock()

    def get_connection(self):
        try:
            return self.connection_queue.get(block=False)
        except Empty:
            if self.connection_queue.qsize() < self.max_connections:
                return self._create_connection()
            return self.connection_queue.get()

    def release_connection(self, connection):
        self.connection_queue.put(connection)

    def _create_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

db_pool = DatabaseConnectionPool('resume_cupid.db')

def get_db_connection():
    return db_pool.get_connection()

def release_db_connection(conn):
    db_pool.release_connection(conn)

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

def register_user(conn, username: str, email: str, password_hash: bytes) -> bool:
    if conn is None:
        logger.error("Cannot register user: database connection is None")
        return False
    
    try:
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO users (username, email, password_hash)
        VALUES (?, ?, ?)
        ''', (username, email, password_hash))
        conn.commit()
        logger.info(f"User registered successfully: {username}")
        return True
    except Exception as e:
        logger.error(f"Error registering user {username}: {str(e)}")
        return False

def get_user(conn, username: str) -> Optional[Dict[str, Any]]:
    if conn is None:
        logger.error("Cannot get user: database connection is None")
        return None
    
    try:
        cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cur.fetchone()
        if user:
            return dict(zip([column[0] for column in cur.description], user))
        return None
    except sqlite3.OperationalError as e:
        if "Cannot operate on a closed database" in str(e):
            logger.warning(f"Database connection closed unexpectedly: {str(e)}")
            return None
        raise
    except Exception as e:
        logger.error(f"Error retrieving user {username}: {str(e)}")
        return None

def get_user_by_email(conn, email: str) -> Optional[Dict[str, Any]]:
    if conn is None:
        logger.error("Cannot get user: database connection is None")
        return None
    
    try:
        cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cur.fetchone()
        if user:
            return dict(zip([column[0] for column in cur.description], user))
        return None
    except sqlite3.OperationalError as e:
        if "Cannot operate on a closed database" in str(e):
            logger.warning(f"Database connection closed unexpectedly: {str(e)}")
            return None
        raise
    except Exception as e:
        logger.error(f"Error retrieving user by email {email}: {str(e)}")
        return None

def save_role(conn=None, role_name: str = "", job_description: str = "") -> bool:
    if not conn:
        conn = get_db_connection()
    if conn is None:
        logger.error("Cannot save role: database connection is None")
        return False
    
    try:
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO saved_roles (role_name, job_description)
        VALUES (?, ?)
        ''', (role_name, job_description))
        conn.commit()
        logger.info(f"Role saved successfully: {role_name}")
        return True
    except Exception as e:
        logger.error(f"Error saving role: {e}")
        return False

def get_saved_roles(conn=None) -> List[Dict[str, Any]]:
    if not conn:
        conn = get_db_connection()
    if conn is None:
        logger.error("Cannot get saved roles: database connection is None")
        return []
    
    try:
        cur = conn.cursor()
        cur.execute('SELECT * FROM saved_roles')
        roles = cur.fetchall()
        return [dict(zip([column[0] for column in cur.description], role)) for role in roles]
    except Exception as e:
        logger.error(f"Error retrieving saved roles: {e}")
        return []

def save_evaluation_result(conn, resume_file_name: str, job_role_id: int, match_score: int, recommendation: str) -> bool:
    if conn is None:
        logger.error("Cannot save evaluation result: database connection is None")
        return False
    
    try:
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO evaluation_results (resume_file_name, job_role_id, match_score, recommendation)
        VALUES (?, ?, ?, ?)
        ''', (resume_file_name, job_role_id, match_score, recommendation))
        conn.commit()
        logger.info(f"Evaluation result saved successfully for resume: {resume_file_name}")
        return True
    except Exception as e:
        logger.error(f"Error saving evaluation result: {e}")
        return False

def get_evaluation_results(conn, job_role_id: int) -> List[Dict[str, Any]]:
    if conn is None:
        logger.error("Cannot get evaluation results: database connection is None")
        return []
    
    try:
        cur = conn.cursor()
        cur.execute('''
        SELECT id, resume_file_name, match_score, recommendation, created_at
        FROM evaluation_results
        WHERE job_role_id = ?
        ORDER BY created_at DESC
        ''', (job_role_id,))
        results = cur.fetchall()
        return [dict(zip([column[0] for column in cur.description], result)) for result in results]
    except Exception as e:
        logger.error(f"Error retrieving evaluation results: {e}")
        return []

def ensure_db_initialized():
    conn = get_db_connection()
    if conn:
        init_db(conn)
        release_db_connection(conn)
    else:
        logger.error("Failed to initialize database: could not establish connection")

ensure_db_initialized()