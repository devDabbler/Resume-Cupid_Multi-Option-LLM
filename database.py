import sqlite3
import logging
from typing import List, Dict, Any, Optional
from config_settings import Config

logger = logging.getLogger(__name__)

def get_db_connection():
    try:
        conn = sqlite3.connect(Config.DB_PATH)
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
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        if conn:
            conn.close()

def register_user(username: str, email: str, password_hash: str) -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO users (username, email, password_hash)
        VALUES (?, ?, ?)
        ''', (username, email, password_hash))
        conn.commit()
        logger.info(f"User registered successfully: {username}")
        return True
    except sqlite3.IntegrityError as ie:
        logger.error(f"IntegrityError registering user {username}: {str(ie)}")
        return False
    except sqlite3.Error as e:
        logger.error(f"Error registering user {username}: {str(e)}")
        return False
    finally:
        conn.close()

def get_user(username: str) -> Optional[Dict[str, Any]]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cur.fetchone()
        return dict(user) if user else None
    except sqlite3.Error as e:
        logger.error(f"Error retrieving user {username}: {str(e)}")
        return None
    finally:
        conn.close()

def save_role(role_name: str, job_description: str) -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO saved_roles (role_name, job_description)
        VALUES (?, ?)
        ''', (role_name, job_description))
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Error saving role: {e}")
        return False
    finally:
        conn.close()

def get_saved_roles() -> List[Dict[str, Any]]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM saved_roles')
        roles = cur.fetchall()
        return [dict(role) for role in roles]
    except sqlite3.Error as e:
        logger.error(f"Error retrieving saved roles: {e}")
        return []
    finally:
        conn.close()

def save_evaluation_result(resume_file_name: str, job_role_id: int, match_score: int, recommendation: str) -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO evaluation_results (resume_file_name, job_role_id, match_score, recommendation)
        VALUES (?, ?, ?, ?)
        ''', (resume_file_name, job_role_id, match_score, recommendation))
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Error saving evaluation result: {e}")
        return False
    finally:
        conn.close()

def get_evaluation_results(job_role_id: int) -> List[Dict[str, Any]]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
        SELECT * FROM evaluation_results WHERE job_role_id = ?
        ORDER BY match_score DESC
        ''', (job_role_id,))
        results = cur.fetchall()
        return [dict(result) for result in results]
    except sqlite3.Error as e:
        logger.error(f"Error retrieving evaluation results: {e}")
        return []
    finally:
        conn.close()

# Initialize the database when this module is imported
init_db()