import sqlite3
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime
import logging
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

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
        
        # Create saved_roles table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS saved_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
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

def save_role(username, role_name, client, job_description, job_description_link):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO saved_roles (username, role_name, client, job_description, job_description_link)
        VALUES (?, ?, ?, ?, ?)
        ''', (username, role_name, client, job_description, job_description_link))
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Error saving role: {e}")
        return False
    finally:
        conn.close()

def get_saved_roles(username):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM saved_roles WHERE username = ?', (username,))
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

def delete_saved_role(username, role_name, force_delete=False):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        if force_delete:
            # Delete all roles for the given username
            cur.execute('DELETE FROM saved_roles WHERE username = ?', (username,))
        else:
            # Normal deletion
            cur.execute('DELETE FROM saved_roles WHERE username = ? AND role_name = ?', (username, role_name))
        
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error as e:
        logger.error(f"Error deleting saved role: {e}")
        return False
    finally:
        conn.close()

def save_feedback(run_id, resume_id, accuracy_rating, content_rating, suggestions):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO feedback (run_id, resume_id, accuracy_rating, content_rating, suggestions)
        VALUES (?, ?, ?, ?, ?)
        ''', (run_id, resume_id, accuracy_rating, content_rating, suggestions))
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Error saving feedback: {e}")
        return False
    finally:
        conn.close()
    
def insert_run_log(run_id: str, event: str, details: str = None):
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

if __name__ == "__main__":
    print(f"Initializing database at: {DB_PATH}")
    init_db()
    print("Database initialization complete.")
    


