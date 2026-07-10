import sqlite3
import threading
from datetime import datetime
import os

DB_FILE = "voice_agent.db"
db_lock = threading.Lock()

def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the shipment_tracking and conversation_logs tables in the SQLite database.
    Prepopulates the shipment_tracking table with 3 mock profiles if they do not exist.
    """
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            # Create shipment_tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shipment_tracking (
                    tracking_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    current_location TEXT NOT NULL,
                    eta_days INTEGER NOT NULL,
                    pickup_address TEXT,
                    destination_address TEXT,
                    package_weight TEXT,
                    pickup_time TEXT
                )
            """)

            # Create conversation_logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    audio_filename TEXT,
                    user_transcript TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    detected_language TEXT,
                    engine_used TEXT,
                    api_latency_ms INTEGER,
                    tracking_id_ref TEXT,
                    FOREIGN KEY (tracking_id_ref) REFERENCES shipment_tracking(tracking_id)
                )
            """)

            # Insert mock tracking profiles if they do not exist
            mock_profiles = [
                ('SH123', 'In Transit', 'Hyderabad', 1, 'Hitech City Office', 'Delhi Hub', '1.5 kg', 'Completed'),
                ('SH456', 'Out for Delivery', 'Ghatkesar', 0, 'Secunderabad Depot', 'Ghatkesar Residential', '0.8 kg', 'Completed'),
                ('SH789', 'Delayed', 'Delhi Hub', 4, 'Mumbai Port', 'Gachibowli Tech Park', '12.0 kg', 'Completed')
            ]

            for profile in mock_profiles:
                cursor.execute("SELECT 1 FROM shipment_tracking WHERE tracking_id = ?", (profile[0],))
                if not cursor.fetchone():
                    cursor.execute("""
                        INSERT INTO shipment_tracking (
                            tracking_id, status, current_location, eta_days, 
                            pickup_address, destination_address, package_weight, pickup_time
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, profile)

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

def get_shipment_details(tracking_id: str) -> dict | None:
    """
    Queries the shipment_tracking table for the given tracking_id.
    Returns a dictionary of column names and values, or None if not found.
    """
    # Canonicalize tracking_id to uppercase
    t_id = tracking_id.strip().upper()
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM shipment_tracking WHERE UPPER(tracking_id) = ?", (t_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

def insert_new_order(tracking_id: str, p_addr: str, d_addr: str, weight: str, p_time: str) -> bool:
    """
    Inserts a brand new shipping record into shipment_tracking.
    Returns True if successful, False otherwise.
    """
    # Default status to 'Created' and current_location to 'Origin Depot', ETA to 5 days
    status = "Created"
    current_location = "Origin Depot"
    eta_days = 5
    t_id = tracking_id.strip().upper()

    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO shipment_tracking (
                    tracking_id, status, current_location, eta_days,
                    pickup_address, destination_address, package_weight, pickup_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (t_id, status, current_location, eta_days, p_addr, d_addr, weight, p_time))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # If tracking_id already exists
            conn.rollback()
            return False
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

def log_interaction(session_id: str, audio_file: str, transcript: str, response: str, lang: str, engine: str, latency: int, tracking_id: str = None):
    """
    Writes a transaction entry to the conversation_logs table.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_id_ref = tracking_id.strip().upper() if tracking_id else None

    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO conversation_logs (
                    session_id, timestamp, audio_filename, user_transcript,
                    ai_response, detected_language, engine_used, api_latency_ms, tracking_id_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, timestamp, audio_file, transcript, response, lang, engine, latency, t_id_ref))
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

def get_conversation_history(session_id: str, limit: int = 10) -> list:
    """
    Retrieves the last `limit` messages of conversation history for the given session_id.
    Returns a list of dictionaries with 'user' and 'assistant' roles.
    """
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT user_transcript, ai_response FROM conversation_logs
                WHERE session_id = ?
                ORDER BY id ASC
            """, (session_id,))
            rows = cursor.fetchall()
            history = []
            for row in rows:
                history.append({"role": "user", "content": row["user_transcript"]})
                history.append({"role": "assistant", "content": row["ai_response"]})
            return history[-limit:]
        finally:
            conn.close()


def cancel_shipment_order(tracking_id: str) -> str:
    """
    Cancels a shipment order if the status is 'Created'.
    Returns:
    - 'NOT_FOUND' if tracking_id does not exist
    - 'ALREADY_CANCELLED' if status is exactly 'Cancelled'
    - 'SUCCESS' if status is exactly 'Created' (status updated to 'Cancelled')
    - 'ALREADY_PICKED_UP' for any other status
    """
    t_id = tracking_id.strip().upper()
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT status FROM shipment_tracking WHERE UPPER(tracking_id) = ?", (t_id,))
            row = cursor.fetchone()
            if not row:
                return "NOT_FOUND"
            status = row["status"]
            if status == "Cancelled":
                return "ALREADY_CANCELLED"
            if status == "Created":
                cursor.execute("UPDATE shipment_tracking SET status = 'Cancelled' WHERE UPPER(tracking_id) = ?", (t_id,))
                conn.commit()
                return "SUCCESS"
            return "ALREADY_PICKED_UP"
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()


