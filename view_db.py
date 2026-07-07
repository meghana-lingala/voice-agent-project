import sqlite3

def view_logs():
    conn = sqlite3.connect("voice_agent.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=== conversation_logs table ===")
    try:
        cursor.execute("SELECT * FROM conversation_logs")
        rows = cursor.fetchall()
        if not rows:
            print("No interactions logged yet.")
        for row in rows:
            print(f"[{row['timestamp']}] Session: {row['session_id']}")
            print(f"  User : {row['user_transcript']}")
            print(f"  Agent: {row['ai_response']}")
            print(f"  Ref  : {row['tracking_id_ref']} | Latency: {row['api_latency_ms']}ms")
            print("-" * 50)
    except sqlite3.OperationalError as e:
        print(f"Error querying table: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    view_logs()
