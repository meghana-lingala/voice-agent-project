import sqlite3

def print_table(title, headers, rows):
    if not rows:
        print(f"\n=== {title} (Empty) ===")
        return

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val if val is not None else "")))

    # Construct separator line
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    
    print(f"\n=== {title} ===")
    print(sep)
    # Header
    print("|" + "|".join(f" {headers[i]:<{col_widths[i]}} " for i in range(len(headers))) + "|")
    print(sep)
    # Rows
    for row in rows:
        print("|" + "|".join(f" {str(val if val is not None else ''):<{col_widths[i]}} " for i, val in enumerate(row)) + "|")
    print(sep)

def view_all_tables():
    conn = sqlite3.connect("voice_agent.db")
    cursor = conn.cursor()
    
    # 1. Query shipment_tracking
    try:
        cursor.execute("SELECT * FROM shipment_tracking")
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        print_table("SHIPMENT TRACKING TABLE", columns, rows)
    except Exception as e:
        print(f"Error reading shipment_tracking: {e}")

    # 2. Query conversation_logs
    try:
        cursor.execute("SELECT id, session_id, timestamp, user_transcript, ai_response, tracking_id_ref FROM conversation_logs")
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        print_table("CONVERSATION LOGS TABLE (Selected Columns)", columns, rows)
    except Exception as e:
        print(f"Error reading conversation_logs: {e}")

    conn.close()

if __name__ == "__main__":
    view_all_tables()
