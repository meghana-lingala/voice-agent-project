import sqlite3
import csv

def export_to_csv():
    conn = sqlite3.connect("voice_agent.db")
    cursor = conn.cursor()
    
    # 1. Export shipment_tracking to CSV
    try:
        cursor.execute("SELECT * FROM shipment_tracking")
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        with open("shipment_tracking.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        print("Successfully exported 'shipment_tracking' table to: shipment_tracking.csv")
    except Exception as e:
        print(f"Error exporting shipment_tracking: {e}")

    # 2. Export conversation_logs to CSV
    try:
        cursor.execute("SELECT * FROM conversation_logs")
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        with open("conversation_logs.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        print("Successfully exported 'conversation_logs' table to: conversation_logs.csv")
    except Exception as e:
        print(f"Error exporting conversation_logs: {e}")
        
    conn.close()

def export_to_markdown():
    conn = sqlite3.connect("voice_agent.db")
    cursor = conn.cursor()
    
    with open("view_database.md", "w", encoding="utf-8") as f:
        f.write("# Voice Agent Database Table View\n\n")
        f.write("This file is a human-readable snapshot of the SQLite tables. Open it in markdown preview mode in VS Code to see clean tables.\n\n")
        
        # 1. shipment_tracking
        try:
            cursor.execute("SELECT * FROM shipment_tracking")
            cols = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            
            f.write("## Table: shipment_tracking\n\n")
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join("---" for _ in cols) + " |\n")
            for row in rows:
                f.write("| " + " | ".join(str(val if val is not None else "") for val in row) + " |\n")
            f.write("\n")
        except Exception as e:
            f.write(f"Error reading shipment_tracking: {e}\n\n")

        # 2. conversation_logs
        try:
            cursor.execute("SELECT id, session_id, timestamp, user_transcript, ai_response, tracking_id_ref FROM conversation_logs")
            cols = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            
            f.write("## Table: conversation_logs (Selected Columns)\n\n")
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join("---" for _ in cols) + " |\n")
            for row in rows:
                # Escape pipe symbols in transcript / response to avoid breaking MD table layout
                escaped_row = []
                for val in row:
                    val_str = str(val if val is not None else "")
                    val_str = val_str.replace("|", "\\|").replace("\n", " ")
                    escaped_row.append(val_str)
                f.write("| " + " | ".join(escaped_row) + " |\n")
            f.write("\n")
        except Exception as e:
            f.write(f"Error reading conversation_logs: {e}\n\n")
            
    print("Successfully exported database tables to: view_database.md")
    conn.close()

if __name__ == "__main__":
    export_to_csv()
    export_to_markdown()
