import os
import json
import threading
from datetime import datetime

LOG_FILE = "transcript_history.json"
# Threading lock to prevent race conditions during concurrent file writes
_log_lock = threading.Lock()

def log_transcript(
    audio_filename: str,
    transcribed_text: str,
    processing_duration_sec: float,
    engine_used: str,
    language_code: str,
    log_file_path: str = None
):
    """
    Thread-safely appends a standardized transcription transaction dictionary
    to the JSON array inside the specified log file.
    """
    if log_file_path is None:
        log_file_path = LOG_FILE
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "audio_filename": audio_filename or "unknown",
        "transcribed_text": transcribed_text,
        "processing_duration_sec": processing_duration_sec,
        "engine_used": engine_used,
        "language_code": language_code
    }
    
    with _log_lock:
        logs = []
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
                    if not isinstance(logs, list):
                        logs = []
            except (json.JSONDecodeError, IOError):
                # If file exists but is corrupted/empty, start fresh
                logs = []
                
        logs.append(log_entry)
        
        try:
            with open(log_file_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=4, ensure_ascii=False)
        except Exception as e:
            # Print error or log it appropriately
            print(f"Failed to write to transcription log: {e}")
            raise
