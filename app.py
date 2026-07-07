import os
import re
import sys
import glob
import requests
import audio_recorder
import uuid
import time

BACKEND_URL = "http://127.0.0.1:8000/transcribe"

def get_next_filename(directory: str) -> str:
    """
    Scans target directory at runtime for files matching sample_*.wav
    and dynamically increments index NNN to avoid overwrites.
    """
    os.makedirs(directory, exist_ok=True)
    existing_files = glob.glob(os.path.join(directory, "sample_*.wav"))
    max_idx = 0
    for file_path in existing_files:
        basename = os.path.basename(file_path)
        match = re.search(r"sample_(\d+)\.wav", basename)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
            
    next_idx = max_idx + 1
    return os.path.join(directory, f"sample_{next_idx:03d}.wav")

def main():
    print("==================================================")
    print("      Voice Agent Interactive STT Client Panel    ")
    print("==================================================")
    
    session_id = str(uuid.uuid4())
    print(f"[SYSTEM] Initialized unique Session ID: {session_id}")
    
    lang_code = "auto"
    target_dir = "samples/01_accuracy_benchmarks"
    
    while True:
        try:
            print(f"\n[SYSTEM] Microphone is live... Speak now. (Session: {session_id})")
            try:
                # Capture audio dynamically using VAD
                audio_data = audio_recorder.record_audio()
            except Exception as e:
                print(f"[ERROR] Failed to record audio: {e}")
                time.sleep(2.0)
                continue
                
            duration_sec = len(audio_data) / 16000.0
            
            # Local float32 RMS calculation for telemetry info
            rms = audio_recorder.calculate_rms(audio_data)
                
            # Generate next sequential path
            output_file_path = get_next_filename(target_dir)
            
            try:
                audio_recorder.save_audio(audio_data, output_file_path)
            except Exception as e:
                print(f"[ERROR] Failed to save WAV file locally: {e}")
                continue
                
            # 5. Network Payload Routing via Requests
            print("[SYSTEM] Processing turn...")
            try:
                with open(output_file_path, "rb") as f:
                    files = {"file": (os.path.basename(output_file_path), f, "audio/wav")}
                    data = {
                        "language_code": lang_code,
                        "session_id": session_id
                    }
                    
                    response = requests.post(BACKEND_URL, files=files, data=data, timeout=30)
                    
                if response.status_code == 200:
                    res_data = response.json()
                    
                    # Elegant mapping output
                    print("\n==========================================")
                    print("        TRANSCRIBE & AGENT SESSION        ")
                    print("==========================================")
                    print(f" You (Speech)   : {res_data.get('transcript')}")
                    if "ai_response" in res_data:
                        print(f" Agent (AI)     : {res_data.get('ai_response')}")
                    print(f" Backend Engine : {res_data.get('telemetry', {}).get('engine_used')}")
                    print(f" Server Latency : {res_data.get('telemetry', {}).get('duration_seconds', 0.0):.3f} seconds")
                    print("==========================================\n")
                    
                    if "new_session_id" in res_data:
                        old_session = session_id
                        session_id = res_data["new_session_id"]
                        print(f"[SYSTEM] Session rotation triggered by Agent!")
                        print(f"  Old Session: {old_session}")
                        print(f"  New Session: {session_id}\n")
                else:
                    try:
                        detail = response.json().get("detail", response.text)
                    except Exception:
                        detail = response.text
                    print(f"[ERROR] Backend returned status code {response.status_code}: {detail}")
                    
            except requests.exceptions.ConnectionError:
                print("\n[CRITICAL ERROR] Could not connect to the FastAPI server.")
                print(f"Please ensure the backend is running locally at http://127.0.0.1:8000")
                print("Run command: .venv\\Scripts\\uvicorn main:app --reload\n")
                time.sleep(2.0)
            except Exception as e:
                print(f"[ERROR] Networking pipeline error: {e}")
            
            # Breathing pause before turning mic back on
            time.sleep(1.0)

        except KeyboardInterrupt:
            print("\nExiting interactive panel. Goodbye!")
            break
        except Exception as e:
            print(f"[CRITICAL] Unexpected client loop crash: {e}")
            time.sleep(2.0)

if __name__ == "__main__":
    main()
