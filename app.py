import os
import re
import sys
import glob
import requests
import audio_recorder

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
    
    while True:
        try:
            print("\n--- New Transcription Session (type 'exit' to quit) ---")
            
            # 1. Target Language Code Selection
            lang_code = input("Enter language code (auto / en / hi-IN / te-IN) [default: auto]: ").strip()
            if lang_code.lower() == 'exit':
                break
            if not lang_code:
                lang_code = "auto"
                
            # 2. Subdirectory Selection
            print("\nSelect target sample directory:")
            print("1. samples/01_accuracy_benchmarks/ (default)")
            print("2. samples/02_noise_scenarios/")
            print("3. Custom path")
            
            dir_choice = input("Choice (1/2/3) [default: 1]: ").strip()
            if dir_choice.lower() == 'exit':
                break
                
            if dir_choice == '2':
                target_dir = "samples/02_noise_scenarios"
            elif dir_choice == '3':
                target_dir = input("Enter custom folder path: ").strip()
                if not target_dir:
                    target_dir = "samples/01_accuracy_benchmarks"
            else:
                target_dir = "samples/01_accuracy_benchmarks"

            # 3. Prompt user to trigger recording
            input(f"\nPress Enter to start recording (Target: {lang_code})...")
            
            print("\n[CLI FEEDBACK] Recording started... Speak naturally. Processing stops automatically after you stop talking.")
            try:
                # Capture audio dynamically using VAD
                audio_data = audio_recorder.record_audio()
            except Exception as e:
                print(f"[ERROR] Failed to record audio: {e}")
                continue
                
            duration_sec = len(audio_data) / 16000.0
            print(f"[CLI FEEDBACK] Recording completed. Saved asset. Duration: {duration_sec:.1f} seconds.")
            
            # 4. Local float32 RMS calculation for telemetry info
            rms = audio_recorder.calculate_rms(audio_data)
            print(f"[CLI FEEDBACK] Local Calculated RMS: {rms:.6f}")
                
            # Generate next sequential path
            output_file_path = get_next_filename(target_dir)
            print(f"[CLI FEEDBACK] Saving local recording to: {output_file_path}")
            
            try:
                audio_recorder.save_audio(audio_data, output_file_path)
            except Exception as e:
                print(f"[ERROR] Failed to save WAV file locally: {e}")
                continue
                
            # 5. Network Payload Routing via Requests
            print(f"[CLI FEEDBACK] Connecting to backend server: {BACKEND_URL} ...")
            try:
                with open(output_file_path, "rb") as f:
                    files = {"file": (os.path.basename(output_file_path), f, "audio/wav")}
                    data = {"language_code": lang_code}
                    
                    response = requests.post(BACKEND_URL, files=files, data=data, timeout=30)
                    
                if response.status_code == 200:
                    res_data = response.json()
                    
                    # Elegant mapping output
                    print("\n==========================================")
                    print("         TRANSCRIPTION SUCCESSFUL         ")
                    print("==========================================")
                    print(f" Transcription Text : {res_data.get('transcript')}")
                    print(f" Backend Engine     : {res_data.get('telemetry', {}).get('engine_used')}")
                    print(f" Server Latency     : {res_data.get('telemetry', {}).get('duration_seconds', 0.0):.3f} seconds")
                    print("==========================================\n")
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
            except Exception as e:
                print(f"[ERROR] Networking pipeline error: {e}")

        except KeyboardInterrupt:
            print("\nExiting interactive panel. Goodbye!")
            break
        except Exception as e:
            print(f"[CRITICAL] Unexpected client loop crash: {e}")

if __name__ == "__main__":
    main()
