import os
import re
import sys
import glob
import requests
import audio_recorder
import uuid
import time
import base64
import pygame
import threading
import collections
import sounddevice as sd
import numpy as np

BACKEND_URL = "http://127.0.0.1:8000/transcribe"
END_SESSION_URL = "http://127.0.0.1:8000/end_session"
INITIATE_WORKFLOW_URL = "http://127.0.0.1:8000/initiate_workflow"

class PersistentAudioCapture:
    def __init__(self, samplerate=16000, channels=1, dtype='float32', blocksize=480, device=None):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.device = device
        
        self.stream = None
        self.queue = collections.deque(maxlen=1000)
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
    def start(self):
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.blocksize,
                device=self.device
            )
            self.stream.start()
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"[WARNING] Failed to start sounddevice persistent stream: {e}", file=sys.stderr)
            
    def _capture_loop(self):
        try:
            for _ in range(10):
                self.stream.read(self.blocksize)
        except Exception:
            pass
        while self.running:
            try:
                data, _ = self.stream.read(self.blocksize)
                with self.lock:
                    self.queue.append(data.copy())
            except Exception:
                time.sleep(0.01)
                
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

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

def calculate_and_set_threshold(persistent_rec):
    # Retrieve all frames collected during server latency window
    with persistent_rec.lock:
        latency_frames = list(persistent_rec.queue)
        persistent_rec.queue.clear()
        
    if latency_frames:
        flat = np.concatenate([f.flatten() for f in latency_frames])
        noise_rms = audio_recorder.calculate_rms(flat)
        # Calculate dynamic threshold above background noise floor
        # Using NOISE_FLOOR_MULTIPLIER = 4.0 and MIN_ENERGY_THRESHOLD = 0.0008
        energy_threshold = max(noise_rms * 4.0, 0.0008)
    else:
        energy_threshold = 0.0008
        noise_rms = 0.0
        
    print(f"[VAD] Pre-speech baseline lock. Noise floor RMS: {noise_rms:.5f} -> Energy threshold set to: {energy_threshold:.5f}")
    audio_recorder.set_pre_calculated_threshold(energy_threshold)

def play_audio_response(audio_b64, persistent_rec):
    if not audio_b64:
        return
    try:
        audio_bytes = base64.b64decode(audio_b64)
        output_active_path = "response_active.wav"
        with open(output_active_path, "wb") as audio_file:
            audio_file.write(audio_bytes)
        
        pygame.mixer.music.load(output_active_path)
        pygame.mixer.music.play()
        
        # Continuously purge the queue during playback to avoid capturing echo
        while pygame.mixer.music.get_busy():
            with persistent_rec.lock:
                persistent_rec.queue.clear()
            time.sleep(0.05)
            
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
    except Exception as play_err:
        print(f"[ERROR] Failed playing audio response: {play_err}")

def main():
    print("==================================================")
    print("      Voice Agent Interactive STT Client Panel    ")
    print("==================================================")
    
    # Initialize pygame mixer
    pygame.mixer.init()
    
    session_id = str(uuid.uuid4())
    print(f"[SYSTEM] Initialized unique Session ID: {session_id}")
    
    lang_code = "auto"
    target_dir = "samples/01_accuracy_benchmarks"
    
    # Step 1: Add Inactivity State Tracking Variables
    last_interaction_time = time.time()
    warning_triggered = False

    # Initialize persistent background streaming core
    persistent_rec = PersistentAudioCapture()
    persistent_rec.start()
    audio_recorder.set_persistent_recorder(persistent_rec)

    # Step 4: Ensure Greeting Invocation on Cold Start
    print("[SYSTEM] Fetching startup greeting workflow...")
    try:
        # Clear background queue before initiating request
        with persistent_rec.lock:
            persistent_rec.queue.clear()

        response = requests.post(INITIATE_WORKFLOW_URL, json={"session_id": session_id}, timeout=30)
        if response.status_code == 200:
            res_data = response.json()
            if "session_id" in res_data:
                session_id = res_data["session_id"]
            welcome_text = res_data.get("response_text", "")
            print("\n==========================================")
            print("        TRANSCRIBE & AGENT SESSION        ")
            print("==========================================")
            print(f" Agent (AI)     : {welcome_text}")
            print("==========================================\n")
            
            # Compute and set threshold from server latency window
            calculate_and_set_threshold(persistent_rec)
            
            audio_b64 = res_data.get("audio_b64")
            if audio_b64:
                play_audio_response(audio_b64, persistent_rec)
            
            # Short pause before starting first user turn
            time.sleep(0.5)
        else:
            print(f"[WARNING] Cold start initiate workflow failed with code {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("\n[CRITICAL ERROR] Could not connect to the FastAPI server.")
        print(f"Please ensure the backend is running locally at http://127.0.0.1:8000")
        print("Run command: .venv\\Scripts\\uvicorn main:app --reload\n")
        persistent_rec.stop()
        sys.exit(1)
    except Exception as e:
        print(f"[WARNING] Cold start connection failed: {e}")
    
    while True:
        try:
            # Step 2: Implement the Two-Stage Idle Rule
            elapsed = time.time() - last_interaction_time
            
            # Stage 2: Automatic Hard Exit (60 Seconds of Inactivity)
            if elapsed > 60.0:
                print("[SYSTEM] Inactivity limit reached. Terminating session safely due to timeout.")
                try:
                    # Dispatch a final non-blocking call to the server
                    requests.post(END_SESSION_URL, data={"session_id": session_id}, timeout=5)
                except Exception:
                    pass
                persistent_rec.stop()
                sys.exit(0)
                
            # Stage 1: The Warning Prompt (30 Seconds of Inactivity)
            if elapsed > 30.0 and not warning_triggered:
                print("[SYSTEM] Idle detected. Prompting user...")
                print("Are you still there? Please let me know if you have any questions, otherwise I will close this session shortly.")
                warning_triggered = True
                # Recalculate elapsed after warning output
                elapsed = time.time() - last_interaction_time
            
            print(f"\n[SYSTEM] Microphone is live... Speak now. (Session: {session_id})")
            
            # Calculate remaining time before timeout transition
            timeout_val = (60.0 - elapsed) if warning_triggered else (30.0 - elapsed)
            
            try:
                # Capture audio dynamically using VAD (uses pre-calculated threshold and persistent queue)
                audio_data = audio_recorder.record_audio(timeout=timeout_val)
            except TimeoutError:
                # Loop back to let the timeout rules process
                continue
            except Exception as e:
                print(f"[ERROR] Failed to record audio: {e}")
                time.sleep(2.0)
                continue
                
            if len(audio_data) == 0:
                continue
                
            # Generate next sequential path
            output_file_path = get_next_filename(target_dir)
            
            try:
                audio_recorder.save_audio(audio_data, output_file_path)
            except Exception as e:
                print(f"[ERROR] Failed to save WAV file locally: {e}")
                continue
                
            # 5. Network Payload Routing via Requests
            print("[SYSTEM] Processing turn...")
            
            # Clear background queue before starting request
            with persistent_rec.lock:
                persistent_rec.queue.clear()
                
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
                    transcript = res_data.get('transcript', '').strip()
                    
                    # Elegant mapping output
                    print("\n==========================================")
                    print("        TRANSCRIBE & AGENT SESSION        ")
                    print("==========================================")
                    print(f" You (Speech)   : {transcript}")
                    if "ai_response" in res_data:
                        print(f" Agent (AI)     : {res_data.get('ai_response')}")
                    print(f" Backend Engine : {res_data.get('telemetry', {}).get('engine_used')}")
                    print(f" Server Latency : {res_data.get('telemetry', {}).get('duration_seconds', 0.0):.3f} seconds")
                    print("==========================================\n")
                    
                    # Compute and set the threshold from the server latency window (Pre-Speech Baseline Lock)
                    calculate_and_set_threshold(persistent_rec)
                    
                    # Automated hands-free playback of agent's audio response
                    audio_b64 = res_data.get("audio_b64")
                    if audio_b64:
                        play_audio_response(audio_b64, persistent_rec)
                    
                    # Update interaction time if valid non-empty transcript is returned
                    if transcript:
                        last_interaction_time = time.time()
                        warning_triggered = False
                    
                    if "new_session_id" in res_data:
                        old_session = session_id
                        session_id = res_data["new_session_id"]
                        print(f"[SYSTEM] Session rotation triggered by Agent!")
                        print(f"  Old Session: {old_session}")
                        print(f"  New Session: {session_id}\n")
                        # Reset timeout state for the new session
                        last_interaction_time = time.time()
                        warning_triggered = False
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
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nExiting interactive panel. Goodbye!")
            persistent_rec.stop()
            break
        except Exception as e:
            print(f"[CRITICAL] Unexpected client loop crash: {e}")
            time.sleep(2.0)

if __name__ == "__main__":
    main()
