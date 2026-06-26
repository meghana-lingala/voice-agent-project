import os
import time
import shutil
import requests

BACKEND_URL = "http://127.0.0.1:8000/transcribe"
HISTORY_FILE = "transcript_history.json"
BACKUP_FILE = "transcript_history.json.bak"

ENGLISH_SAMPLE = r"c:\Users\Lenovo\Desktop\AI_Projects\voice-agent-project\samples\01_accuracy_benchmarks\sample_001.wav"
HINDI_SAMPLE = r"c:\Users\Lenovo\Desktop\AI_Projects\voice-agent-project\samples\01_accuracy_benchmarks\sample_020.wav"

def check_files():
    for name, path in [("English", ENGLISH_SAMPLE), ("Hindi", HINDI_SAMPLE)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required sample file for {name} speech not found at: {path}")
    print("[INFO] Sample audio files validated successfully.")

def backup_history():
    if os.path.exists(HISTORY_FILE):
        print(f"[INFO] Backing up operational database {HISTORY_FILE} to {BACKUP_FILE}...")
        shutil.copy2(HISTORY_FILE, BACKUP_FILE)
    else:
        print("[INFO] No operational database found. No backup needed.")

def restore_history():
    if os.path.exists(BACKUP_FILE):
        print(f"[INFO] Restoring operational database from backup...")
        shutil.move(BACKUP_FILE, HISTORY_FILE)
    elif os.path.exists(HISTORY_FILE):
        print(f"[INFO] Cleaning up created database to preserve system state...")
        os.remove(HISTORY_FILE)

def run_iteration(file_path, lang_code):
    filename = os.path.basename(file_path)
    start_time = time.perf_counter()
    with open(file_path, "rb") as f:
        files = {"file": (filename, f, "audio/wav")}
        data = {"language_code": lang_code}
        response = requests.post(BACKEND_URL, files=files, data=data, timeout=30)
    end_time = time.perf_counter()
    
    if response.status_code != 200:
        raise RuntimeError(f"Request failed with status {response.status_code}: {response.text}")
        
    # Return duration in milliseconds
    return (end_time - start_time) * 1000.0

def run_vector(name, file_path, lang_code, iterations=10):
    print(f"\n[BENCHMARK] Running Vector: {name} (File: {os.path.basename(file_path)}, Lang Code: '{lang_code}')...")
    latencies = []
    for idx in range(1, iterations + 1):
        try:
            lat_ms = run_iteration(file_path, lang_code)
            latencies.append(lat_ms)
            print(f"  Iteration {idx:2d}/10: {lat_ms:.2f} ms")
        except Exception as e:
            print(f"  Iteration {idx:2d}/10: FAILED ({e})")
            
    if not latencies:
        raise RuntimeError(f"All iterations failed for vector: {name}")
        
    return latencies

def main():
    check_files()
    backup_history()
    
    vectors = [
        ("Vector 1: Manual 'en' (English Audio)", ENGLISH_SAMPLE, "en"),
        ("Vector 2: Manual 'hi-IN' (Hindi Audio)", HINDI_SAMPLE, "hi-IN"),
        ("Vector 3: Auto 'auto' (English Audio)", ENGLISH_SAMPLE, "auto"),
        ("Vector 4: Auto 'auto' (Hindi Audio)", HINDI_SAMPLE, "auto")
    ]
    
    results = {}
    try:
        for name, file_path, lang_code in vectors:
            results[name] = run_vector(name, file_path, lang_code)
    finally:
        restore_history()
        
    # Calculate statistics
    stats = {}
    for name, latencies in results.items():
        stats[name] = {
            "min": min(latencies),
            "max": max(latencies),
            "mean": sum(latencies) / len(latencies)
        }
        
    # Calculate Latency Tax Deltas
    en_tax = stats["Vector 3: Auto 'auto' (English Audio)"]["mean"] - stats["Vector 1: Manual 'en' (English Audio)"]["mean"]
    hi_tax = stats["Vector 4: Auto 'auto' (Hindi Audio)"]["mean"] - stats["Vector 2: Manual 'hi-IN' (Hindi Audio)"]["mean"]
    
    # Format and display markdown table
    print("\n" + "="*80)
    print("                      LATENCY BENCHMARK REPORT SUMMARY                      ")
    print("="*80)
    print("\n| Benchmark Vector | Min Latency (ms) | Max Latency (ms) | Mean Latency (ms) | Latency Tax Delta (ms) |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    
    v1_mean = stats["Vector 1: Manual 'en' (English Audio)"]["mean"]
    v1_min = stats["Vector 1: Manual 'en' (English Audio)"]["min"]
    v1_max = stats["Vector 1: Manual 'en' (English Audio)"]["max"]
    print(f"| Vector 1: Manual 'en' (English Audio) | {v1_min:.2f} | {v1_max:.2f} | {v1_mean:.2f} | — |")
    
    v2_mean = stats["Vector 2: Manual 'hi-IN' (Hindi Audio)"]["mean"]
    v2_min = stats["Vector 2: Manual 'hi-IN' (Hindi Audio)"]["min"]
    v2_max = stats["Vector 2: Manual 'hi-IN' (Hindi Audio)"]["max"]
    print(f"| Vector 2: Manual 'hi-IN' (Hindi Audio) | {v2_min:.2f} | {v2_max:.2f} | {v2_mean:.2f} | — |")
    
    v3_min = stats["Vector 3: Auto 'auto' (English Audio)"]["min"]
    v3_max = stats["Vector 3: Auto 'auto' (English Audio)"]["max"]
    v3_mean = stats["Vector 3: Auto 'auto' (English Audio)"]["mean"]
    print(f"| Vector 3: Auto 'auto' (English Audio) | {v3_min:.2f} | {v3_max:.2f} | {v3_mean:.2f} | {en_tax:+.2f} |")
    
    v4_mean = stats["Vector 4: Auto 'auto' (Hindi Audio)"]["mean"]
    v4_min = stats["Vector 4: Auto 'auto' (Hindi Audio)"]["min"]
    v4_max = stats["Vector 4: Auto 'auto' (Hindi Audio)"]["max"]
    print(f"| Vector 4: Auto 'auto' (Hindi Audio) | {v4_min:.2f} | {v4_max:.2f} | {v4_mean:.2f} | {hi_tax:+.2f} |")
    
    print("\n[NOTE] 'Latency Tax Delta' represents the overhead of automatic language detection vs manual routing.")
    print("================================================================================")

if __name__ == "__main__":
    main()
