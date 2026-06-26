# Voice Agent STT Pipeline

A local, voice-activated speech-to-text recording and transcription pipeline featuring WebRTC Voice Activity Detection (VAD) with dynamic noise tracking, spectral noise reduction, a unified FastAPI backend server, and an interactive CLI client loop.

---

## Key Features

- **Voice-Activated Recording:** Captures audio from your microphone dynamically using `sounddevice` and automatically gates speech using WebRTC VAD.
- **Dynamic Noise Floor Tracking:** Continuously monitors room static and background noise (rolling 5th percentile over 4.5 seconds) to automatically adjust the VAD threshold, preventing false speech triggers.
- **Hangover Silence Detection:** Recording automatically stops 1.5 seconds after speech finishes (or 5.0 seconds of initial silence if no speech occurs).
- **Spectral Noise Reduction:** Strips out ambient stationary background hums (like fans, computer vents, or key clicks) using the `noisereduce` spectral suppression filter.
- **Dynamic File Indexing:** Automatically saves files sequentially in categorized folders (e.g., `samples/01_accuracy_benchmarks/sample_001.wav`, `sample_002.wav`) without overwriting previous benchmarks.
- **FastAPI Transcription Server:** Exposes a `/transcribe` endpoint with backend safety VAD filters that reject completely silent uploads locally, saving third-party API credits.
- **Dual API Transcription Engines with Concurrent Routing:** Routes requests intelligently:
  - English audio uses **Groq (Whisper Large V3 Turbo)**.
  - Indian regional languages (Hindi, Telugu) use **Sarvam AI (`saaras:v3`)**.
  - Probes Sarvam Telugu and Hindi concurrently to drastically reduce latency and applies genuine morpheme filters to prevent phonetic false positives, guaranteeing accurate script output.
- **Telemetry & Logging:** Tracks exact network execution durations and logs transaction entries (timestamps, filenames, transcripts, latencies, engine used, language tags) thread-safely into a structured database `transcript_history.json`.

---

## Directory Structure

```
├── app.py                      # Interactive CLI client runner
├── audio_recorder.py           # Client-side audio hardware stream capture
├── main.py                     # FastAPI REST API routing server
├── stt_services.py             # Transcription wrappers for Groq & Sarvam AI
├── transcript_logger.py        # Thread-safe JSON array logger
├── transcript_history.json     # Saved transcription transaction entries
│
├── requirements.txt            # System dependencies list
├── .env                        # Secret credentials config keys
│
└── test_*.py                   # Mock & Integration test suites
```

---

## Getting Started

Follow these steps to set up and run the voice agent pipeline locally.

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.13)
- An active hardware microphone plugged into your system
- A Groq API Key and Sarvam AI subscription key

### 2. Clone the Repository
```bash
git clone <repository-url>
cd voice-agent-project
```

### 3. Setup Virtual Environment
Create a virtual environment to manage dependencies:
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
Install all necessary packages defined in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 5. Configure API Credentials
Create a `.env` file in the project root directory and add your API credentials:
```env
GROQ_API_KEY=your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
```

---

## How to Run

For the complete pipeline to function, you need to run the **FastAPI Backend Server** first, followed by the **Interactive CLI Client**.

### Step A: Spin Up the Backend Server
Start Uvicorn to listen on localhost (port 8000):
```bash
# Ensure virtual environment is active
uvicorn main:app --reload
```
The server will now be listening on `http://127.0.0.1:8000`.

### Step B: Launch the Interactive CLI Client
Open a second terminal window, activate the virtual environment, and run:
```bash
# Ensure virtual environment is active
python app.py
```

#### Client Guide:
1. Select or input your target language tag (`auto` for auto-detect, `en` for English, `hi-IN` for Hindi, `te-IN` for Telugu).
2. Choose your benchmark folder target (`samples/01_accuracy_benchmarks/` or `samples/02_noise_scenarios/`).
3. Press **Enter** to initialize recording.
4. Speak naturally. The pipeline will automatically stop capturing 1.5 seconds after you finish talking, apply noise suppression, save the file, upload it to the backend, and print out transcription and telemetry details!

---

## How to Run Tests

Execute the discover runner to verify that all VAD (including dynamic tracking), file management, logging, concurrent API routing, and loop integrations pass successfully:
```bash
python -m unittest discover -p "test_*.py"
```
All tests should return green.