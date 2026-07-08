# Voice Agent STT Pipeline

A hands-free, continuous turn-taking voice assistant featuring WebRTC Voice Activity Detection (VAD) with dynamic energy gating, spectral noise reduction, a relational SQLite database layer, multi-turn OpenAI LLM conversational brain with function tool calling, and defensive guardrails.

---

## Key Features

- **Hands-Free Continuous Voice Loop:** The CLI client runs in an automated, non-blocking reactivation loop. It automatically turns the microphone on, captures speech, processes, rotates sessions, pauses briefly, and reactivates without any manual keyboard input.
- **Energy-Gated VAD Activation:** Bypasses processing when silent. Streaming starts recording *only* once incoming sound levels cross the dynamically calibrated energy threshold. Includes a 300 ms pre-roll lookback buffer to prevent clipping the start of speech.
- **Relational SQLite Database Layer (`voice_agent.db`):** 
  - `shipment_tracking`: Logistics inventory ledger containing mock shipment profiles.
  - `conversation_logs`: Thread-safe ledger capturing transcriptions, response metadata, server latencies, and session mappings.
- **OpenAI LLM Brain with Scenario Pathways:** Powered by `gpt-4o-mini` with strict system prompting covering shipment tracking, order placement checklist verification, empathetic delay apologies, and FAQs.
- **Context Window Memory Loop:** Automatically retrieves the last 10 turns of conversational logs for the active session and injects them between the system prompt and the user's latest query to maintain memory.
- **Function Tool Calling:** Exposes `create_new_shipment_record()` (automatically triggered when pickup address, destination, weight, and time slot details are collected) and `end_current_session()` (triggered on goodbye).
- **Client-Side Inactivity Timeout Tracker:** 
  - **Stage 1 (Warning at 30s):** Prompts the user with a gentle reminder if silent for 30 seconds.
  - **Stage 2 (Safe Hard Exit at 60s):** Automatically triggers backend session cleanup via `/end_session` and terminates cleanly.
- **Defensive Guardrails & Scope Deflection:** Politely deflects completely off-topic inquiries using a standardized statement. Protects API communication with graceful connection error fallback strings.
- **Dual STT Engine Routing:** Routes English audio to **Groq (Whisper Large V3 Turbo)** and Indian regional languages (Hindi, Telugu) to **Sarvam AI (`saaras:v3`)** concurrently.

---

## Directory Structure

```
├── app.py                      # Hands-free continuous CLI client runner
├── audio_recorder.py           # Client-side audio stream capture with VAD gating
├── database.py                 # Thread-safe SQLite DB schema management
├── llm_service.py              # OpenAI LLM orchestration (Scenario prompt, tools)
├── main.py                     # FastAPI REST API routing server with /transcribe & /end_session
├── stt_services.py             # STT wrappers for Groq & Sarvam AI
├── requirements.txt            # Python dependencies list
├── .env                        # Secret credentials config keys
└── test_*.py                   # Comprehensive unit test suites (71 tests)
```

---

## Getting Started

Follow these steps to set up and run the voice agent pipeline locally.

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.13)
- An active hardware microphone plugged into your system
- API Keys: OpenAI API Key, Groq API Key, and Sarvam AI Key

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
```bash
pip install -r requirements.txt
```

### 5. Configure API Credentials
Create a `.env` file in the project root directory and add your credentials:
```env
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
```

---

## How to Run

For the complete pipeline to function, run the backend server first, followed by the voice client.

### Step A: Spin Up the Backend Server
Start Uvicorn to listen on localhost (port 8000):
```bash
uvicorn main:app --reload
```
The server will now be listening on `http://127.0.0.1:8000`.

### Step B: Launch the Interactive CLI Client
Open a second terminal window, activate the virtual environment, and run:
```bash
python app.py
```
The client will automatically initialize, calibrate to room noise floor, and turn on the microphone. Simply start speaking to talk to the assistant!

---

## How to Run Tests

Execute the discover runner to verify that all modules pass successfully:
```bash
python -m unittest discover -p "test_*.py"
```
All **71 tests** should return green.