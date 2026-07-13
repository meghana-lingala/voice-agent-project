import os
import threading
import time
from sarvamai import SarvamAI
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# Thread-safe lock for generating speech
tts_lock = threading.Lock()

# Instantiate the client
sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY) if SARVAM_API_KEY else None

def generate_speech_b64(text: str, target_language_code: str = "en-IN") -> str | None:
    """
    Generate speech using Sarvam AI Bulbul V3 model.
    Returns the base64-encoded audio string, or None on failure.
    Thread-safe implementation.
    """
    if not sarvam_client:
        print("[TTS] Sarvam client is not initialized (check SARVAM_API_KEY).")
        return None

    # Clean text input
    if not text or not text.strip():
        return None

    # Ensure proper language formatting for Sarvam AI (e.g. te/hi -> te-IN/hi-IN)
    lang = target_language_code
    if lang.startswith("te"):
        lang = "te-IN"
    elif lang.startswith("hi"):
        lang = "hi-IN"
    elif lang.startswith("en"):
        lang = "en-IN"
    else:
        lang = "en-IN"

    with tts_lock:
        # Resilient synthesis retry loop
        for attempt in range(3):
            try:
                response = sarvam_client.text_to_speech.convert(
                    text=text,
                    target_language_code=lang,
                    model="bulbul:v3",
                    speaker="shubh"
                )
                if response and response.audios:
                    return response.audios[0]
                else:
                    print(f"[TTS] Empty response received from Sarvam AI TTS (attempt {attempt + 1}).")
            except Exception as e:
                print(f"[TTS] Error calling Sarvam AI TTS (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    return None
            time.sleep(0.25 * (2 ** attempt))
        return None
