import os
import time
# pyrefly: ignore [missing-import]
import soundfile as sf
from dotenv import load_dotenv
from groq import Groq
from sarvamai import SarvamAI
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# ── API clients ───────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

groq_client   = Groq(api_key=GROQ_API_KEY)   if GROQ_API_KEY   else None
sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY) if SARVAM_API_KEY else None

# ── Script-ratio threshold ────────────────────────────────────────────────────
# Minimum fraction of Indic Unicode characters in a transcript for it to
# be considered as a candidate Indic language.
INDIC_SCRIPT_RATIO_THRESHOLD = 0.30

# Sarvam saaras:v3 rejects files longer than 30 seconds.
SARVAM_MAX_DURATION_SEC = 30.0

# ── Curated word lists ────────────────────────────────────────────────────────
#
# These are authentic morphemes in their respective language that CANNOT
# appear in the phonetic (Indic-script) transcription of English or the
# other Indic language by Groq / Sarvam.  The key insight is:
#
#   English audio → Sarvam te-IN → "హాయ్ ఐ వాంట్ టు స్పీక్"
#     → every "word" is just English phonetics; NONE of these appear here.
#
#   Telugu audio → Sarvam te-IN → "నేను ఇంట్లో ఉన్నాను"
#     → genuine Telugu morphemes; MANY of these appear here.
#
#   Hindi audio → Sarvam te-IN → "నమస్కార హమాపే హిం కాన్"
#     → phonetics of Hindi; NONE of these appear here.
#
# Finding even ONE word from the appropriate list is sufficient to confirm
# the language.  This is more robust than any numerical ratio comparison.
# ─────────────────────────────────────────────────────────────────────────────
TELUGU_CONFIDENT_WORDS = frozenset({
    # Pronouns / postpositions  (never phonetically confused with English)
    'నేను', 'మీరు', 'మీకు', 'నాకు', 'వారు', 'మనం', 'నాతో', 'మీతో',
    'నీకు', 'నీవు', 'వాళ్ళు', 'మాకు',
    # Core verbs / auxiliaries
    'ఉన్నాను', 'ఉన్నారు', 'ఉన్నాయి', 'ఉంది', 'ఉన్నది',
    'కావాలి', 'కావాలు', 'వస్తాయి', 'వస్తాను', 'వస్తారు',
    'తెలుసు', 'అర్థం', 'అయ్యింది', 'అయింది', 'చేస్తాను',
    'ఇస్తాను', 'వెళ్ళాను', 'చెప్పాను',
    # Common nouns
    'ఇంట్లో', 'ఇల్లు', 'పేరు', 'తిండి', 'నీళ్ళు', 'డబ్బు', 'డబ్బులు',
    'కాస్త', 'కాస్తా',
    # Conjunctions / particles
    'కానీ', 'కూడా', 'చాలా', 'చాల', 'అయినా', 'అందుకే', 'అయితే', 'ఐతే',
    # Question words
    'ఎలా', 'ఏమి', 'ఎంత', 'ఎక్కడ', 'ఎప్పుడు', 'ఎవరు', 'ఏదో',
    # Adjectives / other
    'మంచి', 'బావు', 'అందరు', 'కొంచెం', 'కుంచెం', 'కుంచమ',
    'ఇప్పుడు', 'అక్కడ', 'ఇక్కడ', 'ఒకటి',
    # Additional common words from recorded samples
    'పెటగలివ్పే', 'పెట్టగలరా', 'ఉన్నాను',
})

HINDI_CONFIDENT_WORDS = frozenset({
    # Pronouns / postpositions
    'मैं', 'आप', 'हम', 'वो', 'यह', 'वह', 'तुम', 'मेरा', 'आपका',
    'हमारा', 'मुझे', 'आपको', 'उसको', 'हमको',
    # Core verbs / auxiliaries
    'है', 'हैं', 'हो', 'था', 'थे', 'थी', 'हुआ', 'हुई',
    # Common conjunctions / particles
    'लेकिन', 'क्योंकि', 'भी', 'नहीं', 'ठीक', 'अच्छा',
    'कैसे', 'क्यों', 'कब', 'कहां',
    # Greetings
    'नमस्ते', 'धन्यवाद',
    # Other unique Hindi
    'बहुत', 'साथ', 'बात', 'लोग', 'सब', 'कुछ', 'मिलकर',
    'नाचेंगे', 'करेंगे', 'जाएंगे',
})


LATIN_HINDI_WORDS = frozenset({
    "kidhar", "kethar", "kadar", "kya", "karna", "karo", "mera", "meri", "mery", "apka", 
    "aapka", "hoga", "tha", "raha", "chahiye", "bol", "bolo", "batao", "sunte", "sun", 
    "kaha", "kahan", "kab", "kaise", "kon", "kaun", "bhai", "namaste", "shukriya",
    "dhanyawad", "dhanyabad", "hai", "he", "shuru", "kariye", "karna", "krna"
})

LATIN_TELUGU_WORDS = frozenset({
    "ekkada", "daggara", "undi", "unnadu", "chesanu", "chesali", "nenu", "mari", "kavali",
    "ekada", "lodu", "matladu", "cheppu", "enti", "cheyali", "naku", "maku", "meru", "miaku",
    "namaskaram", "dhanyavadalu", "avunu", "kadu", "ledu", "undha", "cheyandi"
})


# ── Private helpers ───────────────────────────────────────────────────────────

def _detect_indic_script_ratio(text: str) -> float:
    """
    Return the fraction of characters in *text* belonging to Telugu
    (U+0C00-0C7F) or Devanagari (U+0900-097F).
    Pure ASCII / Latin returns 0.0.
    """
    if not text:
        return 0.0
    total = len(text)
    indic = sum(
        1 for ch in text
        if '\u0900' <= ch <= '\u097f'   # Devanagari (Hindi)
        or '\u0c00' <= ch <= '\u0c7f'   # Telugu
    )
    return indic / total


def _is_any_indic(text: str, threshold: float = 0.20) -> bool:
    """
    Return True if *text* contains significant content from ANY Indic
    Unicode block (covers Telugu, Hindi, Tamil, Kannada, Malayalam …).

    Used to validate Groq auto-detect results: even when Groq misclassifies
    Telugu as Tamil, the result still has high broad-Indic ratio.
    """
    if not text:
        return False
    total = len(text)
    broad = sum(
        1 for ch in text
        if '\u0900' <= ch <= '\u097f'   # Devanagari
        or '\u0b80' <= ch <= '\u0bff'   # Tamil
        or '\u0c00' <= ch <= '\u0c7f'   # Telugu
        or '\u0c80' <= ch <= '\u0cff'   # Kannada
        or '\u0d00' <= ch <= '\u0d7f'   # Malayalam
    )
    return (broad / total) >= threshold


def _has_genuine_telugu(text: str) -> bool:
    """
    Return True if *text* contains authentic Telugu morphemes, using a length/count
    gate to prevent false positives from English phonetic sounds.
    """
    if not text:
        return False
    words = {w.strip('.,!?:;()[]') for w in text.split()}
    matches = words & TELUGU_CONFIDENT_WORDS
    if not matches:
        return False
    # If we have only 1 match, require it to be a long specific Telugu word (length >= 5)
    # to avoid false positives on short pronouns/particles
    has_long_specific = any(len(m) >= 5 for m in matches)
    return has_long_specific or len(matches) >= 2


def _has_genuine_hindi(text: str) -> bool:
    """
    Return True if *text* contains authentic Hindi morphemes, using a length/count
    gate to prevent false positives from English phonetic sounds.
    """
    if not text:
        return False
    words = {w.strip('.,!?:;()[]') for w in text.split()}
    matches = words & HINDI_CONFIDENT_WORDS
    if not matches:
        return False
    # If we have only 1 match, require it to be a long specific Hindi word (length >= 5)
    has_long_specific = any(len(m) >= 5 for m in matches)
    return has_long_specific or len(matches) >= 2


def _get_audio_duration(file_path: str) -> float:
    """
    Return audio duration in seconds.
    Returns 999.0 on any read error so the caller takes the safe Groq path.
    """
    try:
        return sf.info(file_path).duration
    except Exception:
        return 999.0


def _sarvam_probe(file_path: str, language_code: str) -> tuple:
    """
    Attempt a Sarvam transcription with *language_code*.
    Returns (transcript_text, api_duration_seconds, indic_script_ratio).
    Raises on API errors so the caller can fall through gracefully.
    """
    result = transcribe_indic(file_path, language_code=language_code)
    ratio  = _detect_indic_script_ratio(result["transcript"])
    return result["transcript"], result["telemetry"]["duration_seconds"], ratio


def _groq_forced(file_path: str, filename: str, lang_code: str) -> tuple:
    """
    Re-query Groq Whisper with an explicit 2-letter language hint.
    Returns (transcript_text, api_duration_seconds).
    """
    start = time.perf_counter()
    with open(file_path, "rb") as fh:
        resp = groq_client.audio.transcriptions.create(
            file     = (filename, fh),
            model    = "whisper-large-v3-turbo",
            language = lang_code,
            prompt   = "The user is providing a shipment tracking ID starting with the characters SH followed by digits. Example: SH123, SH456, SH789. Do not confuse SH with 'Hi' or generic words.",
        )
    return resp.text, time.perf_counter() - start


def _groq_auto(file_path: str, filename: str) -> tuple:
    """
    Groq Whisper without a language hint (auto language detection).
    Returns (transcript_text, api_duration_seconds).
    """
    start = time.perf_counter()
    with open(file_path, "rb") as fh:
        resp = groq_client.audio.transcriptions.create(
            file  = (filename, fh),
            model = "whisper-large-v3-turbo",
            prompt = "The user is providing a shipment tracking ID starting with the characters SH followed by digits. Example: SH123, SH456, SH789. Do not confuse SH with 'Hi' or generic words.",
        )
    return resp.text, time.perf_counter() - start


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe_english(file_path: str) -> dict:
    """
    Transcribe an English audio file using Groq Whisper large-v3-turbo.

    Returns:
        { "transcript": str, "telemetry": { "duration_seconds": float } }
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the environment.")
    if not groq_client:
        raise RuntimeError("Groq client is not initialized.")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        start = time.perf_counter()
        response = groq_client.audio.transcriptions.create(
            file  = (filename, f),
            model = "whisper-large-v3-turbo",
            prompt = "The user is providing a shipment tracking ID starting with the characters SH followed by digits. Example: SH123, SH456, SH789. Do not confuse SH with 'Hi' or generic words.",
        )
        duration = time.perf_counter() - start

    return {"transcript": response.text, "telemetry": {"duration_seconds": duration}}


def transcribe_indic(file_path: str, language_code: str = "hi-IN") -> dict:
    """
    Transcribe a Hindi or Telugu audio file using Sarvam AI saaras:v3.

    The model handles code-switching natively: a Telugu sentence with a few
    English words is transcribed correctly as Telugu+English mixed output.

    Args:
        file_path    : path to the WAV file
        language_code: 'hi-IN' (Hindi) or 'te-IN' (Telugu)

    Returns:
        { "transcript": str, "telemetry": { "duration_seconds": float } }
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in the environment.")
    if not sarvam_client:
        raise RuntimeError("Sarvam AI client is not initialized.")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        start = time.perf_counter()
        
        # Resilient transcription retry loop
        for attempt in range(3):
            try:
                f.seek(0)
                response = sarvam_client.speech_to_text.transcribe(
                    file          = (filename, f),
                    language_code = language_code,
                    model         = "saaras:v3",
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(0.25 * (2 ** attempt))
                
        duration = time.perf_counter() - start

    return {
        "transcript": response.transcript,
        "telemetry": {"duration_seconds": duration},
    }


def transcribe_auto(file_path: str) -> dict:
    """
    Auto-detect language and transcribe using concurrent Indic probing with word-check.

    ── Design Strategy ────────────────────────────────────────────────────────
    1. Run Telugu and Hindi probes in parallel to minimize latency.
       - Short audio (<= 30s): Sarvam te-IN and hi-IN probes.
       - Long audio (> 30s): Groq forced te and hi probes.
    2. Check results for genuine Indic words. If Telugu or Hindi contains
       confident genuine words (using length-and-count filters to prevent false
       positives on English phonetic soundalikes), return the Indic language.
    3. If neither passes the genuine word check, call Groq auto-detect to
       transcribe as English.
    ───────────────────────────────────────────────────────────────────────────
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the environment.")
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in the environment.")
    if not groq_client:
        raise RuntimeError("Groq client is not initialized.")

    filename       = os.path.basename(file_path)
    audio_duration = _get_audio_duration(file_path)

    # ═════════════════════════════════════════════════════════════════════════
    # SHORT AUDIO PATH (<= 30 s) — Concurrent Sarvam probes
    # ═════════════════════════════════════════════════════════════════════════
    if audio_duration <= SARVAM_MAX_DURATION_SEC:
        # Run Sarvam and Groq probes in parallel to minimize latency and maximize routing accuracy
        with ThreadPoolExecutor(max_workers=3) as executor:
            te_future = executor.submit(_sarvam_probe, file_path, "te-IN")
            hi_future = executor.submit(_sarvam_probe, file_path, "hi-IN")
            groq_future = executor.submit(_groq_auto, file_path, filename)
            
            # Fetch results
            te_text = te_dur = te_ratio = None
            try:
                te_text, te_dur, te_ratio = te_future.result()
            except Exception as exc:
                print(f"[LID] Sarvam te-IN probe failed: {exc}")
                
            hi_text, hi_dur, hi_ratio = None, None, 0.0
            try:
                hi_text, hi_dur, hi_ratio = hi_future.result()
            except Exception as exc:
                print(f"[LID] Sarvam hi-IN probe failed: {exc}")

            auto_text = auto_dur = None
            try:
                auto_text, auto_dur = groq_future.result()
            except Exception as exc:
                print(f"[LID] Groq auto probe failed: {exc}")

        # 1. Short-circuit: Verify if the audio is actually Indic.
        # If Groq auto-detection transcribed it and determined it does NOT contain Indic characters,
        # we immediately route it as English. This prevents false positives from Sarvam 
        # translating/hallucinating Telugu text from English speech.
        is_indic = False
        if auto_text:
            is_indic = _is_any_indic(auto_text, threshold=0.15)
        
        has_telugu = te_text and _has_genuine_telugu(te_text)
        has_hindi = hi_text and _has_genuine_hindi(hi_text)

        if not is_indic and not (has_telugu or has_hindi) and auto_text is not None:
            # Check for Latin-script phonetic Hindi or Telugu words
            words = {w.strip('.,!?:;()[]').lower() for w in auto_text.split()}
            hi_matches = words & LATIN_HINDI_WORDS
            te_matches = words & LATIN_TELUGU_WORDS
            
            if hi_matches and len(hi_matches) >= 1:
                return {
                    "transcript": auto_text,
                    "telemetry": {
                        "duration_seconds": auto_dur,
                        "engine_used": "Groq",
                        "detected_language": "Hindi",
                    },
                }
            elif te_matches and len(te_matches) >= 1:
                return {
                    "transcript": auto_text,
                    "telemetry": {
                        "duration_seconds": auto_dur,
                        "engine_used": "Groq",
                        "detected_language": "Telugu",
                    },
                }
            
            return {
                "transcript": auto_text,
                "telemetry": {
                    "duration_seconds": auto_dur,
                    "engine_used": "Groq",
                    "detected_language": "English",
                },
            }

        # 2. Check for genuine Telugu
        if te_text and _has_genuine_telugu(te_text):
            return {
                "transcript": te_text,
                "telemetry": {
                    "duration_seconds" : te_dur,
                    "engine_used"      : "Sarvam",
                    "detected_language": "Telugu",
                },
            }

        # 3. Check for genuine Hindi
        if hi_text and _has_genuine_hindi(hi_text):
            return {
                "transcript": hi_text,
                "telemetry": {
                    "duration_seconds" : hi_dur,
                    "engine_used"      : "Sarvam",
                    "detected_language": "Hindi",
                },
            }

        # 4. Fallback matching using script ratios
        te_ratio_val = te_ratio if te_ratio is not None else 0.0
        hi_ratio_val = hi_ratio if hi_ratio is not None else 0.0
        if te_ratio_val >= hi_ratio_val:
            return {
                "transcript": te_text if te_text else auto_text,
                "telemetry": {
                    "duration_seconds" : te_dur if te_dur else auto_dur,
                    "engine_used"      : "Sarvam" if te_text else "Groq",
                    "detected_language": "Telugu",
                },
            }
        else:
            return {
                "transcript": hi_text if hi_text else auto_text,
                "telemetry": {
                    "duration_seconds" : hi_dur if hi_dur else auto_dur,
                    "engine_used"      : "Sarvam" if hi_text else "Groq",
                    "detected_language": "Hindi",
                },
            }

    # ═════════════════════════════════════════════════════════════════════════
    # LONG AUDIO PATH (> 30 s) — Concurrent Groq forced probes
    # ═════════════════════════════════════════════════════════════════════════
    with ThreadPoolExecutor(max_workers=2) as executor:
        te_future = executor.submit(_groq_forced, file_path, filename, "te")
        hi_future = executor.submit(_groq_forced, file_path, filename, "hi")
        
        te_text, te_dur = te_future.result()
        hi_text, hi_dur = hi_future.result()

    te_ratio = _detect_indic_script_ratio(te_text)
    hi_ratio = _detect_indic_script_ratio(hi_text)

    # Check for genuine Telugu
    if _has_genuine_telugu(te_text):
        return {
            "transcript": te_text,
            "telemetry": {
                "duration_seconds" : te_dur,
                "engine_used"      : "Groq Fallback",
                "detected_language": "Telugu",
            },
        }

    # Check for genuine Hindi
    if _has_genuine_hindi(hi_text):
        return {
            "transcript": hi_text,
            "telemetry": {
                "duration_seconds" : hi_dur,
                "engine_used"      : "Groq Fallback",
                "detected_language": "Hindi",
            },
        }

    # Fallback: check if Groq auto-detect confirms Indic
    auto_text, auto_dur = _groq_auto(file_path, filename)
    if _is_any_indic(auto_text, threshold=0.15):
        if te_ratio >= hi_ratio:
            return {
                "transcript": te_text,
                "telemetry": {
                    "duration_seconds" : te_dur,
                    "engine_used"      : "Groq Fallback",
                    "detected_language": "Telugu",
                },
            }
        else:
            return {
                "transcript": hi_text,
                "telemetry": {
                    "duration_seconds" : hi_dur,
                    "engine_used"      : "Groq Fallback",
                    "detected_language": "Hindi",
                },
            }

    # Otherwise, English
    return {
        "transcript": auto_text,
        "telemetry": {
            "duration_seconds" : auto_dur,
            "engine_used"      : "Groq",
            "detected_language": "English",
        },
    }
