import os
def normalize_indic_alphanumerics(text: str) -> str:
    """
    Normalizes Telugu and Hindi phonetic letters and digits to standard English alpha-numerics.
    """
    if not text:
        return text

    # Base dictionary for mapping native and phonetic numbers
    mapping = {
        # Letter replacements
        "ఎస్ హెచ్": "SH",
        "एस एच": "SH",
        "एसएच": "SH",
        
        # Telugu digit words (Phonetic & Native)
        "వన్": "1", "ఒకటి": "1",
        "టూ": "2", "రెండు": "2",
        "త్రీ": "3", "మూడు": "3",
        "ఫోర్": "4", "నాలుగు": "4",
        "ఫైవ్": "5", "ఐదు": "5",
        "సిక్స్": "6", "ఆరు": "6",
        "సెవెన్": "7", "ఏడు": "7",
        "ఎయిట్": "8", "ఎనిమిది": "8",
        "నైన్": "9", "తొమ్మిది": "9",
        "జీరో": "0", "సున్నా": "0",
        
        # Hindi digit words (Phonetic & Native)
        "वन": "1", "एक": "1",
        "टू": "2", "दो": "2",
        "थ्री": "3", "तीन": "3",
        "फोर": "4", "चार": "4",
        "फाइव": "5", "पांच": "5", "पाँच": "5",
        "सिक्स": "6", "छह": "6", "छः": "6",
        "सेवन": "7", "सात": "7",
        "एट": "8", "आठ": "8",
        "नाइन": "9", "नौ": "9",
        "जीरो": "0", "शून्य": "0"
    }

    normalized = text
    # Apply replacements (longest patterns first to avoid partial matching issues)
    for pattern in sorted(mapping.keys(), key=len, reverse=True):
        normalized = normalized.replace(pattern, mapping[pattern])

    # Convert separated "S H" to "SH"
    normalized = re.sub(r'\bS\s+H\b', 'SH', normalized, flags=re.IGNORECASE)

    # Collapse all spaces inside tracking IDs (e.g. "SH 7 7 7" -> "SH777")
    def collapse_sh_spaces(match):
        return match.group(0).replace(" ", "").replace("\t", "")

    normalized = re.sub(r'\bSH(?:\s*\d)+\b', collapse_sh_spaces, normalized, flags=re.IGNORECASE)

    return normalized

def get_localized_tracking_response(tracking_id: str, status: str, location: str, lang_key: str) -> str:
    """
    Builds the localized tracking response based on template format.
    """
    # Status localized translations
    status_map = {
        "te": {
            "in transit": "ఇన్ ట్రాన్సిట్",
            "out for delivery": "డెలివరీ కోసం అవుట్",
            "delayed": "ఆలస్యం",
            "cancelled": "రద్దు",
            "created": "సృష్టించబడింది"
        },
        "hi": {
            "in transit": "इन ट्रांजिट",
            "out for delivery": "डिलीवरी के लिए बाहर",
            "delayed": "देरी",
            "cancelled": "रद्द",
            "created": "तैयार"
        },
        "en": {
            "in transit": "In Transit",
            "out for delivery": "Out for Delivery",
            "delayed": "Delayed",
            "cancelled": "Cancelled",
            "created": "Created"
        }
    }

    # Location localized translations
    location_map = {
        "te": {
            "hyderabad": "హైదరాబాద్",
            "ghatkesar": "ఘట్కేసర్",
            "delhi hub": "ఢిల్లీ హబ్",
            "origin depot": "ఆరిజిన్ డిపో"
        },
        "hi": {
            "hyderabad": "हैदराबाद",
            "ghatkesar": "घटकेसर",
            "delhi hub": "दिल्ली हब",
            "origin depot": "ओरिजिन डिपो"
        },
        "en": {
            "hyderabad": "Hyderabad",
            "ghatkesar": "Ghatkesar",
            "delhi hub": "Delhi Hub",
            "origin depot": "Origin Depot"
        }
    }

    # Normalize inputs
    status_key = status.lower()
    location_key = location.lower()
    
    # Resolve correct language code (e.g. te-IN or te -> te)
    l_key = "en"
    if "te" in lang_key.lower() or "telugu" in lang_key.lower():
        l_key = "te"
    elif "hi" in lang_key.lower() or "hindi" in lang_key.lower():
        l_key = "hi"

    # Map status and location
    mapped_status = status_map[l_key].get(status_key, status)
    mapped_location = location_map[l_key].get(location_key, location)

    if l_key == "te":
        return f"మీ రవాణా పొట్లం పొడవునా చూస్తే, ఐడి {tracking_id} ప్రస్తుతం {mapped_location} లో {mapped_status} లో ఉంది."
    elif l_key == "hi":
        return f"आपका शिपमेंट {tracking_id} वर्तमान में {mapped_location} में {mapped_status} में है। "
    else:
        return f"Your shipment {tracking_id} is currently {mapped_status} at the {mapped_location} hub."

import io
import tempfile
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
import re
import random
import json
import uuid

import webrtcvad
import stt_services
from transcript_logger import log_transcript
import database
import llm_service
import tts_service
TRANSLATIONS = {
    "te": {
        "schedule_success": "నేను మీ షిప్‌మెంట్‌ను విజయవంతంగా షెడ్యూల్ చేసాను! మీ కొత్త ట్రాకింగ్ ఐడి {tracking_id}.",
        "schedule_error": "షిప్‌మెంట్‌ను షెడ్యూల్ చేయడంలో లోపం సంభవించింది: {error}",
        "cancel_success": "నేను షిప్‌మెంట్ {tracking_id}ను విజయవంతంగా రద్దు చేసాను.",
        "cancel_already_cancelled": "షిప్‌మెంట్ {tracking_id} ఇప్పటికే రద్దు చేయబడింది.",
        "cancel_already_picked_up": "క్షమించండి, షిప్‌మెంట్ {tracking_id} ఇప్పటికే పికప్ చేయబడినందున రద్దు చేయబడదు.",
        "cancel_not_found": "రద్దు చేయడానికి {tracking_id} ఐడితో సరిపోలే షిప్‌మెంట్ నాకు కనుగొనబడలేదు.",
        "cancel_invalid_id": "మీ ఆర్డర్‌ను రద్దు చేయడానికి దయచేసి సరైన ట్రాకింగ్ ఐడిని అందించండి.",
        "cancel_error": "షిప్‌మెంట్‌ను రద్దు చేయడంలో లోపం సంభవించింది: {error}",
        "end_session": "LogiRoute Expressని ఉపయోగించినందుకు ధన్యవాదాలు! మంచి రోజు ఉండాలని కోరుకుంటున్నాను, సెలవు."
    },
    "hi": {
        "schedule_success": "मैंने आपका शिपमेंट सफलतापूर्वक शेड्यूल कर दिया है! आपकी नई ट्रैकिंग आईडी {tracking_id} है।",
        "schedule_error": "शिपमेंट शेड्यूल करने में त्रुटि: {error}",
        "cancel_success": "मैंने शिपमेंट {tracking_id} को सफलतापूर्वक रद्द कर दिया है।",
        "cancel_already_cancelled": "शिपमेंट {tracking_id} पहले से ही रद्द है।",
        "cancel_already_picked_up": "क्षमा करें, शिपमेंट {tracking_id} को रद्द नहीं किया जा सकता क्योंकि इसे पहले ही पिकअप कर लिया गया है।",
        "cancel_not_found": "मुझे रद्द करने के लिए {tracking_id} आईडी वाला कोई शिपमेंट नहीं मिला।",
        "cancel_invalid_id": "कृपया अपना ऑर्डर रद्द करने के लिए एक वैध ट्रैकिंग आईडी प्रदान करें।",
        "cancel_error": "शिपमेंट रद्द करने में त्रुटि: {error}",
        "end_session": "LogiRoute एक्सप्रेस का उपयोग करने के लिए धन्यवाद! आपका दिन शुभ हो, अलविदा।"
    },
    "en": {
        "schedule_success": "I have successfully scheduled your shipment! Your new tracking ID is {tracking_id}.",
        "schedule_error": "Error scheduling order: {error}",
        "cancel_success": "I have successfully cancelled shipment {tracking_id}.",
        "cancel_already_cancelled": "Shipment {tracking_id} is already cancelled.",
        "cancel_already_picked_up": "I'm sorry, shipment {tracking_id} cannot be cancelled because it has already been picked up.",
        "cancel_not_found": "I couldn't find a shipment matching ID {tracking_id} to cancel.",
        "cancel_invalid_id": "Please provide a valid tracking ID to cancel your order.",
        "cancel_error": "Error cancelling order: {error}",
        "end_session": "Thank you for using LogiRoute Express! Have a wonderful day, goodbye."
    }
}

def get_language_from_code_or_name(lang: str) -> str:
    if not lang:
        return "en"
    lang_lower = lang.lower()
    if "te" in lang_lower or "telugu" in lang_lower:
        return "te"
    if "hi" in lang_lower or "hindi" in lang_lower:
        return "hi"
    return "en"

UNIVERSAL_FILLER_WORDS = {"hello", "hi", "hey", "ok", "okay", "thank you", "thanks", "bye", "goodbye", "yes", "no", "sure"}

def detect_language_override(text: str) -> str | None:
    """
    Screens the transcript for explicit language switch phrases.
    Returns: "English", "Telugu", "Hindi", or None if no override keyword is found.
    Bypasses override check if text consists purely of numbers/digits and/or tokens in UNIVERSAL_FILLER_WORDS.
    """
    if not text:
        return None

    # Lowercase and strip punctuation
    cleaned = text.lower().strip()
    punctuation = '.,?!;:"\'()[]{}<>-।`~@#$%^&*_+='
    cleaned = "".join(c for c in cleaned if c not in punctuation)
    
    # Strip digits/numbers
    cleaned_no_digits = re.sub(r'\d+', '', cleaned)
    
    # Extract words
    words = cleaned_no_digits.split()
    
    # Check if all remaining words are in UNIVERSAL_FILLER_WORDS
    if not words or all(w in UNIVERSAL_FILLER_WORDS for w in words):
        return None

    text_lower = text.lower()
    # Check English
    if "switch to english" in text_lower or "speak in english" in text_lower or "english translation" in text_lower or "अंग्रेजी में बोलो" in text_lower or "ఇంగ్లీష్" in text_lower:
        return "English"
    # Check Telugu
    if "telugu to change" in text_lower or "speak in telugu" in text_lower or "switch to telugu" in text_lower or "telugu lo matladu" in text_lower or "తెలుగులో మాట్లాడు" in text_lower or "తెలుగు" in text_lower:
        return "Telugu"
    # Check Hindi
    if "hindi lo matladu" in text_lower or "speak in hindi" in text_lower or "switch to hindi" in text_lower or "hindi mein bolo" in text_lower or "हिंदी में बोलो" in text_lower or "हिंदी" in text_lower:
        return "Hindi"
    return None


def match_conversational_phrase(text: str) -> str | None:
    """
    Checks if the normalized text is a simple greeting, okay acknowledgment, or thank you closing.
    Returns: 'thank_you', 'okay', 'greeting', or None.
    """
    if not text:
        return None
    
    cleaned = text.lower().strip()
    punctuation = '.,?!;:"\'()[]{}<>-।`~@#$%^&*_+='
    cleaned = "".join(c for c in cleaned if c not in punctuation)
    
    thank_you_patterns = {
        "thank you", "thanks", "thankyou", 
        "థాంక్యూ", "థాంక్స్", 
        "धन्यवाद", "थैंक यू", "शुक्रिया"
    }
    okay_patterns = {
        "okay", "ok", "fine", "sure", 
        "సరే", "ఓకే", 
        "ठीक है", "ओके"
    }
    hello_patterns = {
        "hello", "hi", "hey", "greetings", 
        "హలో", "నమస్కారం", 
        "नमस्ते", "नमस्कार"
    }
    
    if cleaned in thank_you_patterns:
        return "thank_you"
    if cleaned in okay_patterns:
        return "okay"
    if cleaned in hello_patterns:
        return "greeting"
        
    return None

def get_localized_phrase_response(phrase_type: str, lang_key: str) -> str:
    """
    Returns localized responses for common conversational acknowledgements/phrases.
    """
    l_key = "en"
    if "te" in lang_key.lower():
        l_key = "te"
    elif "hi" in lang_key.lower():
        l_key = "hi"

    phrase_matrix = {
        "te": {
            "thank_you": "మీకు స్వాగతం! మీకు సహాయం చేయడానికి సంతోషిస్తున్నాను. ఇంకా ఏదైనా సహాయం కావాలా?",
            "okay": "సరే, మీకు ఇంకా ఏమైనా సహాయం కావాలా?",
            "greeting": "హలో! లాజిరూట్ ఎక్స్‌ప్రెస్‌కు స్వాగతం. ఈరోజు నేను మీకు ఎలా సహాయం చేయగలను?"
        },
        "hi": {
            "thank_you": "आपका स्वागत है! मुझे आपकी सहायता करके खुशी हुई। क्या आपको कोई और सहायता चाहिए?",
            "okay": "ठीक है, क्या आपको कोई और सहायता चाहिए?",
            "greeting": "नमस्ते! लॉजीरूट एक्सप्रेस में आपका स्वागत है। आज मैं आपकी क्या सहायता कर सकता हूँ?",
        },
        "en": {
            "thank_you": "You're welcome! I'm glad I could help. Do you need any further assistance?",
            "okay": "Okay, is there anything else I can help you with?",
            "greeting": "Hello! Welcome to LogiRoute Express. How can I assist you today?"
        }
    }
    
    return phrase_matrix[l_key].get(phrase_type, phrase_matrix[l_key]["greeting"])


def normalize_indic_alphanumerics(text: str) -> str:
    """
    Normalizes Telugu and Hindi phonetic letters and digits to standard English alpha-numerics.
    """
    if not text:
        return text

    # Base dictionary for mapping
    mapping = {
        # Letter replacements
        "エス ヘッチ": "SH",
        "ఎస్ హెచ్": "SH",
        "एस एच": "SH",
        "एसएच": "SH",
        "s.h.": "SH",
        "s h": "SH",
        "s-h": "SH",
        
        # Telugu digits
        "వన్": "1",
        "ఒన్": "1",
        "టూ": "2",
        "త్రీ": "3",
        "ఫోర్": "4",
        "ఫైవ్": "5",
        
        # Hindi digits
        "वन": "1",
        "टू": "2",
        "थ्री": "3",
        "फोर": "4",
        "फाइव": "5"
    }

    # Case-insensitive replacement helper
    def replace_case_insensitive(s, old, new):
        pattern = re.compile(re.escape(old), re.IGNORECASE)
        return pattern.sub(new, s)

    normalized = text
    # Apply replacements (longest patterns first)
    for pattern in sorted(mapping.keys(), key=len, reverse=True):
        normalized = replace_case_insensitive(normalized, pattern, mapping[pattern])

    # Collapse spaces in any "SH" followed by digits (e.g. "SH 1 2 3" -> "SH123")
    def collapse_sh_spaces(match):
        return match.group(0).replace(" ", "").replace("\t", "").upper()

    normalized = re.sub(r'(?i)\bSH[\s\d]+', collapse_sh_spaces, normalized)

    return normalized

def get_localized_tracking_response(tracking_id: str, status: str, location: str, lang_key: str) -> str:
    """
    Builds the localized tracking response based on template format.
    """
    # Status localized translations
    status_map = {
        "te": {
            "in transit": "ఇన్ ట్రాన్సిట్",
            "out for delivery": "డెలివరీ కోసం అవుట్",
            "delayed": "ఆలస్యం",
            "cancelled": "రద్దు",
            "created": "సృష్టించబడింది"
        },
        "hi": {
            "in transit": "इन ट्रांजिट",
            "out for delivery": "डिलीवरी के लिए बाहर",
            "delayed": "देरी",
            "cancelled": "रद्द",
            "created": "तैयार"
        },
        "en": {
            "in transit": "In Transit",
            "out for delivery": "Out for Delivery",
            "delayed": "Delayed",
            "cancelled": "Cancelled",
            "created": "Created"
        }
    }

    # Location localized translations
    location_map = {
        "te": {
            "hyderabad": "హైదరాబాద్",
            "ghatkesar": "ఘట్కేసర్",
            "delhi hub": "ఢిల్లీ హబ్",
            "origin depot": "ఆరిజిన్ డిపో"
        },
        "hi": {
            "hyderabad": "हैदराबाद",
            "ghatkesar": "घटकेसर",
            "delhi hub": "दिल्ली हब",
            "origin depot": "ओरिजिन डिपो"
        },
        "en": {
            "hyderabad": "Hyderabad",
            "ghatkesar": "Ghatkesar",
            "delhi hub": "Delhi Hub",
            "origin depot": "Origin Depot"
        }
    }

    # Normalize inputs
    status_key = status.lower()
    location_key = location.lower()
    
    # Resolve correct language code (e.g. te-IN or te -> te)
    l_key = "en"
    if "te" in lang_key.lower() or "telugu" in lang_key.lower():
        l_key = "te"
    elif "hi" in lang_key.lower() or "hindi" in lang_key.lower():
        l_key = "hi"

    # Map status and location
    mapped_status = status_map[l_key].get(status_key, status)
    mapped_location = location_map[l_key].get(location_key, location)

    if l_key == "te":
        return f"మీ రవాణా పొట్లం పొడవునా చూస్తే, ఐడి {tracking_id} ప్రస్తుతం {mapped_location} లో {mapped_status} లో ఉంది."
    elif l_key == "hi":
        return f"आपका शिपमेंट {tracking_id} वर्तमान में {mapped_location} में {mapped_status} में है।"
    else:
        return f"Your shipment {tracking_id} is currently {mapped_status} at the {mapped_location} hub."


def classify_intent(text: str) -> str:
    """
    Classifies the user text into one of exactly 5 core intents:
    - TRACK_SHIPMENT
    - SCHEDULE_PICKUP
    - DELIVERY_DELAY_INQUIRY
    - GENERAL_INQUIRY
    - GOODBYE
    """
    if not text:
        return "GENERAL_INQUIRY"
        
    t = text.lower().strip()
    
    # Do not treat cancellations as tracking shipment even if they contain tracking ID
    if "cancel" in t or "రద్దు" in t or "रद्द" in t:
        return "GENERAL_INQUIRY"
    
    # GOODBYE intent
    goodbye_words = {
        "bye", "goodbye", "thank you", "thanks", "done", "completed", "no help", "nothing else", "no thanks",
        "థాంక్యూ", "థాంక్స్", "సెలవు", "వద్దు", "చాలు",
        "धन्यवाद", "अलविदा", "शुक्रिया", "बस", "नहीं चाहिए"
    }
    if any(w in t for w in goodbye_words):
        return "GOODBYE"
        
    # TRACK_SHIPMENT intent (including matching SH tracking ID pattern directly)
    track_words = {
        "track", "status", "where is", "where's", "tracking",
        "ట్రాక్", "ఎక్కడ", "స్థితి", "స్టేటస్",
        "ट्रैक", "स्थिति", "कहाँ है", "कहा है"
    }
    if any(w in t for w in track_words) or re.search(r"SH\d{3}", text, re.IGNORECASE):
        return "TRACK_SHIPMENT"
        
    # SCHEDULE_PICKUP intent
    pickup_words = {
        "schedule", "pickup", "pick up", "book", "courier", "send parcel", "create shipment", "place order", "place an order",
        "పికప్", "బుక్", "షెడ్యూల్", "పార్సెల్",
        "पिकअप", "बुक", "शेड्यूल", "पार्सल", "भेजना"
    }
    if any(w in t for w in pickup_words):
        return "SCHEDULE_PICKUP"
        
    # DELIVERY_DELAY_INQUIRY intent
    delay_words = {
        "delay", "delayed", "late", "not arrived", "postponed", "why is my package", "where is my delayed",
        "ఆలస్యం", "లేట్", "ఎందుకు రాలేదు",
        "देरी", "लेट", "क्यों नहीं आया"
    }
    if any(w in t for w in delay_words):
        return "DELIVERY_DELAY_INQUIRY"
        
    return "GENERAL_INQUIRY"


def is_off_topic_query(text: str) -> bool:
    """
    Detects if the user text is a completely off-topic request (like asking for a joke/weather)
    or contains gibberish/unparseable input.
    """
    if not text:
        return False
    t = text.lower().strip()
    off_topic_keywords = {
        "joke", "weather", "song", "story", "meaning of life", "who are you", "what is your name", "tell me a",
        "asdf", "qwerty", "zxcv"
    }
    if any(k in t for k in off_topic_keywords):
        return True
    # Check for gibberish (e.g. no vowels in a word of length >= 4)
    words = t.split()
    for w in words:
        if len(w) >= 4 and not any(v in w for v in "aeiouy\u0c05\u0c06\u0c07\u0c08\u0c09\u0c0a\u0c0b\u0c0c\u0c0e\u0c0f\u0c10\u0c12\u0c13\u0c14\u0905\u0906\u0907\u0908\u0909\u090a\u090b\u090c\u090f\u0910\u0913\u0914"):
            return True
    return False


active_workflow_sessions = set()

app = FastAPI(title="Voice Agent Translation & Transcription API")

@app.on_event("startup")
async def startup_event():
    database.init_db()


@app.post("/initiate_workflow")
async def initiate_workflow(data: dict = Body(None)):
    session_id = None
    if data:
        session_id = data.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    
    database.update_session_state(session_id, "GREETING", {}, "en-IN")
    active_workflow_sessions.add(session_id)
    
    welcome_text = "Hello! Welcome to Colaberry Logistics Support. How may I assist you today?"
    audio_b64 = tts_service.generate_speech_b64(text=welcome_text, target_language_code="en-IN")
    
    return {
        "session_id": session_id,
        "response_text": welcome_text,
        "audio_b64": audio_b64
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language_code: str = Form("en"),
    session_id: str = Form("default_session")
):
    """
    Exposes a POST endpoint /transcribe that accepts an UploadFile binary audio payload
    and a language_code form parameter string (defaulting to 'en').
    
    Performs local silence detection via RMS amplitude checks before sending to external APIs.
    """
    # Read incoming binary audio frames
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")

    # Try decoding audio data as float32 frames using soundfile
    try:
        audio_data, samplerate = sf.read(io.BytesIO(contents), dtype='float32')
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse audio file. Please upload a valid WAV, FLAC, or other supported audio format. Error: {str(e)}"
        )

    # Perform safety VAD check to catch empty/silent recordings locally
    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="No speech detected. Please try again.")

    # webrtcvad only supports 8000, 16000, 32000, 48000 Hz
    if samplerate not in (8000, 16000, 32000, 48000):
        raise HTTPException(
            status_code=400,
            detail=f"webrtcvad does not support audio sample rate {samplerate}. Must be 8kHz, 16kHz, 32kHz, or 48kHz."
        )

    # Convert to mono if multichannel
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        mono_data = audio_data[:, 0]
    else:
        mono_data = audio_data.flatten()

    # Simple and robust peak-based silence check on the server side
    # to avoid false VAD rejections of already client-gated voice streams.
    peak = np.max(np.abs(mono_data))
    if peak < 0.001:
        raise HTTPException(status_code=400, detail="No speech detected. Please try again.")

    # Write the bytes to a unique temp file safely to prevent concurrent access collisions
    temp_file_fd, temp_file_path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(temp_file_fd, "wb") as temp_file:
            temp_file.write(contents)
            
        # Dynamically forward to established stt_services based on language code
        active_lang = language_code
        if language_code == "auto":
            cached_lang = database.get_session_language(session_id)
            if cached_lang:
                if cached_lang == "Telugu" or cached_lang.startswith("te"):
                    active_lang = "te-IN"
                elif cached_lang == "Hindi" or cached_lang.startswith("hi"):
                    active_lang = "hi-IN"
                elif cached_lang == "English" or cached_lang.startswith("en"):
                    active_lang = "en-IN"
                else:
                    active_lang = cached_lang

        if active_lang == "auto":
            result = stt_services.transcribe_auto(temp_file_path)
            engine_used = result["telemetry"]["engine_used"]
            logged_lang = result["telemetry"].get("detected_language", "auto")
        elif active_lang.startswith("en"):
            engine_used = "Groq"
            result = stt_services.transcribe_english(temp_file_path)
            logged_lang = language_code if language_code != "auto" else "English"
        elif active_lang.startswith("te"):
            engine_used = "Sarvam"
            result = stt_services.transcribe_indic(temp_file_path, language_code="te-IN")
            logged_lang = language_code if language_code != "auto" else "Telugu"
        elif active_lang.startswith("hi"):
            engine_used = "Sarvam"
            result = stt_services.transcribe_indic(temp_file_path, language_code="hi-IN")
            logged_lang = language_code if language_code != "auto" else "Hindi"
        else:
            engine_used = "Sarvam"
            result = stt_services.transcribe_indic(temp_file_path, language_code=active_lang)
            logged_lang = language_code if language_code != "auto" else active_lang

        transcript = result["transcript"]
        duration = result["telemetry"]["duration_seconds"]
        if logged_lang == "auto" and "telemetry" in result and "detected_language" in result["telemetry"]:
            logged_lang = result["telemetry"]["detected_language"]

        # Normalize Telugu/Hindi digit and character spellings
        transcript = normalize_indic_alphanumerics(transcript)

        # Check if the session was initiated via /initiate_workflow
        is_workflow_session = session_id in active_workflow_sessions

        if is_workflow_session:
            # Get active conversation progress state
            current_stage, pending_slots, cached_lang = database.get_session_state(session_id)
            
            # Check for user override phrase to explicitly transition the sticky session language
            override_lang = detect_language_override(transcript)
            if override_lang:
                if override_lang == "Telugu":
                    logged_lang = "te-IN"
                elif override_lang == "Hindi":
                    logged_lang = "hi-IN"
                elif override_lang == "English":
                    logged_lang = "en-IN"
                else:
                    logged_lang = override_lang
            else:
                if cached_lang:
                    if cached_lang == "Telugu" or cached_lang.lower().startswith("te"):
                        logged_lang = "te-IN"
                    elif cached_lang == "Hindi" or cached_lang.lower().startswith("hi"):
                        logged_lang = "hi-IN"
                    elif cached_lang == "English" or cached_lang.lower().startswith("en"):
                        logged_lang = "en-IN"
                    else:
                        logged_lang = cached_lang
                else:
                    logged_lang = active_lang
        else:
            current_stage = "GREETING"
            pending_slots = {}
            cached_lang = database.get_session_language(session_id)
            
            override_lang = detect_language_override(transcript)
            if override_lang:
                logged_lang = override_lang
            else:
                if cached_lang:
                    if cached_lang == "Telugu" or cached_lang.lower().startswith("te"):
                        logged_lang = "te-IN"
                    elif cached_lang == "Hindi" or cached_lang.lower().startswith("hi"):
                        logged_lang = "hi-IN"
                    elif cached_lang == "English" or cached_lang.lower().startswith("en"):
                        logged_lang = "en-IN"
                    else:
                        logged_lang = cached_lang
                else:
                    if language_code == "auto":
                        logged_lang = result["telemetry"].get("detected_language", "English")
                    else:
                        if active_lang.startswith("en"):
                            logged_lang = language_code if language_code != "auto" else "English"
                        elif active_lang.startswith("te"):
                            logged_lang = language_code if language_code != "auto" else "Telugu"
                        elif active_lang.startswith("hi"):
                            logged_lang = language_code if language_code != "auto" else "Hindi"
                        else:
                            logged_lang = language_code if language_code != "auto" else active_lang

        ai_response = None
        tool_calls = None
        new_session_id = None
        tracking_id = None
        context_data = None
        mode = "general"

        if is_workflow_session:
            # Explicit Intent Router
            intent = classify_intent(transcript)

            # Unexpected Input Deflection check
            is_gathering_parameters = (
                current_stage in ("WAITING_FOR_TRACKING_ID", "WAITING_FOR_DELAY_ID") or 
                current_stage.startswith("PICKUP_WAITING_")
            )
            
            is_off_topic = False
            if is_gathering_parameters:
                if current_stage in ("WAITING_FOR_TRACKING_ID", "WAITING_FOR_DELAY_ID"):
                    has_tracking_id = bool(re.search(r"SH\d{3}", transcript, re.IGNORECASE))
                    if not has_tracking_id and is_off_topic_query(transcript):
                        is_off_topic = True
                elif current_stage.startswith("PICKUP_WAITING_"):
                    if is_off_topic_query(transcript):
                        is_off_topic = True

            if is_off_topic:
                if "te" in logged_lang.lower():
                    ai_response = "నేను అర్థం చేసుకున్నాను, కానీ మీ అభ్యర్థనను పూర్తి చేయడానికి నేను కోరిన లాజిస్టిక్స్ వివరాలను అందించాలి. దయచేసి వివరాలను అందించండి."
                elif "hi" in logged_lang.lower():
                    ai_response = "मैं समझता हूँ, लेकिन आपके अनुरोध को पूरा करने के लिए मुझे आपके द्वारा अनुरोधित रसद विवरण प्रदान करने की आवश्यकता है। कृपया विवरण प्रदान करें।"
                else:
                    ai_response = "I understand, but to help you with your request, I need you to provide the requested logistics information. Please provide the details."
                database.update_session_state(session_id, current_stage, pending_slots, logged_lang)
                
            elif current_stage == "OFFER_ADDITIONAL_ASSISTANCE" and intent == "GOODBYE":
                current_stage = "COMPLETED"
                if "te" in logged_lang.lower():
                    ai_response = "కొలాబెర్రీ లాజిస్టిక్స్ సపోర్ట్‌ను సంప్రదించినందుకు ధన్యవాదాలు! మీ రోజు బాగుండాలని కోరుకుంటున్నాను."
                elif "hi" in logged_lang.lower():
                    ai_response = "कोलैबेरी लॉजिस्टिक्स सपोर्ट से संपर्क करने के लिए धन्यवाद! आपका दिन शुभ हो।"
                else:
                    ai_response = "Thank you for contacting Colaberry Logistics Support. Have a great day!"
                database.update_session_state(session_id, current_stage, {}, logged_lang)

            elif intent == "TRACK_SHIPMENT" or current_stage == "WAITING_FOR_TRACKING_ID":
                match = re.search(r"SH\d{3}", transcript, re.IGNORECASE)
                if match:
                    tracking_id = match.group(0).upper()
                    context_data = database.get_shipment_details(tracking_id)
                    if context_data:
                        lang_key = get_language_from_code_or_name(logged_lang)
                        ai_response = get_localized_tracking_response(
                            tracking_id=tracking_id,
                            status=context_data.get("status"),
                            location=context_data.get("current_location"),
                            lang_key=lang_key
                        )
                        current_stage = "OFFER_ADDITIONAL_ASSISTANCE"
                        database.update_session_state(session_id, current_stage, {}, logged_lang)
                    else:
                        if "te" in logged_lang.lower():
                            ai_response = "ఆ ఐడితో సరిపోలే షిప్‌మెంట్ నాకు కనుగొనబడలేదు. దయచేసి నంబర్‌ను సరిచూసి మళ్లీ ప్రయత్నించండి."
                        elif "hi" in logged_lang.lower():
                            ai_response = "मुझे उस आईडी से मेल खाने वाला कोई शिपमेंट नहीं मिला। कृपया नंबर दोबारा जांचें और फिर प्रयास करें।"
                        else:
                            ai_response = "I couldn't find a shipment matching that ID. Please double-check the number and try again."
                        current_stage = "WAITING_FOR_TRACKING_ID"
                        database.update_session_state(session_id, current_stage, {}, logged_lang)
                else:
                    if "te" in logged_lang.lower():
                        ai_response = "దయచేసి మీ షిప్‌మెంట్ ఐడిని అందించండి."
                    elif "hi" in logged_lang.lower():
                        ai_response = "कृपया अपनी शिपमेंट आईडी प्रदान करें।"
                    else:
                        ai_response = "Please provide your shipment ID."
                    current_stage = "WAITING_FOR_TRACKING_ID"
                    database.update_session_state(session_id, current_stage, {}, logged_lang)

            elif intent == "SCHEDULE_PICKUP" or current_stage.startswith("PICKUP_"):
                if not current_stage.startswith("PICKUP_"):
                    current_stage = "PICKUP_GATHERING_SLOTS"
                    pending_slots = {}
                    weight_match = re.search(r'\b(\d+(?:\.\d+)?\s*(?:kg|kgs|kilogram|kilograms|lbs|pounds|కేజీ|కేజీలు|కిలో|కిలోలు|किलो|किलोग्राम|g))\b', transcript, re.IGNORECASE)
                    if weight_match:
                        pending_slots["package_weight"] = weight_match.group(0)
                    date_match = re.search(r'\b(tomorrow|today|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|రేపు|ఈ రోజు|కల్|आज|परसों)\b', transcript, re.IGNORECASE)
                    if date_match:
                        pending_slots["delivery_date"] = date_match.group(0)
                    loc_match = re.search(r'\b(hyderabad|ghatkesar|delhi|mumbai|kphb|hitech city|secunderabad|బెంగళూరు|వరంగల్|मुंबई|दिल्ली|हैदराबाद|बेंगलुरु)\b', transcript, re.IGNORECASE)
                    if loc_match:
                        pending_slots["pickup_location"] = loc_match.group(0)
                else:
                    if current_stage == "PICKUP_WAITING_FOR_LOCATION":
                        pending_slots["pickup_location"] = transcript
                    elif current_stage == "PICKUP_WAITING_FOR_DATE":
                        pending_slots["delivery_date"] = transcript
                    elif current_stage == "PICKUP_WAITING_FOR_WEIGHT":
                        pending_slots["package_weight"] = transcript

                if not pending_slots.get("pickup_location"):
                    current_stage = "PICKUP_WAITING_FOR_LOCATION"
                    if "te" in logged_lang.lower():
                        ai_response = "దయచేసి పికప్ లొకేషన్‌ను అందించండి."
                    elif "hi" in logged_lang.lower():
                        ai_response = "कृपया पिकअप स्थान प्रदान करें।"
                    else:
                        ai_response = "Please provide the pickup location."
                elif not pending_slots.get("delivery_date"):
                    current_stage = "PICKUP_WAITING_FOR_DATE"
                    if "te" in logged_lang.lower():
                        ai_response = "దయచేసి డెలివరీ తేదీని అందించండి."
                    elif "hi" in logged_lang.lower():
                        ai_response = "कृपया डिलीवरी की तारीख प्रदान करें।"
                    else:
                        ai_response = "Please provide the delivery date."
                elif not pending_slots.get("package_weight"):
                    current_stage = "PICKUP_WAITING_FOR_WEIGHT"
                    if "te" in logged_lang.lower():
                        ai_response = "దయచేసి ప్యాకేజీ బరువును అందించండి."
                    elif "hi" in logged_lang.lower():
                        ai_response = "कृपया पैकेज का वजन प्रदान करें।"
                    else:
                        ai_response = "Please provide the package weight."
                else:
                    while True:
                        generated_id = f"SH{random.randint(100, 999)}"
                        if not database.get_shipment_details(generated_id):
                            tracking_id = generated_id
                            break
                    database.insert_new_order(
                        tracking_id=tracking_id,
                        p_addr=pending_slots["pickup_location"],
                        d_addr="Delhi Hub",
                        weight=pending_slots["package_weight"],
                        p_time=pending_slots["delivery_date"]
                    )
                    lang_key = get_language_from_code_or_name(logged_lang)
                    t = TRANSLATIONS[lang_key]
                    ai_response = t["schedule_success"].format(tracking_id=tracking_id)
                    current_stage = "OFFER_ADDITIONAL_ASSISTANCE"
                    pending_slots = {}

                database.update_session_state(session_id, current_stage, pending_slots, logged_lang)

            elif intent == "DELIVERY_DELAY_INQUIRY" or current_stage == "WAITING_FOR_DELAY_ID":
                match = re.search(r"SH\d{3}", transcript, re.IGNORECASE)
                if match:
                    tracking_id = match.group(0).upper()
                    context_data = database.get_shipment_details(tracking_id)
                    if context_data:
                        status = context_data.get("status", "")
                        location = context_data.get("current_location", "")
                        eta = context_data.get("eta_days", 0)
                        if status == "Delayed":
                            if "te" in logged_lang.lower():
                                ai_response = f"మీ రవాణా పొట్లం {tracking_id} ఆలస్యమైంది. ఇది ప్రస్తుతం {location} లో ఉంది. ఇది చేరడానికి మరో {eta} రోజులు పడుతుంది."
                            elif "hi" in logged_lang.lower():
                                ai_response = f"आपका शिपमेंट {tracking_id} विलंबित है। यह वर्तमान में {location} में है। इसमें {eta} दिन और लगेंगे।"
                            else:
                                ai_response = f"Your shipment {tracking_id} is delayed. It is currently at the {location} hub. It is expected to arrive in {eta} days."
                        else:
                            lang_key = get_language_from_code_or_name(logged_lang)
                            ai_response = get_localized_tracking_response(
                                tracking_id=tracking_id,
                                status=status,
                                location=location,
                                lang_key=lang_key
                            )
                        current_stage = "OFFER_ADDITIONAL_ASSISTANCE"
                        database.update_session_state(session_id, current_stage, {}, logged_lang)
                    else:
                        if "te" in logged_lang.lower():
                            ai_response = "ఆ ఐడితో సరిపోలే షిప్‌మెంట్ నాకు కనుగొనబడలేదు. దయచేసి నంబర్‌ను సరిచూసి మళ్లీ ప్రయత్నించండి."
                        elif "hi" in logged_lang.lower():
                            ai_response = "मुझे उस आईडी से मेल खाने वाला कोई शिपमेंट नहीं मिला। कृपया नंबर दोबारा जांचें और फिर प्रयास करें।"
                        else:
                            ai_response = "I couldn't find a shipment matching that ID. Please double-check the number and try again."
                        current_stage = "WAITING_FOR_DELAY_ID"
                        database.update_session_state(session_id, current_stage, {}, logged_lang)
                else:
                    if "te" in logged_lang.lower():
                        ai_response = "ఆ ఐడితో సరిపోలే షిప్‌మెంట్ నాకు కనుగొనబడలేదు. దయచేసి నంబర్‌ను సరిచూసి మళ్లీ ప్రయత్నించండి."
                    elif "hi" in logged_lang.lower():
                        ai_response = "मुझे उस आईडी से मेल खाने वाला कोई शिपमेंट नहीं मिला। कृपया नंबर दोबारा जांचें और फिर प्रयास करें।"
                    else:
                        ai_response = "I couldn't find a shipment matching that ID. Please double-check the number and try again."
                    if current_stage != "WAITING_FOR_DELAY_ID":
                        if "te" in logged_lang.lower():
                            ai_response = "దయచేసి మీ షిప్‌మెంట్ ఐడిని అందించండి."
                        elif "hi" in logged_lang.lower():
                            ai_response = "कृपया अपनी शिपमेंट आईडी प्रदान करें।"
                        else:
                            ai_response = "Please provide your shipment ID."
                    current_stage = "WAITING_FOR_DELAY_ID"
                    database.update_session_state(session_id, current_stage, {}, logged_lang)

            else:
                phrase_type = match_conversational_phrase(transcript)
                if phrase_type and cached_lang:
                    ai_response = get_localized_phrase_response(phrase_type, logged_lang)
                else:
                    res_dict = llm_service.generate_response(
                        user_text=transcript,
                        session_id=session_id,
                        context_data=context_data,
                        mode=mode
                    )
                    ai_response = res_dict["ai_response"]
                    tool_calls = res_dict["tool_calls"]

                if tool_calls:
                    lang_key = get_language_from_code_or_name(logged_lang)
                    t = TRANSLATIONS[lang_key]
                    for tool_call in tool_calls:
                        func_name = tool_call.function.name
                        if func_name == "create_new_shipment_record":
                            try:
                                args = json.loads(tool_call.function.arguments)
                                pickup_address = args.get("pickup_address")
                                destination_address = args.get("destination_address")
                                package_weight = args.get("package_weight")
                                pickup_time = args.get("pickup_time")
                                
                                while True:
                                    generated_id = f"SH{random.randint(100, 999)}"
                                    if not database.get_shipment_details(generated_id):
                                        tracking_id = generated_id
                                        break
                                        
                                database.insert_new_order(
                                    tracking_id=tracking_id,
                                    p_addr=pickup_address,
                                    d_addr=destination_address,
                                    weight=package_weight,
                                    p_time=pickup_time
                                )
                                ai_response = t["schedule_success"].format(tracking_id=tracking_id)
                            except Exception as e:
                                ai_response = t["schedule_error"].format(error=str(e))
                        elif func_name == "end_current_session":
                            new_session_id = str(uuid.uuid4())
                            ai_response = t["end_session"]
                        elif func_name == "cancel_shipment_order":
                            try:
                                args = json.loads(tool_call.function.arguments)
                                t_id = args.get("tracking_id")
                                if t_id:
                                    tracking_id = t_id.strip().upper()
                                    result = database.cancel_shipment_order(tracking_id)
                                    if result == "SUCCESS":
                                        ai_response = t["cancel_success"].format(tracking_id=tracking_id)
                                    elif result == "ALREADY_CANCELLED":
                                        ai_response = t["cancel_already_cancelled"].format(tracking_id=tracking_id)
                                    elif result == "ALREADY_PICKED_UP":
                                        ai_response = t["cancel_already_picked_up"].format(tracking_id=tracking_id)
                                    elif result == "NOT_FOUND":
                                        ai_response = t["cancel_not_found"].format(tracking_id=tracking_id)
                                else:
                                    ai_response = t["cancel_invalid_id"]
                            except Exception as e:
                                ai_response = t["cancel_error"].format(error=str(e))
        else:
            # Legacy/Direct LLM-based logic
            phrase_type = match_conversational_phrase(transcript)
            match = re.search(r"SH\d{3}", transcript, re.IGNORECASE)
            if match:
                tracking_id = match.group(0).upper()
                context_data = database.get_shipment_details(tracking_id)
                mode = "tracking"

            if phrase_type and cached_lang:
                ai_response = get_localized_phrase_response(phrase_type, logged_lang)
            elif mode == "tracking" and context_data:
                lang_key = get_language_from_code_or_name(logged_lang)
                ai_response = get_localized_tracking_response(
                    tracking_id=tracking_id,
                    status=context_data.get("status"),
                    location=context_data.get("current_location"),
                    lang_key=lang_key
                )
            else:
                res_dict = llm_service.generate_response(
                    user_text=transcript,
                    session_id=session_id,
                    context_data=context_data,
                    mode=mode
                )
                ai_response = res_dict["ai_response"]
                tool_calls = res_dict["tool_calls"]

            if tool_calls:
                lang_key = get_language_from_code_or_name(logged_lang)
                t = TRANSLATIONS[lang_key]
                for tool_call in tool_calls:
                    func_name = tool_call.function.name
                    if func_name == "create_new_shipment_record":
                        try:
                            args = json.loads(tool_call.function.arguments)
                            pickup_address = args.get("pickup_address")
                            destination_address = args.get("destination_address")
                            package_weight = args.get("package_weight")
                            pickup_time = args.get("pickup_time")
                            
                            while True:
                                generated_id = f"SH{random.randint(100, 999)}"
                                if not database.get_shipment_details(generated_id):
                                    tracking_id = generated_id
                                    break
                                    
                            database.insert_new_order(
                                tracking_id=tracking_id,
                                p_addr=pickup_address,
                                d_addr=destination_address,
                                weight=package_weight,
                                p_time=pickup_time
                            )
                            ai_response = t["schedule_success"].format(tracking_id=tracking_id)
                        except Exception as e:
                            ai_response = t["schedule_error"].format(error=str(e))
                    elif func_name == "end_current_session":
                        new_session_id = str(uuid.uuid4())
                        ai_response = t["end_session"]
                    elif func_name == "cancel_shipment_order":
                        try:
                            args = json.loads(tool_call.function.arguments)
                            t_id = args.get("tracking_id")
                            if t_id:
                                tracking_id = t_id.strip().upper()
                                result = database.cancel_shipment_order(tracking_id)
                                if result == "SUCCESS":
                                    ai_response = t["cancel_success"].format(tracking_id=tracking_id)
                                elif result == "ALREADY_CANCELLED":
                                    ai_response = t["cancel_already_cancelled"].format(tracking_id=tracking_id)
                                elif result == "ALREADY_PICKED_UP":
                                    ai_response = t["cancel_already_picked_up"].format(tracking_id=tracking_id)
                                elif result == "NOT_FOUND":
                                    ai_response = t["cancel_not_found"].format(tracking_id=tracking_id)
                            else:
                                ai_response = t["cancel_invalid_id"]
                        except Exception as e:
                            ai_response = t["cancel_error"].format(error=str(e))

        # Generate TTS audio payload
        audio_b64 = tts_service.generate_speech_b64(text=ai_response, target_language_code=logged_lang)

        # Save transactional details to the structured log file
        log_transcript(
            audio_filename=file.filename,
            transcribed_text=transcript,
            processing_duration_sec=duration,
            engine_used=engine_used,
            language_code=logged_lang
        )

        # Log interaction to SQLite database
        database.log_interaction(
            session_id=session_id,
            audio_file=file.filename,
            transcript=transcript,
            response=ai_response,
            lang=logged_lang,
            engine="openai/gpt-4o-mini",
            latency=int(duration * 1000),
            tracking_id=tracking_id
        )

        resp_payload = {
            "transcript": transcript,
            "ai_response": ai_response,
            "telemetry": {
                "duration_seconds": duration,
                "engine_used": engine_used
            },
            "audio_b64": audio_b64
        }
        if new_session_id:
            resp_payload["new_session_id"] = new_session_id

        return resp_payload

    except Exception as e:
        # Standardize 500 error propagation if external APIs fail
        raise HTTPException(status_code=500, detail=f"Transcription pipeline error: {str(e)}")

    finally:
        # Securely remove temporary file to prevent system locks/leaks
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as cleanup_err:
                print(f"Warning: Failed to delete temp file {temp_file_path}: {cleanup_err}")


@app.post("/end_session")
async def end_session(session_id: str = Form("default_session")):
    """
    Direct endpoint to trigger end_current_session() tool logic for context rotation on timeout.
    """
    ai_response = "Thank you for using LogiRoute Express! Have a wonderful day, goodbye."
    new_session_id = str(uuid.uuid4())

    # Log interaction to SQLite database
    database.log_interaction(
        session_id=session_id,
        audio_file="timeout_signal",
        transcript="[Timeout Session Close]",
        response=ai_response,
        lang="en",
        engine="openai/gpt-4o-mini",
        latency=0,
        tracking_id=None
    )

    return {
        "ai_response": ai_response,
        "new_session_id": new_session_id
    }

