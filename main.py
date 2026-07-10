import os
import io
import tempfile
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

app = FastAPI(title="Voice Agent Translation & Transcription API")

@app.on_event("startup")
async def startup_event():
    database.init_db()


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
        if language_code == "auto":
            result = stt_services.transcribe_auto(temp_file_path)
            engine_used = result["telemetry"]["engine_used"]
            logged_lang = result["telemetry"].get("detected_language", "auto")
        elif language_code.startswith("en"):
            engine_used = "Groq"
            result = stt_services.transcribe_english(temp_file_path)
            logged_lang = language_code
        else:
            engine_used = "Sarvam"
            result = stt_services.transcribe_indic(temp_file_path, language_code=language_code)
            logged_lang = language_code

        transcript = result["transcript"]
        duration = result["telemetry"]["duration_seconds"]

        # Parse tracking ID from transcript using case-insensitive regex SH\d{3}
        match = re.search(r"SH\d{3}", transcript, re.IGNORECASE)
        tracking_id = None
        context_data = None
        mode = "general"
        if match:
            tracking_id = match.group(0).upper()
            context_data = database.get_shipment_details(tracking_id)
            mode = "tracking"

        # Generate response using OpenAI LLM
        res_dict = llm_service.generate_response(
            user_text=transcript,
            session_id=session_id,
            context_data=context_data,
            mode=mode
        )
        ai_response = res_dict["ai_response"]
        tool_calls = res_dict["tool_calls"]
        new_session_id = None

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
                        
                        # Generate unique random tracking ID SH + 3 digits
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
                            # Update the outer tracking_id for database logging tracking_id_ref
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

