import unittest
import os
import io
import json
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import database
import main
from main import app

class TestMainAPI(unittest.TestCase):

    def setUp(self):
        # Patch database DB_FILE to avoid dirty data pollution
        self.db_file_patcher = patch('database.DB_FILE', 'test_voice_agent.db')
        self.db_file_patcher.start()
        
        if os.path.exists("test_voice_agent.db"):
            try:
                os.remove("test_voice_agent.db")
            except Exception:
                pass
        database.init_db()

        self.client = TestClient(app)
        self.test_log_file = "test_transcript_history.json"
        
        # Patch the LOG_FILE variable in transcript_logger to use our test log file
        self.log_file_patcher = patch('transcript_logger.LOG_FILE', self.test_log_file)
        self.log_file_patcher.start()

        # Patch new database and LLM service functions to avoid live calls in test
        self.gen_resp_patcher = patch('llm_service.generate_response', return_value={"ai_response": "Mocked AI Response", "tool_calls": None})
        self.mock_gen_resp = self.gen_resp_patcher.start()

        self.log_int_patcher = patch('database.log_interaction')
        self.mock_log_int = self.log_int_patcher.start()

        self.get_ship_patcher = patch('database.get_shipment_details', return_value=None)
        self.mock_get_ship = self.get_ship_patcher.start()

        # Patch the tts_service.generate_speech_b64 function
        self.tts_patcher = patch('tts_service.generate_speech_b64', return_value="dGVzdCBhdWRpbw==")
        self.mock_tts = self.tts_patcher.start()
        
        # Clean up any leftover test log file
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)
            
        # Generate silent WAV bytes (1 second of zeros at 16kHz)
        silent_buf = io.BytesIO()
        sf.write(silent_buf, np.zeros(16000), 16000, format='WAV')
        self.silent_wav_bytes = silent_buf.getvalue()

        # Generate loud WAV bytes (1 second sine wave at 16kHz, amplitude 0.5)
        loud_buf = io.BytesIO()
        t = np.linspace(0, 1, 16000, endpoint=False)
        sine = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        sf.write(loud_buf, sine, 16000, format='WAV')
        self.loud_wav_bytes = loud_buf.getvalue()

    def tearDown(self):
        self.db_file_patcher.stop()
        self.log_file_patcher.stop()
        self.gen_resp_patcher.stop()
        self.log_int_patcher.stop()
        self.get_ship_patcher.stop()
        self.tts_patcher.stop()
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)
        if os.path.exists("test_voice_agent.db"):
            try:
                os.remove("test_voice_agent.db")
            except Exception:
                pass


    @patch('stt_services.transcribe_sarvam')
    def test_transcribe_english_success(self, mock_transcribe_eng):
        # Mock transcription return values
        mock_transcribe_eng.return_value = {
            "transcript": "Hello, this is a test recording.",
            "telemetry": {
                "duration_seconds": 1.25
            }
        }

        # Send request with English tag
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en"}
        )

        # Assert response details
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["transcript"], "Hello, this is a test recording.")
        self.assertEqual(json_data["telemetry"]["engine_used"], "Sarvam")
        self.assertEqual(json_data["telemetry"]["duration_seconds"], 1.25)

        # Assert logging integration persists transaction correctly
        self.assertTrue(os.path.exists(self.test_log_file))
        with open(self.test_log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0]["audio_filename"], "speech.wav")
            self.assertEqual(logs[0]["transcribed_text"], "Hello, this is a test recording.")
            self.assertEqual(logs[0]["engine_used"], "Sarvam")
            self.assertEqual(logs[0]["language_code"], "English")

    @patch('stt_services.transcribe_sarvam')
    def test_transcribe_indic_success(self, mock_transcribe_ind):
        # Mock transcription return values
        mock_transcribe_ind.return_value = {
            "transcript": "नमस्ते, यह एक परीक्षण रिकॉर्डिंग है।",
            "telemetry": {
                "duration_seconds": 0.85
            }
        }

        # Send request with Hindi tag
        response = self.client.post(
            "/transcribe",
            files={"file": ("hindi.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "hi-IN"}
        )

        # Assert response details
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["transcript"], "नमस्ते, यह एक परीक्षण रिकॉर्डिंग है।")
        self.assertEqual(json_data["telemetry"]["engine_used"], "Sarvam")
        self.assertEqual(json_data["telemetry"]["duration_seconds"], 0.85)

        # Assert logging integration persists transaction correctly
        self.assertTrue(os.path.exists(self.test_log_file))
        with open(self.test_log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0]["audio_filename"], "hindi.wav")
            self.assertEqual(logs[0]["transcribed_text"], "नमस्ते, यह एक परीक्षण रिकॉर्डिंग है।")
            self.assertEqual(logs[0]["engine_used"], "Sarvam")
            self.assertEqual(logs[0]["language_code"], "Hindi")

    def test_transcribe_silence_rejection(self):
        # Send silent audio payload
        response = self.client.post(
            "/transcribe",
            files={"file": ("silence.wav", self.silent_wav_bytes, "audio/wav")},
            data={"language_code": "en"}
        )

        # Assert rejection status code and local rejection message
        self.assertEqual(response.status_code, 400)
        self.assertIn("No speech detected. Please try again.", response.json()["detail"])

        # Log file should not have been created
        self.assertFalse(os.path.exists(self.test_log_file))

    def test_transcribe_invalid_audio_format(self):
        # Send garbage file bytes
        response = self.client.post(
            "/transcribe",
            files={"file": ("bad_format.txt", b"not-audio-bytes-at-all", "text/plain")},
            data={"language_code": "en"}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Could not parse audio file", response.json()["detail"])
        self.assertFalse(os.path.exists(self.test_log_file))

    @patch('stt_services.transcribe_auto')
    def test_transcribe_auto_english_flow(self, mock_transcribe_auto):
        mock_transcribe_auto.return_value = {
            "transcript": "Hello, automated language detection.",
            "telemetry": {
                "duration_seconds": 1.1,
                "engine_used": "Sarvam",
                "detected_language": "English"
            }
        }

        response = self.client.post(
            "/transcribe",
            files={"file": ("auto_speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "auto"}
        )

        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["transcript"], "Hello, automated language detection.")
        self.assertEqual(json_data["telemetry"]["engine_used"], "Sarvam")

        # Check logs
        self.assertTrue(os.path.exists(self.test_log_file))
        with open(self.test_log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
            self.assertEqual(logs[0]["audio_filename"], "auto_speech.wav")
            self.assertEqual(logs[0]["engine_used"], "Sarvam")
            self.assertEqual(logs[0]["language_code"], "English")

    @patch('stt_services.transcribe_auto')
    def test_transcribe_auto_indic_flow(self, mock_transcribe_auto):
        mock_transcribe_auto.return_value = {
            "transcript": "नमस्ते",
            "telemetry": {
                "duration_seconds": 1.9,
                "engine_used": "Sarvam",
                "detected_language": "Hindi"
            }
        }

        response = self.client.post(
            "/transcribe",
            files={"file": ("auto_indic.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "auto"}
        )

        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["transcript"], "नमस्ते")
        self.assertEqual(json_data["telemetry"]["engine_used"], "Sarvam")

        # Check logs
        self.assertTrue(os.path.exists(self.test_log_file))
        with open(self.test_log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
            self.assertEqual(logs[0]["audio_filename"], "auto_indic.wav")
            self.assertEqual(logs[0]["engine_used"], "Sarvam")
            self.assertEqual(logs[0]["language_code"], "Hindi")

    @patch('stt_services.transcribe_sarvam')
    @patch('database.insert_new_order', return_value=True)
    def test_transcribe_tool_call_create_shipment(self, mock_insert, mock_transcribe_eng):
        mock_transcribe_eng.return_value = {
            "transcript": "Create a shipment please.",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        # Mock tool call structure
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "create_new_shipment_record"
        mock_tool_call.function.arguments = json.dumps({
            "pickup_address": "Office A",
            "destination_address": "Hub B",
            "package_weight": "2 kg",
            "pickup_time": "Tomorrow at 4 PM"
        })
        
        self.mock_gen_resp.return_value = {
            "ai_response": "",
            "tool_calls": [mock_tool_call]
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en", "session_id": "test_session_tool"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("I have successfully scheduled your shipment! Your new tracking ID is SH", json_data["ai_response"])
        mock_insert.assert_called_once()
        args, kwargs = mock_insert.call_args
        self.assertEqual(kwargs["p_addr"], "Office A")
        self.assertEqual(kwargs["d_addr"], "Hub B")
        self.assertEqual(kwargs["weight"], "2 kg")
        self.assertEqual(kwargs["p_time"], "Tomorrow at 4 PM")

    @patch('stt_services.transcribe_sarvam')
    def test_transcribe_tool_call_end_session(self, mock_transcribe_eng):
        mock_transcribe_eng.return_value = {
            "transcript": "Goodbye.",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "end_current_session"
        mock_tool_call.function.arguments = "{}"
        
        self.mock_gen_resp.return_value = {
            "ai_response": "",
            "tool_calls": [mock_tool_call]
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en", "session_id": "test_session_tool"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["ai_response"], "Thank you for using LogiRoute Express! Have a wonderful day, goodbye.")
        self.assertIn("new_session_id", json_data)

    @patch('stt_services.transcribe_sarvam')
    @patch('database.cancel_shipment_order', return_value="SUCCESS")
    def test_transcribe_tool_call_cancel_shipment_success(self, mock_cancel, mock_transcribe_eng):
        mock_transcribe_eng.return_value = {
            "transcript": "Please cancel my shipment SH888",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "cancel_shipment_order"
        mock_tool_call.function.arguments = json.dumps({"tracking_id": "SH888"})
        
        self.mock_gen_resp.return_value = {
            "ai_response": "",
            "tool_calls": [mock_tool_call]
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en", "session_id": "test_session_cancel"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["ai_response"], "I have successfully cancelled shipment SH888.")
        mock_cancel.assert_called_once_with("SH888")

    @patch('stt_services.transcribe_sarvam')
    @patch('database.cancel_shipment_order', return_value="ALREADY_PICKED_UP")
    def test_transcribe_tool_call_cancel_shipment_already_picked_up(self, mock_cancel, mock_transcribe_eng):
        mock_transcribe_eng.return_value = {
            "transcript": "Cancel SH123",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "cancel_shipment_order"
        mock_tool_call.function.arguments = json.dumps({"tracking_id": "SH123"})
        
        self.mock_gen_resp.return_value = {
            "ai_response": "",
            "tool_calls": [mock_tool_call]
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en", "session_id": "test_session_cancel"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["ai_response"], "I'm sorry, shipment SH123 cannot be cancelled because it has already been picked up.")
        mock_cancel.assert_called_once_with("SH123")

    @patch('stt_services.transcribe_sarvam')
    @patch('database.cancel_shipment_order', return_value="ALREADY_CANCELLED")
    def test_transcribe_tool_call_cancel_shipment_already_cancelled(self, mock_cancel, mock_transcribe_eng):
        mock_transcribe_eng.return_value = {
            "transcript": "Cancel SH777",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "cancel_shipment_order"
        mock_tool_call.function.arguments = json.dumps({"tracking_id": "SH777"})
        
        self.mock_gen_resp.return_value = {
            "ai_response": "",
            "tool_calls": [mock_tool_call]
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en", "session_id": "test_session_cancel"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["ai_response"], "Shipment SH777 is already cancelled.")
        mock_cancel.assert_called_once_with("SH777")

    @patch('stt_services.transcribe_sarvam')
    @patch('database.cancel_shipment_order', return_value="NOT_FOUND")
    def test_transcribe_tool_call_cancel_shipment_not_found(self, mock_cancel, mock_transcribe_eng):
        mock_transcribe_eng.return_value = {
            "transcript": "Cancel SH999",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "cancel_shipment_order"
        mock_tool_call.function.arguments = json.dumps({"tracking_id": "SH999"})
        
        self.mock_gen_resp.return_value = {
            "ai_response": "",
            "tool_calls": [mock_tool_call]
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en", "session_id": "test_session_cancel"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["ai_response"], "I couldn't find a shipment matching ID SH999 to cancel.")
        mock_cancel.assert_called_once_with("SH999")

    @patch('stt_services.transcribe_sarvam')
    def test_transcribe_includes_audio_b64(self, mock_transcribe_eng):
        mock_transcribe_eng.return_value = {
            "transcript": "Hello, is my package shipped?",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["audio_b64"], "dGVzdCBhdWRpbw==")
        self.mock_tts.assert_called_once_with(text="Mocked AI Response", target_language_code="English")

    @patch('database.get_session_language', return_value="Telugu")
    @patch('stt_services.transcribe_sarvam')
    @patch('stt_services.transcribe_auto')
    def test_transcribe_sticky_language_bypass(self, mock_transcribe_auto, mock_transcribe_indic, mock_get_lang):
        mock_transcribe_indic.return_value = {
            "transcript": "నమస్కారం",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "auto", "session_id": "sticky_session_123"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["transcript"], "నమస్కారం")
        
        # Bypassed auto-detect and hit Sarvam Telugu directly
        mock_transcribe_indic.assert_called_once()
        self.assertEqual(mock_transcribe_indic.call_args[1]["language_code"], "te-IN")
        mock_transcribe_auto.assert_not_called()

    @patch('database.get_shipment_details')
    @patch('stt_services.transcribe_sarvam')
    def test_transcribe_tracking_bypass_english(self, mock_transcribe_eng, mock_get_details):
        mock_get_details.return_value = {
            "tracking_id": "SH123",
            "status": "In Transit",
            "current_location": "Hyderabad"
        }
        mock_transcribe_eng.return_value = {
            "transcript": "Status of SH123",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["ai_response"], "Your shipment SH123 is currently In Transit at the Hyderabad hub.")
        self.mock_gen_resp.assert_not_called()

    @patch('database.get_shipment_details')
    @patch('stt_services.transcribe_sarvam')
    def test_transcribe_tracking_bypass_telugu(self, mock_transcribe_indic, mock_get_details):
        mock_get_details.return_value = {
            "tracking_id": "SH123",
            "status": "In Transit",
            "current_location": "Hyderabad"
        }
        mock_transcribe_indic.return_value = {
            "transcript": "ఎస్ హెచ్ వన్ టూ త్రీ స్థితి ఏమిటి",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "te-IN"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["ai_response"], "మీ రవాణా పొట్లం పొడవునా చూస్తే, ఐడి SH123 ప్రస్తుతం హైదరాబాద్ లో ఇన్ ట్రాన్సిట్ లో ఉంది.")
        self.mock_gen_resp.assert_not_called()

    @patch('database.get_shipment_details')
    @patch('stt_services.transcribe_sarvam')
    def test_transcribe_tracking_bypass_telugu_with_mixed_digits(self, mock_transcribe_indic, mock_get_details):
        mock_get_details.return_value = {
            "tracking_id": "SH507",
            "status": "In Transit",
            "current_location": "Hyderabad"
        }
        mock_transcribe_indic.return_value = {
            "transcript": "SH5జీరో 7 స్థితి ఏమిటి",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "te-IN"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["ai_response"], "మీ రవాణా పొట్లం పొడవునా చూస్తే, ఐడి SH507 ప్రస్తుతం హైదరాబాద్ లో ఇన్ ట్రాన్సిట్ లో ఉంది.")
        self.mock_gen_resp.assert_not_called()

    @patch('database.get_shipment_details')
    @patch('stt_services.transcribe_sarvam')
    def test_transcribe_tracking_bypass_hindi_with_precomposed_zha(self, mock_transcribe_indic, mock_get_details):
        mock_get_details.return_value = {
            "tracking_id": "SH507",
            "status": "In Transit",
            "current_location": "Hyderabad"
        }
        mock_transcribe_indic.return_value = {
            "transcript": "SH5ज़ीरो 7",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "hi-IN"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["ai_response"], "आपका शिपमेंट SH507 वर्तमान में हैदराबाद में इन ट्रांजिट में है।")
        self.mock_gen_resp.assert_not_called()

    @patch('database.get_shipment_details')
    @patch('stt_services.transcribe_sarvam')
    def test_transcribe_tracking_bypass_hindi_with_phonetic_one(self, mock_transcribe_indic, mock_get_details):
        mock_get_details.return_value = {
            "tracking_id": "SH212",
            "status": "In Transit",
            "current_location": "Hyderabad"
        }
        mock_transcribe_indic.return_value = {
            "transcript": "SH2वन 2",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "hi-IN"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["ai_response"], "आपका शिपमेंट SH212 वर्तमान में हैदराबाद में इन ट्रांजिट में है।")
        self.mock_gen_resp.assert_not_called()

    @patch('database.get_session_language', return_value="Telugu")
    @patch('database.get_shipment_details')
    @patch('stt_services.transcribe_sarvam')
    def test_transcribe_filler_word_preserves_telugu_lock(self, mock_transcribe_indic, mock_get_details, mock_get_lang):
        mock_get_details.return_value = {
            "tracking_id": "SH456",
            "status": "Out for Delivery",
            "current_location": "Ghatkesar"
        }
        mock_transcribe_indic.return_value = {
            "transcript": "Hello SH456",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "auto", "session_id": "sticky_telugu_session"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        
        # Assert response is in Telugu matching template
        self.assertEqual(json_data["ai_response"], "మీ రవాణా పొట్లం పొడవునా చూస్తే, ఐడి SH456 ప్రస్తుతం ఘట్కేసర్ లో డెలివరీ కోసం అవుట్ లో ఉంది.")
        self.mock_gen_resp.assert_not_called()
        self.mock_tts.assert_called_once_with(text=json_data["ai_response"], target_language_code="Telugu")

    @patch('database.get_session_language', return_value="Telugu")
    @patch('stt_services.transcribe_sarvam')
    def test_transcribe_thank_you_bypasses_llm_in_telugu(self, mock_transcribe_indic, mock_get_lang):
        mock_transcribe_indic.return_value = {
            "transcript": "థాంక్యూ",
            "telemetry": {"duration_seconds": 1.0}
        }
        
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "auto", "session_id": "sticky_telugu_session"}
        )
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        
        self.assertEqual(json_data["ai_response"], "మీకు స్వాగతం! మీకు సహాయం చేయడానికి సంతోషిస్తున్నాను. ఇంకా ఏదైనా సహాయం కావాలా?")
        self.mock_gen_resp.assert_not_called()
        self.mock_tts.assert_called_once_with(text=json_data["ai_response"], target_language_code="Telugu")

    @patch('stt_services.transcribe_sarvam')
    def test_workflow_state_machine_slot_filling(self, mock_transcribe_eng):
        # 1. Initiate workflow
        resp_init = self.client.post("/initiate_workflow", json={"session_id": "test_workflow_session_1"})
        self.assertEqual(resp_init.status_code, 200)
        data_init = resp_init.json()
        self.assertEqual(data_init["session_id"], "test_workflow_session_1")
        self.assertIn("Hello! Welcome to LogiRoute Express.", data_init["response_text"])
        
        # 2. Start pickup scheduling - missing pickup location
        mock_transcribe_eng.return_value = {
            "transcript": "I want to schedule a pickup",
            "telemetry": {"duration_seconds": 1.0}
        }
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en", "session_id": "test_workflow_session_1"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ai_response"], "Please provide the pickup location.")
        
        # 3. Provide pickup location - missing drop location
        mock_transcribe_eng.return_value = {
            "transcript": "Hyderabad",
            "telemetry": {"duration_seconds": 1.0}
        }
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en", "session_id": "test_workflow_session_1"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ai_response"], "Please provide the drop location.")

        # 4. Provide drop location - missing time
        mock_transcribe_eng.return_value = {
            "transcript": "Ghatkesar",
            "telemetry": {"duration_seconds": 1.0}
        }
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en", "session_id": "test_workflow_session_1"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ai_response"], "Please provide the pickup time.")
        
        # 5. Provide pickup time - missing weight
        mock_transcribe_eng.return_value = {
            "transcript": "tomorrow at 4 PM",
            "telemetry": {"duration_seconds": 1.0}
        }
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en", "session_id": "test_workflow_session_1"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ai_response"], "Please provide the package weight.")
        
        # 6. Provide weight - complete scheduling
        mock_transcribe_eng.return_value = {
            "transcript": "5 kg",
            "telemetry": {"duration_seconds": 1.0}
        }
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en", "session_id": "test_workflow_session_1"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("I have successfully scheduled your shipment! Your new tracking ID is SH", response.json()["ai_response"])

        # 6. Unexpected input deflection test
        self.client.post("/initiate_workflow", json={"session_id": "test_workflow_session_2"})
        mock_transcribe_eng.return_value = {
            "transcript": "I want to track a shipment",
            "telemetry": {"duration_seconds": 1.0}
        }
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en", "session_id": "test_workflow_session_2"}
        )
        self.assertEqual(response.json()["ai_response"], "Please provide your shipment ID.")
        
        mock_transcribe_eng.return_value = {
            "transcript": "tell me a joke",
            "telemetry": {"duration_seconds": 1.0}
        }
        response = self.client.post(
            "/transcribe",
            files={"file": ("speech.wav", self.loud_wav_bytes, "audio/wav")},
            data={"language_code": "en", "session_id": "test_workflow_session_2"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ai_response"], "I understand, but to help you with your request, I need you to provide the requested logistics information. Please provide the details.")
        
        stage, slots, lang = database.get_session_state("test_workflow_session_2")
        self.assertEqual(stage, "WAITING_FOR_TRACKING_ID")

    @patch('database.get_session_language', return_value="Telugu")
    @patch('tts_service.generate_speech_b64', return_value="dummy_audio_b64")
    def test_idle_warning_telugu(self, mock_tts, mock_get_lang):
        response = self.client.post(
            "/idle_warning",
            json={"session_id": "test_session_id"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["response_text"], "మీకు ఇకపై ఎటువంటి ప్రశ్నలు లేకపోతే, ఈ సెషన్ 30 సెకన్లలో ఆటోమేటిక్‌గా ముగిసిపోతుంది.")
        self.assertEqual(data["audio_b64"], "dummy_audio_b64")

if __name__ == '__main__':
    unittest.main()



