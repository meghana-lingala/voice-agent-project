import unittest
import os
import io
import json
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import main
from main import app

class TestMainAPI(unittest.TestCase):

    def setUp(self):
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
        self.log_file_patcher.stop()
        self.gen_resp_patcher.stop()
        self.log_int_patcher.stop()
        self.get_ship_patcher.stop()
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)


    @patch('stt_services.transcribe_english')
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
        self.assertEqual(json_data["telemetry"]["engine_used"], "Groq")
        self.assertEqual(json_data["telemetry"]["duration_seconds"], 1.25)

        # Assert logging integration persists transaction correctly
        self.assertTrue(os.path.exists(self.test_log_file))
        with open(self.test_log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0]["audio_filename"], "speech.wav")
            self.assertEqual(logs[0]["transcribed_text"], "Hello, this is a test recording.")
            self.assertEqual(logs[0]["engine_used"], "Groq")
            self.assertEqual(logs[0]["language_code"], "en")

    @patch('stt_services.transcribe_indic')
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
            self.assertEqual(logs[0]["language_code"], "hi-IN")

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
                "engine_used": "Groq",
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
        self.assertEqual(json_data["telemetry"]["engine_used"], "Groq")

        # Check logs
        self.assertTrue(os.path.exists(self.test_log_file))
        with open(self.test_log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
            self.assertEqual(logs[0]["audio_filename"], "auto_speech.wav")
            self.assertEqual(logs[0]["engine_used"], "Groq")
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

    @patch('stt_services.transcribe_english')
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

    @patch('stt_services.transcribe_english')
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

if __name__ == '__main__':
    unittest.main()

