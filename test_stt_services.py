import unittest
from unittest.mock import patch, MagicMock, mock_open
import stt_services


class TestTranscribeSarvam(unittest.TestCase):
    """Tests for the core transcribe_sarvam() function."""

    def setUp(self):
        patch('stt_services.SARVAM_API_KEY', 'dummy').start()
        self.addCleanup(patch.stopall)

    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services.sarvam_client')
    def test_english_success(self, mock_sarvam, _):
        mock_resp = MagicMock(); mock_resp.transcript = "Hello world"
        mock_sarvam.speech_to_text.transcribe.return_value = mock_resp
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_sarvam("dummy.wav", language_code="en-IN")
        self.assertEqual(result["transcript"], "Hello world")
        self.assertGreaterEqual(result["telemetry"]["duration_seconds"], 0.0)
        kw = mock_sarvam.speech_to_text.transcribe.call_args[1]
        self.assertEqual(kw["language_code"], "en-IN")
        self.assertEqual(kw["model"], "saaras:v3")

    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services.sarvam_client')
    def test_hindi_success(self, mock_sarvam, _):
        mock_resp = MagicMock(); mock_resp.transcript = "नमस्ते दुनिया"
        mock_sarvam.speech_to_text.transcribe.return_value = mock_resp
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_sarvam("dummy.wav", language_code="hi-IN")
        self.assertEqual(result["transcript"], "नमस्ते दुनिया")
        kw = mock_sarvam.speech_to_text.transcribe.call_args[1]
        self.assertEqual(kw["language_code"], "hi-IN")

    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services.sarvam_client')
    def test_telugu_success(self, mock_sarvam, _):
        mock_resp = MagicMock(); mock_resp.transcript = "నమస్కారం ఎలా ఉన్నారు"
        mock_sarvam.speech_to_text.transcribe.return_value = mock_resp
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_sarvam("dummy.wav", language_code="te-IN")
        self.assertEqual(result["transcript"], "నమస్కారం ఎలా ఉన్నారు")
        kw = mock_sarvam.speech_to_text.transcribe.call_args[1]
        self.assertEqual(kw["language_code"], "te-IN")

    @patch('stt_services.os.path.exists', return_value=False)
    def test_file_not_found(self, _):
        with self.assertRaises(FileNotFoundError):
            stt_services.transcribe_sarvam("no.wav")

    def test_missing_key_raises(self):
        with patch('stt_services.SARVAM_API_KEY', None):
            with self.assertRaises(ValueError):
                stt_services.transcribe_sarvam("dummy.wav")


class TestTranscribeAuto(unittest.TestCase):
    """Tests for the 3-way Sarvam parallel probe auto-detection."""

    def setUp(self):
        patch('stt_services.SARVAM_API_KEY', 'dummy_sarvam').start()
        self.addCleanup(patch.stopall)

    def _make_probe_side_effect(self, en_text, hi_text, te_text):
        """Helper: returns a side_effect for _sarvam_probe that dispatches
        by language_code argument."""
        def side_effect(file_path, language_code):
            if language_code == "en-IN":
                return (en_text, 0.4)
            elif language_code == "hi-IN":
                return (hi_text, 0.4)
            else:
                return (te_text, 0.4)
        return side_effect

    # ── Telugu detected ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._sarvam_probe')
    def test_telugu_detected(self, mock_probe, _exists):
        mock_probe.side_effect = self._make_probe_side_effect(
            en_text="Namaskaram how are you",
            hi_text="नमस्कारम ऐला उन्नारु",
            te_text="నమస్కారం ఎలా ఉన్నారు",  # genuine Telugu: ఎలా, ఉన్నారు
        )
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "Telugu")
        self.assertEqual(result["telemetry"]["engine_used"], "Sarvam")
        self.assertIn("ఉన్నారు", result["transcript"])

    # ── Hindi detected ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._sarvam_probe')
    def test_hindi_detected(self, mock_probe, _exists):
        mock_probe.side_effect = self._make_probe_side_effect(
            en_text="Namaste how are you",
            hi_text="नमस्ते कैसे हो आप",  # genuine Hindi: नमस्ते, कैसे, हो, आप
            te_text="నమస్తే కైసే హో ఆప్",
        )
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "Hindi")
        self.assertEqual(result["telemetry"]["engine_used"], "Sarvam")
        self.assertIn("नमस्ते", result["transcript"])

    # ── English detected (no genuine Indic morphemes) ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._sarvam_probe')
    def test_english_detected(self, mock_probe, _exists):
        mock_probe.side_effect = self._make_probe_side_effect(
            en_text="Hello world I want to place an order",
            hi_text="हैलो वर्ल्ड आई वांट टू प्लेस एन ऑर्डर",  # phonetic transliteration, no genuine Hindi
            te_text="హలో వరల్డ్ ఐ వాంట్ టూ ప్లేస్ ఏన్ ఆర్డర్",  # phonetic, no genuine Telugu
        )
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "English")
        self.assertEqual(result["telemetry"]["engine_used"], "Sarvam")
        self.assertEqual(result["transcript"], "Hello world I want to place an order")

    # ── Telugu+English code-switching ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._sarvam_probe')
    def test_telugu_english_code_switch(self, mock_probe, _exists):
        mock_probe.side_effect = self._make_probe_side_effect(
            en_text="Namaskaram naku coffee kavali",
            hi_text="नमस्कारं नाकु कॉफी कावालि",
            te_text="నమస్కారం నాకు coffee కావాలి",  # genuine Telugu: నాకు, కావాలి
        )
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "Telugu")
        self.assertIn("కావాలి", result["transcript"])

    # ── Hindi+English code-switching ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._sarvam_probe')
    def test_hindi_english_code_switch(self, mock_probe, _exists):
        mock_probe.side_effect = self._make_probe_side_effect(
            en_text="Namaste mera name hai Ram",
            hi_text="नमस्ते मेरा name है राम",  # genuine Hindi: नमस्ते, मेरा, है
            te_text="నమస్తే మేరా నేమ్ హై రామ్",
        )
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "Hindi")
        self.assertIn("नमस्ते", result["transcript"])

    # ── All probes fail → empty string fallback ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._sarvam_probe')
    def test_all_probes_fail_returns_empty(self, mock_probe, _exists):
        mock_probe.side_effect = Exception("Sarvam network error")
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "English")
        self.assertEqual(result["transcript"], "")

    # ── Only en-IN probe succeeds ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._sarvam_probe')
    def test_only_english_probe_succeeds(self, mock_probe, _exists):
        call_count = [0]
        def side_effect(file_path, language_code):
            call_count[0] += 1
            if language_code == "en-IN":
                return ("Where is my order", 0.4)
            raise Exception("Sarvam network error")
        mock_probe.side_effect = side_effect
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "English")
        self.assertEqual(result["transcript"], "Where is my order")

    # ── SH tracking ID shouldn't affect language ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._sarvam_probe')
    def test_tracking_id_no_language_confusion(self, mock_probe, _exists):
        mock_probe.side_effect = self._make_probe_side_effect(
            en_text="The shipment ID is SH123",
            hi_text="शिपमेंट आईडी एसएच123 है",  # no genuine Hindi
            te_text="షిప్మెంట్ ఐడీ ఎస్ హెచ్ 123",  # no genuine Telugu
        )
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "English")

    # ── Guard clauses ──
    @patch('stt_services.os.path.exists', return_value=False)
    def test_file_not_found(self, _):
        with self.assertRaises(FileNotFoundError):
            stt_services.transcribe_auto("missing.wav")

    def test_missing_sarvam_key_raises(self):
        with patch('stt_services.SARVAM_API_KEY', None), \
             patch('stt_services.os.path.exists', return_value=True):
            with self.assertRaises(ValueError):
                stt_services.transcribe_auto("dummy.wav")


class TestGenuineWordMatching(unittest.TestCase):
    """Tests for _has_genuine_telugu and _has_genuine_hindi helpers."""

    def test_genuine_telugu_with_long_word(self):
        self.assertTrue(stt_services._has_genuine_telugu("నేను ఉన్నాను ఇక్కడ"))

    def test_genuine_telugu_two_short_words(self):
        self.assertTrue(stt_services._has_genuine_telugu("ఎలా ఏమి"))

    def test_no_telugu_in_phonetic_english(self):
        # Sarvam transliteration of English into Telugu script
        self.assertFalse(stt_services._has_genuine_telugu("హలో వరల్డ్ ఐ వాంట్"))

    def test_genuine_hindi_with_long_word(self):
        self.assertTrue(stt_services._has_genuine_hindi("नमस्ते कैसे हो"))

    def test_genuine_hindi_two_short_words(self):
        self.assertTrue(stt_services._has_genuine_hindi("है हो"))

    def test_no_hindi_in_phonetic_english(self):
        # Sarvam transliteration of English into Devanagari
        self.assertFalse(stt_services._has_genuine_hindi("हैलो वर्ल्ड आई वांट"))

    def test_empty_string(self):
        self.assertFalse(stt_services._has_genuine_telugu(""))
        self.assertFalse(stt_services._has_genuine_hindi(""))

    def test_none_input(self):
        self.assertFalse(stt_services._has_genuine_telugu(None))
        self.assertFalse(stt_services._has_genuine_hindi(None))


if __name__ == '__main__':
    unittest.main()
