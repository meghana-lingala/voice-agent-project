import unittest
from unittest.mock import patch, MagicMock, mock_open
import stt_services
from stt_services import (
    _detect_indic_script_ratio,
    INDIC_SCRIPT_RATIO_THRESHOLD,
)


class TestDetectIndicScriptRatio(unittest.TestCase):

    def test_pure_latin_returns_zero(self):
        self.assertAlmostEqual(_detect_indic_script_ratio("Hello world"), 0.0)

    def test_pure_telugu_high_ratio(self):
        self.assertGreater(_detect_indic_script_ratio("నమస్కారం ఎలా ఉన్నారు"), 0.7)

    def test_pure_hindi_high_ratio(self):
        self.assertGreater(_detect_indic_script_ratio("नमस्ते दुनिया"), 0.7)

    def test_mixed_telugu_english_moderate(self):
        ratio = _detect_indic_script_ratio("నమస్కారం how are you")
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 1.0)

    def test_empty_string_returns_zero(self):
        self.assertEqual(_detect_indic_script_ratio(""), 0.0)

    def test_whitespace_only_returns_zero(self):
        self.assertAlmostEqual(_detect_indic_script_ratio("   "), 0.0)


class TestTranscribeEnglish(unittest.TestCase):

    def setUp(self):
        patch('stt_services.GROQ_API_KEY', 'dummy').start()
        self.addCleanup(patch.stopall)

    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services.groq_client')
    def test_success(self, mock_groq, _exists):
        mock_resp = MagicMock(); mock_resp.text = "Hello world"
        mock_groq.audio.transcriptions.create.return_value = mock_resp
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_english("dummy.wav")
        self.assertEqual(result["transcript"], "Hello world")
        self.assertGreaterEqual(result["telemetry"]["duration_seconds"], 0.0)
        kw = mock_groq.audio.transcriptions.create.call_args[1]
        self.assertEqual(kw["model"], "whisper-large-v3-turbo")
        self.assertEqual(kw["file"][0], "dummy.wav")

    @patch('stt_services.os.path.exists', return_value=False)
    def test_file_not_found(self, _):
        with self.assertRaises(FileNotFoundError):
            stt_services.transcribe_english("no.wav")

    def test_missing_key_raises(self):
        with patch('stt_services.GROQ_API_KEY', None):
            with self.assertRaises(ValueError):
                stt_services.transcribe_english("dummy.wav")


class TestTranscribeIndic(unittest.TestCase):

    def setUp(self):
        patch('stt_services.SARVAM_API_KEY', 'dummy').start()
        self.addCleanup(patch.stopall)

    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services.sarvam_client')
    def test_hindi_success(self, mock_sarvam, _):
        mock_resp = MagicMock(); mock_resp.transcript = "नमस्ते दुनिया"
        mock_sarvam.speech_to_text.transcribe.return_value = mock_resp
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_indic("dummy.wav", language_code="hi-IN")
        self.assertEqual(result["transcript"], "नमस्ते दुनिया")
        kw = mock_sarvam.speech_to_text.transcribe.call_args[1]
        self.assertEqual(kw["language_code"], "hi-IN")
        self.assertEqual(kw["model"], "saaras:v3")

    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services.sarvam_client')
    def test_telugu_success(self, mock_sarvam, _):
        mock_resp = MagicMock(); mock_resp.transcript = "నమస్కారం ఎలా ఉన్నారు"
        mock_sarvam.speech_to_text.transcribe.return_value = mock_resp
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_indic("dummy.wav", language_code="te-IN")
        self.assertEqual(result["transcript"], "నమస్కారం ఎలా ఉన్నారు")
        kw = mock_sarvam.speech_to_text.transcribe.call_args[1]
        self.assertEqual(kw["language_code"], "te-IN")

    @patch('stt_services.os.path.exists', return_value=False)
    def test_file_not_found(self, _):
        with self.assertRaises(FileNotFoundError):
            stt_services.transcribe_indic("no.wav")

    def test_missing_key_raises(self):
        with patch('stt_services.SARVAM_API_KEY', None):
            with self.assertRaises(ValueError):
                stt_services.transcribe_indic("dummy.wav")


class TestTranscribeAuto(unittest.TestCase):

    def setUp(self):
        patch('stt_services.GROQ_API_KEY',   'dummy_groq').start()
        patch('stt_services.SARVAM_API_KEY', 'dummy_sarvam').start()
        self.addCleanup(patch.stopall)

    def _make_indic_side_effect(self, te_transcript, hi_transcript):
        def side_effect(file_path, language_code):
            transcript = te_transcript if language_code == "te-IN" else hi_transcript
            return transcript, 0.4, _detect_indic_script_ratio(transcript)
        return side_effect

    # ── Short audio – Telugu detected ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._get_audio_duration', return_value=10.0)
    @patch('stt_services._groq_auto')
    @patch('stt_services._sarvam_probe')
    def test_short_telugu_success(self, mock_probe, mock_groq_auto, _dur, _exists):
        mock_probe.side_effect = self._make_indic_side_effect(
            te_transcript="నమస్కారం ఎలా ఉన్నారు",  # contains Telugu words
            hi_transcript="Namaskaaram ela unnaru",
        )
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "Telugu")
        self.assertEqual(result["telemetry"]["engine_used"], "Sarvam")
        mock_groq_auto.assert_not_called()
        self.assertEqual(mock_probe.call_count, 2)

    # ── Short audio – Hindi detected ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._get_audio_duration', return_value=10.0)
    @patch('stt_services._groq_auto')
    @patch('stt_services._sarvam_probe')
    def test_short_hindi_success(self, mock_probe, mock_groq_auto, _dur, _exists):
        mock_probe.side_effect = self._make_indic_side_effect(
            te_transcript="Namaskaaram kaise ho",
            hi_transcript="नमस्ते कैसे हो आप",   # contains Hindi words
        )
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "Hindi")
        self.assertEqual(result["telemetry"]["engine_used"], "Sarvam")
        mock_groq_auto.assert_not_called()
        self.assertEqual(mock_probe.call_count, 2)

    # ── Short audio – English ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._get_audio_duration', return_value=10.0)
    @patch('stt_services._groq_auto')
    @patch('stt_services._sarvam_probe')
    def test_short_english_success(self, mock_probe, mock_groq_auto, _dur, _exists):
        mock_probe.side_effect = self._make_indic_side_effect(
            te_transcript="Hello",
            hi_transcript="Hello",
        )
        mock_groq_auto.return_value = ("Hello world", 0.5)
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "English")
        self.assertEqual(result["telemetry"]["engine_used"], "Groq")
        self.assertEqual(result["transcript"], "Hello world")
        self.assertEqual(mock_probe.call_count, 2)
        mock_groq_auto.assert_called_once()

    # ── Short audio – Telugu+English code-switching ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._get_audio_duration', return_value=8.0)
    @patch('stt_services._groq_auto')
    @patch('stt_services._sarvam_probe')
    def test_short_telugu_english_code_switch(self, mock_probe, mock_groq_auto, _dur, _exists):
        mock_probe.side_effect = self._make_indic_side_effect(
            te_transcript="నమస్కారం నాకు coffee కావాలి",
            hi_transcript="Hello coffee needed",
        )
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "Telugu")
        self.assertIn("నమస్కారం", result["transcript"])
        self.assertIn("coffee", result["transcript"])

    # ── Short audio – Hindi+English code-switching ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._get_audio_duration', return_value=8.0)
    @patch('stt_services._groq_auto')
    @patch('stt_services._sarvam_probe')
    def test_short_hindi_english_code_switch(self, mock_probe, mock_groq_auto, _dur, _exists):
        mock_probe.side_effect = self._make_indic_side_effect(
            te_transcript="Hello mera naam",
            hi_transcript="नमस्ते मेरा name है राम",
        )
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "Hindi")
        self.assertIn("नमस्ते", result["transcript"])

    # ── Short audio – Sarvam exception fallback to Groq auto Indic ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._get_audio_duration', return_value=10.0)
    @patch('stt_services._groq_auto')
    @patch('stt_services._sarvam_probe')
    def test_short_sarvam_both_fail_falls_to_groq(self, mock_probe, mock_groq_auto, _dur, _exists):
        mock_probe.side_effect = Exception("Sarvam network error")
        mock_groq_auto.return_value = ("నమస్కారం", 0.5)
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "Telugu")
        self.assertEqual(result["telemetry"]["engine_used"], "Groq")
        self.assertEqual(result["transcript"], "నమస్కారం")

    # ── Long audio – Telugu ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._get_audio_duration', return_value=60.0)
    @patch('stt_services._groq_auto')
    @patch('stt_services._groq_forced')
    def test_long_telugu_via_groq_forced(self, mock_forced, mock_groq_auto, _dur, _exists):
        te = ("నమస్కారం ఎలా ఉన్నారు మీరు", 0.5)
        hi = ("Namaskaaram", 0.5)
        mock_forced.side_effect = [te, hi]
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "Telugu")
        self.assertEqual(result["telemetry"]["engine_used"], "Groq Fallback")
        self.assertEqual(mock_forced.call_count, 2)
        mock_groq_auto.assert_not_called()

    # ── Long audio – Hindi ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._get_audio_duration', return_value=60.0)
    @patch('stt_services._groq_auto')
    @patch('stt_services._groq_forced')
    def test_long_hindi_via_groq_forced(self, mock_forced, mock_groq_auto, _dur, _exists):
        te = ("Namaskar kaise ho", 0.5)
        hi = ("नमस्ते कैसे हो आप", 0.5)
        mock_forced.side_effect = [te, hi]
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "Hindi")
        self.assertEqual(result["telemetry"]["engine_used"], "Groq Fallback")
        self.assertEqual(mock_forced.call_count, 2)
        mock_groq_auto.assert_not_called()

    # ── Long audio – English ──
    @patch('stt_services.os.path.exists', return_value=True)
    @patch('stt_services._get_audio_duration', return_value=60.0)
    @patch('stt_services._groq_auto')
    @patch('stt_services._groq_forced')
    def test_long_english_via_groq(self, mock_forced, mock_groq_auto, _dur, _exists):
        te = ("Hello", 0.5)
        hi = ("Hello", 0.5)
        mock_forced.side_effect = [te, hi]
        mock_groq_auto.return_value = ("Hi I would like to place an order", 0.5)
        with patch('builtins.open', mock_open(read_data=b"audio")):
            result = stt_services.transcribe_auto("dummy.wav")
        self.assertEqual(result["telemetry"]["detected_language"], "English")
        self.assertEqual(result["telemetry"]["engine_used"], "Groq")
        self.assertEqual(mock_forced.call_count, 2)
        mock_groq_auto.assert_called_once()

    # ── Guard clauses ──
    @patch('stt_services.os.path.exists', return_value=False)
    def test_file_not_found(self, _):
        with self.assertRaises(FileNotFoundError):
            stt_services.transcribe_auto("missing.wav")

    def test_missing_groq_key_raises(self):
        with patch('stt_services.GROQ_API_KEY', None), \
             patch('stt_services.os.path.exists', return_value=True):
            with self.assertRaises(ValueError):
                stt_services.transcribe_auto("dummy.wav")

    def test_missing_sarvam_key_raises(self):
        with patch('stt_services.SARVAM_API_KEY', None), \
             patch('stt_services.os.path.exists', return_value=True):
            with self.assertRaises(ValueError):
                stt_services.transcribe_auto("dummy.wav")


if __name__ == '__main__':
    unittest.main()
