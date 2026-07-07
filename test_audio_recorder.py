import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import audio_recorder
from audio_recorder import (
    NOISE_FLOOR_MULTIPLIER, MIN_ENERGY_THRESHOLD,
    SPEECH_WINDOW_FRAMES, SPEECH_ACTIVATION_RATIO,
    POST_SPEECH_SILENCE_SEC, INITIAL_SILENCE_SEC,
)


class TestCalculateRMS(unittest.TestCase):

    def test_empty_array(self):
        self.assertEqual(audio_recorder.calculate_rms(np.array([])), 0.0)

    def test_zeros_array(self):
        self.assertEqual(audio_recorder.calculate_rms(np.zeros(16000)), 0.0)

    def test_constant_array(self):
        self.assertAlmostEqual(audio_recorder.calculate_rms(np.full(1000, 0.5)), 0.5)

    def test_sine_wave_rms(self):
        t    = np.linspace(0, 1, 16000, endpoint=False)
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        # RMS of a pure sine with amplitude 1 is 1/sqrt(2) ≈ 0.707
        self.assertAlmostEqual(audio_recorder.calculate_rms(sine), 1.0 / np.sqrt(2), places=2)


class TestIsSilent(unittest.TestCase):

    def test_silent_below_threshold(self):
        silent = np.random.normal(0, 0.001, 16000)
        self.assertTrue(audio_recorder.is_silent(silent, threshold=0.01))

    def test_loud_above_threshold(self):
        loud = np.random.normal(0, 0.05, 16000)
        self.assertFalse(audio_recorder.is_silent(loud, threshold=0.01))


class TestCalibrateNoiseFloor(unittest.TestCase):

    def test_dynamic_threshold_above_noise(self):
        """Threshold must be above MIN and proportional to noise level."""
        noise_frame = np.full((480, 1), 0.05, dtype='float32')  # loud noise -> clearly above MIN
        mock_stream = MagicMock()
        mock_stream.read.return_value = (noise_frame, False)

        _, threshold = audio_recorder._calibrate_noise_floor(mock_stream, 480, 16000)

        # At 0.05 amplitude, noise_rms ~= 0.05, threshold ~= 0.05 * 4.0 = 0.20
        self.assertGreater(threshold, MIN_ENERGY_THRESHOLD)
        self.assertGreater(threshold, 0.05)  # must be above raw noise level

    def test_silent_room_returns_minimum(self):
        """Completely silent room should return exactly MIN_ENERGY_THRESHOLD."""
        silence_frame = np.zeros((480, 1), dtype='float32')
        mock_stream   = MagicMock()
        mock_stream.read.return_value = (silence_frame, False)

        _, threshold = audio_recorder._calibrate_noise_floor(mock_stream, 480, 16000)
        self.assertEqual(threshold, MIN_ENERGY_THRESHOLD)

    def test_returns_calibration_frames(self):
        """Calibration frames are returned for inclusion in the recording."""
        frame = np.full((480, 1), 0.05, dtype='float32')  # loud enough to not hit exception path
        mock_stream = MagicMock()
        mock_stream.read.return_value = (frame, False)

        frames, threshold = audio_recorder._calibrate_noise_floor(mock_stream, 480, 16000)
        self.assertIsInstance(frames, list)
        # At least 2 calibration frames (n_calib = max(2, ...))
        self.assertGreaterEqual(len(frames), 2)

    def test_calibration_exception_returns_minimum(self):
        """If stream.read raises, calibration falls back to MIN_ENERGY_THRESHOLD."""
        mock_stream = MagicMock()
        mock_stream.read.side_effect = RuntimeError("stream broke")

        frames, threshold = audio_recorder._calibrate_noise_floor(mock_stream, 480, 16000)
        self.assertEqual(threshold, MIN_ENERGY_THRESHOLD)
        self.assertEqual(frames, [])


class TestRecordAudio(unittest.TestCase):
    """Tests for record_audio; _calibrate_noise_floor is always mocked."""

    # ── Speech detected, stops on post-speech silence ─────────────────────────

    @patch('sounddevice.InputStream')
    @patch('webrtcvad.Vad')
    @patch('noisereduce.reduce_noise')
    @patch('audio_recorder._calibrate_noise_floor')
    def test_speech_then_silence_stops_recording(
        self, mock_calib, mock_reduce, mock_vad_cls, mock_stream_cls
    ):
        """
        Hybrid VAD: 4 high-energy speech frames then sustained silence
        should print 'Speech detected!' and later stop on post-speech silence.
        """
        mock_calib.return_value = ([], 0.003)   # threshold=0.003, no calib frames

        mock_vad_instance = MagicMock()
        mock_vad_cls.return_value = mock_vad_instance
        # VAD mirrors energy: True for speech frames, False for silence
        mock_vad_instance.is_speech.side_effect = [True]*4 + [False]*200

        speech_frame  = np.full((480, 1), 0.05,   dtype='float32')  # RMS >> 0.003
        silence_frame = np.full((480, 1), 0.0001, dtype='float32')  # RMS << 0.003
        frames = [speech_frame]*4 + [silence_frame]*200

        mock_stream = MagicMock()
        mock_stream.read.side_effect = [(f, False) for f in frames]
        mock_stream_cls.return_value.__enter__.return_value = mock_stream

        mock_reduce.return_value = np.zeros(2000)

        audio_recorder.record_audio(duration=10.0)
        mock_reduce.assert_called_once()

    # ── Energy gate rejects low-energy frames even when VAD says speech ───────

    @patch('sounddevice.InputStream')
    @patch('webrtcvad.Vad')
    @patch('noisereduce.reduce_noise')
    @patch('audio_recorder._calibrate_noise_floor')
    def test_energy_gate_overrides_vad(
        self, mock_calib, mock_reduce, mock_vad_cls, mock_stream_cls
    ):
        """
        Even when WebRTC VAD returns True every frame, frames with energy below
        the threshold must not count as speech.  Recording must time out on the
        initial-silence limit (~167 frames @ 30ms = 5.0 s).
        """
        mock_calib.return_value = ([], 0.003)

        mock_vad_instance = MagicMock()
        mock_vad_cls.return_value = mock_vad_instance
        mock_vad_instance.is_speech.return_value = True   # VAD always says speech

        # But energy is well below threshold (RMS ≈ 0.0001)
        low_energy_frame = np.full((480, 1), 0.0001, dtype='float32')
        mock_stream = MagicMock()
        mock_stream.read.return_value = (low_energy_frame, False)
        mock_stream_cls.return_value.__enter__.return_value = mock_stream

        mock_reduce.return_value = np.zeros(8000)

        audio_recorder.record_audio(duration=10.0)

        # Because energy remains below threshold, the wait loop times out and WebRTC VAD is never called
        call_count = mock_vad_instance.is_speech.call_count
        self.assertEqual(call_count, 0)


    # ── Sliding-window bridges short inter-word pauses ────────────────────────

    @patch('sounddevice.InputStream')
    @patch('webrtcvad.Vad')
    @patch('noisereduce.reduce_noise')
    @patch('audio_recorder._calibrate_noise_floor')
    def test_sliding_window_bridges_short_pauses(
        self, mock_calib, mock_reduce, mock_vad_cls, mock_stream_cls
    ):
        """
        A single silence frame between two speech bursts must NOT advance the
        silence counter (the sliding window still has > 40 % speech).
        Recording should not stop prematurely.
        """
        mock_calib.return_value = ([], 0.003)

        mock_vad_instance = MagicMock()
        mock_vad_cls.return_value = mock_vad_instance
        # speech, speech, ONE silence, speech, speech, then sustained silence
        mock_vad_instance.is_speech.side_effect = [True, True, False, True, True] + [False]*200

        speech_frame  = np.full((480, 1), 0.05,   dtype='float32')
        silence_frame = np.full((480, 1), 0.0001, dtype='float32')
        frames = [speech_frame]*2 + [silence_frame] + [speech_frame]*2 + [silence_frame]*200

        mock_stream = MagicMock()
        mock_stream.read.side_effect = [(f, False) for f in frames]
        mock_stream_cls.return_value.__enter__.return_value = mock_stream

        mock_reduce.return_value = np.zeros(1000)

        audio_recorder.record_audio(duration=10.0)
        mock_reduce.assert_called_once()

    # ── Initial silence timeout ───────────────────────────────────────────────

    @patch('sounddevice.InputStream')
    @patch('webrtcvad.Vad')
    @patch('noisereduce.reduce_noise')
    @patch('audio_recorder._calibrate_noise_floor')
    def test_initial_silence_timeout(
        self, mock_calib, mock_reduce, mock_vad_cls, mock_stream_cls
    ):
        """No speech ever → stops after INITIAL_SILENCE_SEC (5.0 s ≈ 167 frames)."""
        mock_calib.return_value = ([], 0.003)

        mock_vad_instance = MagicMock()
        mock_vad_cls.return_value = mock_vad_instance
        mock_vad_instance.is_speech.return_value = False

        silence_frame = np.zeros((480, 1), dtype='float32')
        mock_stream   = MagicMock()
        mock_stream.read.return_value = (silence_frame, False)
        mock_stream_cls.return_value.__enter__.return_value = mock_stream

        mock_reduce.return_value = np.zeros(8000)

        audio_recorder.record_audio(duration=10.0)

        # Because energy remains below threshold, the wait loop times out and WebRTC VAD is never called
        call_count = mock_vad_instance.is_speech.call_count
        self.assertEqual(call_count, 0)

    # ── VAD aggressiveness ────────────────────────────────────────────────────

    @patch('sounddevice.InputStream')
    @patch('webrtcvad.Vad')
    @patch('noisereduce.reduce_noise')
    @patch('audio_recorder._calibrate_noise_floor')
    def test_vad_initialised_with_mode_2(
        self, mock_calib, mock_reduce, mock_vad_cls, mock_stream_cls
    ):
        """record_audio must initialise WebRTC VAD with aggressiveness mode 2."""
        mock_calib.return_value = ([], 0.003)
        mock_vad_instance = MagicMock()
        mock_vad_cls.return_value = mock_vad_instance
        mock_vad_instance.is_speech.return_value = False

        silence_frame = np.zeros((480, 1), dtype='float32')
        mock_stream   = MagicMock()
        mock_stream.read.return_value = (silence_frame, False)
        mock_stream_cls.return_value.__enter__.return_value = mock_stream
        mock_reduce.return_value = np.zeros(100)

        audio_recorder.record_audio(duration=5.1)
        mock_vad_cls.assert_called_once_with(2)

    # ── save_audio ─────────────────────────────────────────────────────────────

    @patch('soundfile.write')
    def test_save_audio_calls_sf_write(self, mock_write):
        data = np.zeros((16000, 1), dtype='float32')
        audio_recorder.save_audio(data, "output.wav", samplerate=16000)
        mock_write.assert_called_once_with("output.wav", data, 16000)


if __name__ == '__main__':
    unittest.main()
