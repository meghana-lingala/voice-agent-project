import unittest
import os
import shutil
from unittest.mock import patch, MagicMock
import requests
import app

class TestAppClient(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_samples_dir"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_get_next_filename_empty(self):
        """Dynamic filename indexing starts at sample_001.wav in empty directories."""
        next_file = app.get_next_filename(self.test_dir)
        self.assertEqual(os.path.basename(next_file), "sample_001.wav")

    def test_get_next_filename_sequential_increment(self):
        """Dynamic filename indexing increments sequentially based on maximum index found."""
        os.makedirs(self.test_dir, exist_ok=True)
        # Seed directory with existing files
        with open(os.path.join(self.test_dir, "sample_001.wav"), "wb") as f:
            f.write(b"")
        with open(os.path.join(self.test_dir, "sample_002.wav"), "wb") as f:
            f.write(b"")
            
        next_file = app.get_next_filename(self.test_dir)
        self.assertEqual(os.path.basename(next_file), "sample_003.wav")

    def test_get_next_filename_gap_handling(self):
        """Dynamic filename indexing finds maximum index even with gap/non-consecutive indices."""
        os.makedirs(self.test_dir, exist_ok=True)
        with open(os.path.join(self.test_dir, "sample_010.wav"), "wb") as f:
            f.write(b"")
            
        next_file = app.get_next_filename(self.test_dir)
        self.assertEqual(os.path.basename(next_file), "sample_011.wav")

    @patch('app.time.sleep', side_effect=KeyboardInterrupt)
    @patch('app.audio_recorder.record_audio')
    @patch('app.audio_recorder.calculate_rms')
    @patch('app.audio_recorder.save_audio')
    @patch('requests.post')
    def test_main_loop_success(self, mock_post, mock_save, mock_rms, mock_record, mock_sleep):
        """Interactive client loop successfully records, saves, and transcribes."""
        import numpy as np
        from unittest.mock import mock_open
        
        mock_record.return_value = np.zeros((16000, 1))
        mock_rms.return_value = 0.05
        
        # Mock server response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "transcript": "Test speech transcription.",
            "telemetry": {
                "duration_seconds": 1.45,
                "engine_used": "Groq"
            }
        }
        mock_post.return_value = mock_response
        
        with patch('builtins.open', mock_open(read_data=b"dummy wav data")):
            with patch('app.get_next_filename') as mock_get_filename:
                mock_get_filename.return_value = "samples/01_accuracy_benchmarks/sample_001.wav"
                app.main()
                
        mock_record.assert_called_once()
        mock_save.assert_called_once()
        mock_post.assert_called_once()

    @patch('app.time.sleep', side_effect=KeyboardInterrupt)
    @patch('app.audio_recorder.record_audio')
    @patch('app.audio_recorder.calculate_rms')
    @patch('requests.post')
    def test_main_loop_connection_error(self, mock_post, mock_rms, mock_record, mock_sleep):
        """Interactive client loop elegantly catches backend offline connection errors."""
        import numpy as np
        from unittest.mock import mock_open
        
        mock_record.return_value = np.zeros((16000, 1))
        mock_rms.return_value = 0.05
        
        # Raise ConnectionError on post call
        mock_post.side_effect = requests.exceptions.ConnectionError("Offline")
        
        with patch('builtins.open', mock_open(read_data=b"dummy wav data")):
            with patch('app.get_next_filename') as mock_get_filename:
                mock_get_filename.return_value = "samples/01_accuracy_benchmarks/sample_001.wav"
                with patch('app.audio_recorder.save_audio'):
                    app.main()
                    
        mock_post.assert_called_once()

if __name__ == "__main__":
    unittest.main()
