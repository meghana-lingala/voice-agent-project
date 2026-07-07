import os
import argparse
import sys
import collections
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
import noisereduce as nr
from dotenv import load_dotenv

load_dotenv()

# ── VAD / Silence-Detection constants ─────────────────────────────────────────
NOISE_CALIBRATION_SECS  = 0.4
NOISE_FLOOR_MULTIPLIER  = 4.0
MIN_ENERGY_THRESHOLD    = 0.0008
SPEECH_WINDOW_FRAMES    = 10
SPEECH_ACTIVATION_RATIO = 0.40
POST_SPEECH_SILENCE_SEC = 2.0
INITIAL_SILENCE_SEC     = 5.0

def calculate_rms(audio_data: np.ndarray) -> float:
    """Return the Root Mean Square amplitude of a 1-D audio array."""
    if len(audio_data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_data))))

def _calibrate_noise_floor(stream, blocksize: int, samplerate: int):
    """
    Read NOISE_CALIBRATION_SECS of audio from the already-open *stream* to
    measure the ambient noise RMS and derive a dynamic per-frame energy
    threshold that sits above the background noise floor.
    """
    try:
        # Warmup: discard the first 0.3 seconds of stream input to let hardware/driver stabilize
        try:
            n_warmup = max(1, int(0.3 * samplerate / blocksize))
            for _ in range(n_warmup):
                stream.read(blocksize)
        except Exception as warmup_exc:
            print(f"[VAD] Warmup warning: {warmup_exc}", file=sys.stderr)

        n_calib = max(2, int(NOISE_CALIBRATION_SECS * samplerate / blocksize))
        frames  = []
        for _ in range(n_calib):
            data, _ = stream.read(blocksize)
            frames.append(data.copy())

        flat      = np.concatenate([f.flatten() for f in frames])
        noise_rms = calculate_rms(flat)
        threshold = max(noise_rms * NOISE_FLOOR_MULTIPLIER, MIN_ENERGY_THRESHOLD)
        print(
            f"[VAD] Noise floor RMS: {noise_rms:.5f}  "
            f"->  Energy threshold set to: {threshold:.5f}"
        )
        return frames, threshold

    except Exception as exc:
        print(
            f"[VAD] Calibration failed ({exc}). "
            f"Using default threshold: {MIN_ENERGY_THRESHOLD:.5f}",
            file=sys.stderr
        )
        return [], MIN_ENERGY_THRESHOLD

def record_audio(
    duration: float = None,
    samplerate: int  = 16000,
    channels: int    = 1,
    device: int      = None,
) -> np.ndarray:
    """
    Record mono audio using an adaptive VAD pipeline starting on speech detection.
    """
    if samplerate not in (8000, 16000, 32000, 48000):
        raise ValueError(
            f"webrtcvad requires sample rate in (8000, 16000, 32000, 48000), "
            f"got {samplerate}."
        )
    if channels != 1:
        raise ValueError("webrtcvad requires mono (1-channel) audio.")

    frame_ms  = 30
    blocksize = int(samplerate * frame_ms / 1000)   # 480 samples @ 16 kHz
    frame_sec = frame_ms / 1000.0                    # 0.030 s per frame

    try:
        vad = webrtcvad.Vad(2)   # mode 2: medium aggressiveness
    except Exception as err:
        print(f"Error initialising WebRTC VAD: {err}", file=sys.stderr)
        raise

    max_sec    = duration if duration is not None else 60.0
    max_chunks = int(max_sec / frame_sec)

    try:
        with sd.InputStream(
            samplerate = samplerate,
            channels   = channels,
            dtype      = 'float32',
            blocksize  = blocksize,
            device     = device,
        ) as stream:

            # ── Adaptive noise-floor calibration ──────────────────────────
            calib_frames, energy_threshold = _calibrate_noise_floor(
                stream, blocksize, samplerate
            )
            # Rolling history of frame RMS values (last 150 frames = 4.5 seconds)
            rms_history = collections.deque(maxlen=150)
            if calib_frames:
                for f in calib_frames:
                    rms_history.append(calculate_rms(f.flatten()))
            else:
                rms_history.append(energy_threshold)

            # ── Gate-Opening Wait Loop (Wait for Speech) ──────────────────
            # Pre-roll lookback buffer (10 frames = 300 ms) to avoid clipping words
            preroll_buffer = collections.deque(maxlen=10)
            has_spoken = False
            frames_read = 0
            
            while True:
                # If a duration limit is specified, don't loop forever in tests/calls
                if duration is not None and frames_read >= max_chunks:
                    print("No speech detected. Saving audio for diagnostics...")
                    break

                data, _ = stream.read(blocksize)
                frames_read += 1
                preroll_buffer.append(data.copy())
                
                frame_rms = calculate_rms(data.flatten())
                rms_history.append(frame_rms)
                
                # Check if current RMS exceeds the calibrated energy threshold
                if frame_rms > energy_threshold:
                    print("[VAD] Speech started... Recording initialized.")
                    has_spoken = True
                    break

            if has_spoken:
                # Start recording: include the pre-roll frames to capture the beginning of the speech
                audio_chunks = list(preroll_buffer)
                silence_seconds = 0.0
                
                # Pre-fill speech window with True as speech has started
                speech_window = collections.deque(maxlen=SPEECH_WINDOW_FRAMES)
                for _ in range(SPEECH_WINDOW_FRAMES):
                    speech_window.append(True)

                # ── Main VAD loop ─────────────────────────────────────────────
                # Adjust remaining chunks
                remaining_chunks = max_chunks - frames_read
                for _ in range(max(0, remaining_chunks)):
                    data, _ = stream.read(blocksize)
                    audio_chunks.append(data.copy())

                    flat = data.flatten()

                    # Gate 1: adaptive energy with rolling noise floor
                    frame_rms    = calculate_rms(flat)
                    rms_history.append(frame_rms)
                    
                    # Dynamic noise floor is the 5th percentile of recent history
                    noise_floor  = np.percentile(list(rms_history), 5)
                    dynamic_threshold = max(noise_floor * 3.0, MIN_ENERGY_THRESHOLD)
                    above_energy = frame_rms > dynamic_threshold

                    # Gate 2: WebRTC VAD on 16-bit PCM
                    clipped    = np.clip(flat, -1.0, 1.0)
                    pcm_bytes  = (clipped * 32767).astype(np.int16).tobytes()
                    vad_speech = vad.is_speech(pcm_bytes, samplerate)

                    # Hybrid: both gates must agree
                    is_active = above_energy and vad_speech

                    # Sliding-window smoothing
                    speech_window.append(is_active)
                    smooth_speech = (
                        sum(speech_window) / len(speech_window)
                        >= SPEECH_ACTIVATION_RATIO
                    )

                    if smooth_speech:
                        silence_seconds = 0.0
                    else:
                        silence_seconds += frame_sec

                    # Termination (VAD silence limit check)
                    if silence_seconds >= POST_SPEECH_SILENCE_SEC:
                        print("Silence detected. Stopping recording...")
                        break
            else:
                # No speech was detected during the wait loop
                audio_chunks = list(preroll_buffer)

        if not audio_chunks:
            return np.zeros((0, 1), dtype='float32')

        audio_data = np.concatenate(audio_chunks, axis=0)

        if len(audio_data) > 0:
            print("Applying spectral noise reduction...")
            audio_data = nr.reduce_noise(y=audio_data.flatten(), sr=samplerate)
            audio_data = audio_data.reshape(-1, 1)

        return audio_data

    except Exception as exc:
        print(f"Error during audio recording: {exc}", file=sys.stderr)
        raise

def is_silent(audio_data: np.ndarray, threshold: float = 0.01) -> bool:
    """Return True if the audio's RMS is below *threshold*."""
    rms = calculate_rms(audio_data)
    if rms < threshold:
        print("No speech detected. Please try again.")
        return True
    return False

def save_audio(audio_data: np.ndarray, file_path: str, samplerate: int = 16000):
    """Write a float32 numpy array to a WAV file."""
    try:
        sf.write(file_path, audio_data, samplerate)
    except Exception as exc:
        print(f"Error saving audio file: {exc}", file=sys.stderr)
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Voice-Activated Audio Recorder with Adaptive VAD"
    )
    parser.add_argument("--duration",  type=float, default=None,
                        help="Maximum recording duration in seconds")
    parser.add_argument("--output",    type=str,   default="output.wav",
                        help="Output WAV file path")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="Silence RMS threshold (legacy, unused by adaptive VAD)")
    parser.add_argument("--device",    type=int,   default=None,
                        help="Input device index (optional)")
    args = parser.parse_args()

    print("Press Enter to trigger recording...")
    input()

    try:
        audio_data = record_audio(duration=args.duration, device=args.device)
    except Exception:
        sys.exit(1)

    save_audio(audio_data, args.output)
    print(f"Saved recording to {args.output}")

if __name__ == "__main__":
    main()
