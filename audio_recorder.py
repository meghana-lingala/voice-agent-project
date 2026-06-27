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
# These are all tuneable; the adaptive calibration is the key improvement.

# Seconds of pre-roll audio read at stream-open to measure the ambient noise.
NOISE_CALIBRATION_SECS  = 0.4

# Dynamic threshold = ambient_rms * this multiplier.
# 4× the noise floor means only frames that are significantly louder than
# background hum / fan noise will pass the energy gate.
NOISE_FLOOR_MULTIPLIER  = 4.0

# Absolute floor so the threshold never collapses to near-zero in a very
# quiet room (which would let background noise through).
MIN_ENERGY_THRESHOLD    = 0.0008

# Sliding-window smoothing: how many 30 ms frames to look back.
# 10 frames = 300 ms of history.
SPEECH_WINDOW_FRAMES    = 10

# Fraction of the window that must be "active speech" for the window to be
# considered speaking.  0.40 = need 4 out of 10 frames → bridges natural
# inter-word pauses without letting noise spikes trigger false speech.
SPEECH_ACTIVATION_RATIO = 0.40

# Seconds of silence (post-speech) before stopping.
POST_SPEECH_SILENCE_SEC = 2.0

# Seconds of silence before any speech → give up waiting.
INITIAL_SILENCE_SEC     = 5.0
# ──────────────────────────────────────────────────────────────────────────────


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

    Args:
        stream    : an open sounddevice.InputStream
        blocksize : samples per frame
        samplerate: Hz

    Returns:
        (calib_frames, energy_threshold)
        calib_frames     – list[np.ndarray] captured during calibration
                           (included in the final recording so no audio is lost)
        energy_threshold – float, the derived RMS threshold
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
    Record mono audio using an adaptive, hybrid VAD pipeline.

    ── Per 30 ms frame ──────────────────────────────────────────────────────
    1. Adaptive energy gate
       The energy threshold is calibrated at recording start by measuring
       the ambient noise floor (fan hum, keyboard clicks, room noise).
       Only frames whose RMS exceeds  noise_floor × 4  pass the gate.

    2. WebRTC VAD (mode 2 – medium aggressiveness)
       A frame must also pass WebRTC's voice-activity check.
       Both gates must agree for a frame to count as speech.

    3. Sliding-window smoothing (300 ms)
       The last 10 frames are tracked; the window is counted as "speaking"
       only when ≥ 40 % of frames are active speech.  This bridges natural
       inter-word pauses (∼200 ms) without letting noise spikes keep the
       recorder alive.

    ── Termination rules ────────────────────────────────────────────────────
    • 5 s of silence before any speech detected  → stop (diagnostic save)
    • 2 s of silence after speech has been heard → stop (transcribe)
    • 60 s hard cap (or explicit *duration* argument)

    Args:
        duration   : hard-cap in seconds (default 60)
        samplerate : must be 8000 / 16000 / 32000 / 48000 Hz
        channels   : must be 1 (mono)
        device     : sounddevice input device index (None = system default)

    Returns:
        np.ndarray of shape (N, 1), dtype float32 – noise-reduced audio.
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

    has_spoken      = False
    silence_seconds = 0.0

    # Pre-fill the sliding window with silence so we don't need a warm-up.
    speech_window = collections.deque(maxlen=SPEECH_WINDOW_FRAMES)
    for _ in range(SPEECH_WINDOW_FRAMES):
        speech_window.append(False)

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

            # Include calibration frames in output (audio before press-Enter)
            audio_chunks = list(calib_frames)

            # ── Main VAD loop ─────────────────────────────────────────────
            for _ in range(max_chunks):
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
                    if not has_spoken:
                        print("Speech detected!")
                        has_spoken = True
                    silence_seconds = 0.0
                else:
                    silence_seconds += frame_sec

                # Termination
                limit = POST_SPEECH_SILENCE_SEC if has_spoken else INITIAL_SILENCE_SEC
                if silence_seconds >= limit:
                    if not has_spoken:
                        print("No speech detected. Saving audio for diagnostics...")
                    else:
                        print("Silence detected. Stopping recording...")
                    break

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
