"""MediaPipe Audio Transcriber demosu (Parakeet): konusmayi yaziya cevirir.

Kullanim: python demos/audio_transcriber.py [ses_dosyasi.wav]
Not: Model ~615 MB, ilk calistirmada indirme birkac dakika surebilir.
"""
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
import numpy as np
from common import asset_path, download_model

MODEL_URL = ("https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/resolve/main/"
             "parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm")
TARGET_RATE = 16000

BaseOptions = mp.tasks.BaseOptions
AudioTranscriber = mp.tasks.audio.AudioTranscriber
AudioTranscriberOptions = mp.tasks.audio.AudioTranscriberOptions


def load_wav(path: str) -> np.ndarray:
    """16-bit PCM WAV dosyasini mono, 16 kHz, [-1, 1] float32 diziye cevirir."""
    with wave.open(path, "rb") as f:
        if f.getsampwidth() != 2:
            raise SystemExit("Sadece 16-bit PCM WAV destekleniyor.")
        rate, channels = f.getframerate(), f.getnchannels()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != TARGET_RATE:
        new_length = int(len(samples) * TARGET_RATE / rate)
        samples = np.interp(np.linspace(0, len(samples) - 1, new_length),
                            np.arange(len(samples)), samples).astype(np.float32)
    return samples


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else asset_path("speech.wav")
    samples = load_wav(audio_path)

    options = AudioTranscriberOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL),
                                 delegate=BaseOptions.Delegate.CPU),
        sample_rate_hz=TARGET_RATE,
    )
    with AudioTranscriber.create_from_options(options) as transcriber:
        result = transcriber.transcribe(samples, end_of_stream=True)

    print(f"Ses dosyasi: {audio_path} ({len(samples) / TARGET_RATE:.1f} sn)")
    print(f"Yazi: {result.text}")
    for segment in result.segments:
        print(f"  [{segment.from_timestamp_ms} - {segment.to_timestamp_ms} ms] {segment.transcript}")


if __name__ == "__main__":
    main()
