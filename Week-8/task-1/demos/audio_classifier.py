"""MediaPipe Audio Classifier demosu (YAMNet): bir ses dosyasindaki sesin turunu bulur.

Kullanim: python demos/audio_classifier.py [ses_dosyasi.wav]
Not: 16-bit PCM WAV dosyasi beklenir (mono veya stereo; 16 kHz'e otomatik cevrilir).
"""
import sys
import wave
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
import numpy as np
from common import asset_path, download_model

MODEL_URL = "https://huggingface.co/litert-community/MediaPipe-AudioClassifier-YAMNet/resolve/main/yamnet.tflite"
TARGET_RATE = 16000

BaseOptions = mp.tasks.BaseOptions
AudioClassifier = mp.tasks.audio.AudioClassifier
AudioClassifierOptions = mp.tasks.audio.AudioClassifierOptions
AudioRunningMode = mp.tasks.audio.RunningMode
AudioData = mp.tasks.components.containers.AudioData


def load_wav(path: str) -> np.ndarray:
    """WAV dosyasini mono, 16 kHz, [-1, 1] araliginda float32 diziye cevirir."""
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

    options = AudioClassifierOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        running_mode=AudioRunningMode.AUDIO_CLIPS,
        max_results=-1,
    )
    audio_data = AudioData.create_from_array(samples, sample_rate=TARGET_RATE)
    with AudioClassifier.create_from_options(options) as classifier:
        results = classifier.classify(audio_data)

    totals = defaultdict(float)
    for window in results:
        for category in window.classifications[0].categories:
            totals[category.category_name] += category.score

    print(f"Ses dosyasi: {audio_path} ({len(samples) / TARGET_RATE:.1f} sn, {len(results)} pencere)")
    print("Ortalama skora gore en olasi 5 ses turu:")
    for name, total in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        print(f"  {name:<30} {total / len(results):.3f}")


if __name__ == "__main__":
    main()
