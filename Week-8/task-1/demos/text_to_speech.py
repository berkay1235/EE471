"""MediaPipe Text-to-Speech demosu (Phonon 31M): metni sese cevirir.

Kullanim: python demos/text_to_speech.py ["okunacak ingilizce metin"]
Cikti: outputs/text_to_speech.wav  (dinlemek icin: afplay outputs/text_to_speech.wav)
"""
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
import numpy as np
from common import OUTPUT_DIR, download_model

MODEL_URL = "https://huggingface.co/litert-community/phonon-31m-en/resolve/main/phonon_int8.litertlm"

BaseOptions = mp.tasks.BaseOptions
TextToSpeech = mp.tasks.audio.TextToSpeech
TextToSpeechOptions = mp.tasks.audio.TextToSpeechOptions

DEFAULT_TEXT = "Hello from MediaPipe! This voice was generated on my own computer."


def main():
    text = " ".join(sys.argv[1:]) or DEFAULT_TEXT
    options = TextToSpeechOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)))

    with TextToSpeech.create_from_options(options) as tts:
        voices = tts.list_voices()
        print(f"Kullanilabilir ses sayisi: {len(voices)}")
        for voice in voices[:5]:
            print(f"  {voice.voice_id} ({voice.display_name}, {voice.locale})")
        voice_id = voices[0].voice_id if voices else None
        result = tts.synthesize(text, voice_id=voice_id)

    samples = np.clip(np.asarray(result.samples, dtype=np.float32), -1.0, 1.0)
    pcm16 = (samples * 32767).astype("<i2")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "text_to_speech.wav"
    with wave.open(str(out_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(result.sample_rate_hz)
        f.writeframes(pcm16.tobytes())
    print(f'Metin: "{text}"')
    print(f"Sure: {len(samples) / result.sample_rate_hz:.1f} sn, {result.sample_rate_hz} Hz")
    print(f"Cikti kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
