"""MediaPipe Language Detector demosu: bir metnin hangi dilde yazildigini bulur.

Kullanim: python demos/language_detector.py ["kendi cumlen"]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import download_model

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/language_detector/"
             "language_detector/float32/latest/language_detector.tflite")

BaseOptions = mp.tasks.BaseOptions
LanguageDetector = mp.tasks.text.LanguageDetector
LanguageDetectorOptions = mp.tasks.text.LanguageDetectorOptions

DEFAULT_TEXTS = [
    "Merhaba, bugun hava cok guzel.",
    "The weather is really nice today.",
    "Das Wetter ist heute wirklich schoen.",
    "Il fait vraiment beau aujourd'hui.",
]


def main():
    texts = sys.argv[1:] or DEFAULT_TEXTS
    options = LanguageDetectorOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        max_results=3,
    )

    with LanguageDetector.create_from_options(options) as detector:
        for text in texts:
            result = detector.detect(text)
            print(f'"{text}"')
            for prediction in result.languages:
                print(f"  {prediction.language_code:<8} {prediction.probability:.3f}")


if __name__ == "__main__":
    main()
