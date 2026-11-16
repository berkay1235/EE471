"""MediaPipe Text Classifier demosu: cumlelerin duygusunu (positive/negative) siniflandirir.

Kullanim: python demos/text_classifier.py ["kendi cumlen"]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import download_model

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/text_classifier/"
             "bert_classifier/float32/latest/bert_classifier.tflite")

BaseOptions = mp.tasks.BaseOptions
TextClassifier = mp.tasks.text.TextClassifier
TextClassifierOptions = mp.tasks.text.TextClassifierOptions

DEFAULT_TEXTS = [
    "An unforgettable and genuinely moving film!",
    "The service was slow and the food was cold.",
]


def main():
    texts = sys.argv[1:] or DEFAULT_TEXTS
    options = TextClassifierOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)))

    with TextClassifier.create_from_options(options) as classifier:
        for text in texts:
            result = classifier.classify(text)
            best = max(result.classifications[0].categories, key=lambda c: c.score)
            print(f'"{text}"\n  -> {best.category_name} ({best.score:.3f})')


if __name__ == "__main__":
    main()
