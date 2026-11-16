"""MediaPipe Image Classifier demosu: bir fotografi ImageNet siniflarina gore siniflandirir.

Kullanim: python demos/image_classifier.py [gorsel_yolu]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import asset_path, download_model

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_classifier/"
             "efficientnet_lite0/float32/latest/efficientnet_lite0.tflite")

BaseOptions = mp.tasks.BaseOptions
ImageClassifier = mp.tasks.vision.ImageClassifier
ImageClassifierOptions = mp.tasks.vision.ImageClassifierOptions
RunningMode = mp.tasks.vision.RunningMode


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else asset_path("image.jpg")

    options = ImageClassifierOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        running_mode=RunningMode.IMAGE,
        max_results=5,
    )
    with ImageClassifier.create_from_options(options) as classifier:
        mp_image = mp.Image.create_from_file(image_path)
        result = classifier.classify(mp_image)

    print(f"Gorsel: {image_path}")
    for category in result.classifications[0].categories:
        print(f"  {category.category_name:<30} {category.score:.3f}")


if __name__ == "__main__":
    main()
