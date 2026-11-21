"""MediaPipe Gesture Recognizer demosu: elin hareketini tanir
(Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou).

Kullanim: python demos/gesture_recognizer.py [gorsel_yolu]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import asset_path, download_model

MODEL_URL = "https://huggingface.co/litert-community/MediaPipe-GestureRecognizer/resolve/main/gesture_recognizer.task"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else asset_path("hand.jpg")

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
    )
    with GestureRecognizer.create_from_options(options) as recognizer:
        mp_image = mp.Image.create_from_file(image_path)
        result = recognizer.recognize(mp_image)

    print(f"Gorsel: {image_path}")
    print(f"Bulunan el sayisi: {len(result.gestures)}")
    for gestures, handedness in zip(result.gestures, result.handedness):
        gesture = gestures[0]
        print(f"  {handedness[0].category_name} el -> {gesture.category_name} ({gesture.score:.2f})")


if __name__ == "__main__":
    main()
