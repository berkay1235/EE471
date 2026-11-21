"""MediaPipe Hand Landmarker demosu: eldeki 21 noktayi bulur ve iskelet cizer.

Kullanim: python demos/hand_landmarker.py [gorsel_yolu]
Cikti: outputs/hand_landmarker.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
from common import OUTPUT_DIR, asset_path, download_model

MODEL_URL = "https://huggingface.co/litert-community/MediaPipe-HandLandmarker-Full/resolve/main/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
RunningMode = mp.tasks.vision.RunningMode


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else asset_path("hand.jpg")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
    )
    with HandLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(image_path)
        result = landmarker.detect(mp_image)

    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    print(f"Gorsel: {image_path}")
    print(f"Bulunan el sayisi: {len(result.hand_landmarks)}")

    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        side = handedness[0]
        print(f"  {side.category_name} el (skor={side.score:.2f}), {len(landmarks)} nokta")
        pixels = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
        for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(image, pixels[connection.start], pixels[connection.end], (0, 255, 0), 2)
        for pixel in pixels:
            cv2.circle(image, pixel, 4, (0, 0, 255), -1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "hand_landmarker.jpg"
    cv2.imwrite(str(out_path), image)
    print(f"Cikti kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
