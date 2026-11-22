"""MediaPipe Holistic Landmarker demosu: yuz (478), vucut (33) ve el (21+21) noktalarini
tek seferde bulur.

Kullanim: python demos/holistic_landmarker.py [gorsel_yolu]
Cikti: outputs/holistic_landmarker.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
from common import OUTPUT_DIR, asset_path, download_model

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
PoseConnections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
HandConnections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
RunningMode = mp.tasks.vision.RunningMode


def flat(landmarks):
    """Sonuc bazen ic ice liste gelebilir; her zaman duz liste dondurur."""
    if landmarks and isinstance(landmarks[0], (list, tuple)):
        return list(landmarks[0])
    return list(landmarks or [])


def draw(image, landmarks, connections=None, color=(0, 255, 0)):
    height, width = image.shape[:2]
    pixels = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for connection in connections or []:
        cv2.line(image, pixels[connection.start], pixels[connection.end], color, 2)
    for pixel in pixels:
        cv2.circle(image, pixel, 2, (0, 0, 255), -1)


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else asset_path("pose-1.jpg")

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        running_mode=RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
    )
    with HolisticLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp.Image.create_from_file(image_path))

    parts = {
        "yuz": (flat(result.face_landmarks), None),
        "vucut": (flat(result.pose_landmarks), PoseConnections),
        "sol el": (flat(result.left_hand_landmarks), HandConnections),
        "sag el": (flat(result.right_hand_landmarks), HandConnections),
    }
    print(f"Gorsel: {image_path}")
    image = cv2.imread(image_path)
    for name, (landmarks, connections) in parts.items():
        print(f"  {name:<7} {len(landmarks)} nokta")
        if landmarks:
            draw(image, landmarks, connections)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "holistic_landmarker.jpg"
    cv2.imwrite(str(out_path), image)
    print(f"Cikti kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
