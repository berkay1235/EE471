"""MediaPipe Pose Landmarker demosu: vucuttaki 33 noktayi bulur ve iskelet cizer.

Kullanim: python demos/pose_landmarker.py [gorsel_yolu]
Cikti: outputs/pose_landmarker.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
from common import OUTPUT_DIR, asset_path, download_model

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarksConnections = mp.tasks.vision.PoseLandmarksConnections
PoseLandmark = mp.tasks.vision.PoseLandmark
RunningMode = mp.tasks.vision.RunningMode


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else asset_path("pose-1.jpg")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )
    with PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(image_path)
        result = landmarker.detect(mp_image)

    print(f"Gorsel: {image_path}")
    print(f"Bulunan kisi sayisi: {len(result.pose_landmarks)}")
    if not result.pose_landmarks:
        return

    landmarks = result.pose_landmarks[0]
    for point in (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_WRIST,
                  PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_WRIST):
        lm = landmarks[point.value]
        print(f"  {point.name:<15} x={lm.x:.2f} y={lm.y:.2f}")

    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    pixels = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for connection in PoseLandmarksConnections.POSE_LANDMARKS:
        cv2.line(image, pixels[connection.start], pixels[connection.end], (0, 255, 0), 2)
    for pixel in pixels:
        cv2.circle(image, pixel, 4, (0, 0, 255), -1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "pose_landmarker.jpg"
    cv2.imwrite(str(out_path), image)
    print(f"Cikti kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
