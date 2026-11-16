"""MediaPipe Face Landmarker demosu: yuzdeki 478 noktayi bulur, kontur cizer
ve en belirgin 3 yuz ifadesi skorunu (blendshape) yazdirir.

Kullanim: python demos/face_landmarker.py [gorsel_yolu]
Cikti: outputs/face_landmarker.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
from common import OUTPUT_DIR, asset_path, download_model

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections
RunningMode = mp.tasks.vision.RunningMode


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else asset_path("face-1.png")

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
    )
    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(image_path)
        result = landmarker.detect(mp_image)

    print(f"Gorsel: {image_path}")
    print(f"Bulunan yuz sayisi: {len(result.face_landmarks)}")
    if not result.face_landmarks:
        return

    landmarks = result.face_landmarks[0]
    print(f"Nokta sayisi: {len(landmarks)}")

    blendshapes = sorted(result.face_blendshapes[0], key=lambda c: c.score, reverse=True)
    print("En belirgin yuz ifadeleri:")
    for category in blendshapes[:3]:
        print(f"  {category.category_name:<25} {category.score:.3f}")

    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    pixels = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for connection in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
        cv2.line(image, pixels[connection.start], pixels[connection.end], (0, 255, 0), 1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "face_landmarker.jpg"
    cv2.imwrite(str(out_path), image)
    print(f"Cikti kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
