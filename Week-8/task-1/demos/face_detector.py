"""MediaPipe Face Detector demosu: yuzleri bulur, kutu ve 6 anahtar noktayi cizer.

Kullanim: python demos/face_detector.py [gorsel_yolu]
Cikti: outputs/face_detector.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
from common import OUTPUT_DIR, asset_path, download_model

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else asset_path("face.jpg")

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        running_mode=RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    with FaceDetector.create_from_options(options) as detector:
        mp_image = mp.Image.create_from_file(image_path)
        result = detector.detect(mp_image)

    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    print(f"Bulunan yuz sayisi: {len(result.detections)}")

    for i, detection in enumerate(result.detections, start=1):
        box = detection.bounding_box
        score = detection.categories[0].score
        print(f"  Yuz {i}: guven={score:.2f}, kutu=({box.origin_x}, {box.origin_y}, {box.width}, {box.height})")
        cv2.rectangle(image, (box.origin_x, box.origin_y),
                      (box.origin_x + box.width, box.origin_y + box.height), (0, 255, 0), 2)
        for keypoint in detection.keypoints:
            cv2.circle(image, (int(keypoint.x * width), int(keypoint.y * height)), 4, (0, 0, 255), -1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "face_detector.jpg"
    cv2.imwrite(str(out_path), image)
    print(f"Cikti kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
