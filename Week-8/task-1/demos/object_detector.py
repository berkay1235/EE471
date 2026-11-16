"""MediaPipe Object Detector demosu: goruntudeki nesneleri bulur ve kutu cizer.

Kullanim: python demos/object_detector.py [gorsel_yolu]
Cikti: outputs/object_detector.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
from common import OUTPUT_DIR, asset_path, download_model

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/object_detector/"
             "efficientdet_lite0/int8/latest/efficientdet_lite0.tflite")

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
RunningMode = mp.tasks.vision.RunningMode


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else asset_path("pose-1.jpg")

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        running_mode=RunningMode.IMAGE,
        score_threshold=0.3,
        max_results=5,
    )
    with ObjectDetector.create_from_options(options) as detector:
        mp_image = mp.Image.create_from_file(image_path)
        result = detector.detect(mp_image)

    image = cv2.imread(image_path)
    print(f"Gorsel: {image_path}")
    print(f"Bulunan nesne sayisi: {len(result.detections)}")

    for detection in result.detections:
        category = detection.categories[0]
        box = detection.bounding_box
        print(f"  {category.category_name:<15} guven={category.score:.2f} "
              f"kutu=({box.origin_x}, {box.origin_y}, {box.width}, {box.height})")
        top_left = (box.origin_x, box.origin_y)
        bottom_right = (box.origin_x + box.width, box.origin_y + box.height)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        label = f"{category.category_name} {category.score:.2f}"
        cv2.putText(image, label, (box.origin_x, max(box.origin_y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "object_detector.jpg"
    cv2.imwrite(str(out_path), image)
    print(f"Cikti kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
