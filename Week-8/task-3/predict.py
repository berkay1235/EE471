import sys
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"

NOSE_TIP = 1
FACE_EDGE_IMAGE_LEFT = 234
FACE_EDGE_IMAGE_RIGHT = 454

LOW_THRESHOLD = 0.40
HIGH_THRESHOLD = 0.60


def ensure_model():
    if not MODEL_PATH.exists():
        print("Model indiriliyor:", MODEL_URL, file=sys.stderr)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


def create_detector():
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def nose_ratio(landmarks):
    nose_x = landmarks[NOSE_TIP].x
    left_x = landmarks[FACE_EDGE_IMAGE_LEFT].x
    right_x = landmarks[FACE_EDGE_IMAGE_RIGHT].x
    return (nose_x - left_x) / (right_x - left_x)


def direction_from_ratio(ratio):
    if ratio > HIGH_THRESHOLD:
        return "left"
    if ratio < LOW_THRESHOLD:
        return "right"
    return "straight"


def classify(image_path):
    detector = create_detector()
    image = mp.Image.create_from_file(str(image_path))
    result = detector.detect(image)
    if not result.face_landmarks:
        return "None", None
    ratio = nose_ratio(result.face_landmarks[0])
    return direction_from_ratio(ratio), ratio


def main():
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    verbose = "--verbose" in sys.argv[1:]
    if not args:
        raise SystemExit("Kullanim: python predict.py [--verbose] <gorsel_yolu> [...]")
    for path in args:
        label, ratio = classify(path)
        if verbose and ratio is not None:
            print(f"{path}: {label} (burun orani={ratio:.2f})")
        else:
            print(f"{path}: {label}")


if __name__ == "__main__":
    main()
