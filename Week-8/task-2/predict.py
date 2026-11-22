import sys
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker_lite.task"

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
MIN_VISIBILITY = 0.5


def ensure_model():
    if not MODEL_PATH.exists():
        print("Model indiriliyor:", MODEL_URL, file=sys.stderr)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


def create_detector():
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def arm_is_up(landmarks, shoulder_idx, wrist_idx):
    shoulder = landmarks[shoulder_idx]
    wrist = landmarks[wrist_idx]
    if wrist.visibility < MIN_VISIBILITY or shoulder.visibility < MIN_VISIBILITY:
        return False
    return wrist.y < shoulder.y


def classify(image_path):
    detector = create_detector()
    image = mp.Image.create_from_file(str(image_path))
    result = detector.detect(image)
    if not result.pose_landmarks:
        return "None"
    landmarks = result.pose_landmarks[0]
    left_up = arm_is_up(landmarks, LEFT_SHOULDER, LEFT_WRIST)
    right_up = arm_is_up(landmarks, RIGHT_SHOULDER, RIGHT_WRIST)
    if left_up and right_up:
        return "both"
    if left_up:
        return "left"
    if right_up:
        return "right"
    return "None"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Kullanim: python predict.py <gorsel_yolu> [...]")
    for path in sys.argv[1:]:
        print(f"{path}: {classify(path)}")


if __name__ == "__main__":
    main()
