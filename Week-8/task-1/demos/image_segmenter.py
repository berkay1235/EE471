"""MediaPipe Image Segmenter demosu (DeepLab-V3): her pikseli bir sinifa atar.

Kullanim: python demos/image_segmenter.py [gorsel_yolu]
Cikti: outputs/image_segmenter.jpg (orijinal + renkli maske)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
import numpy as np
from common import OUTPUT_DIR, asset_path, download_model

MODEL_URL = ("https://huggingface.co/litert-community/MediaPipe-ImageSegmenter-DeepLab-V3/"
             "resolve/main/deeplab_v3.tflite")

# DeepLab-V3 PASCAL VOC siniflari (standart siralama, 21 sinif)
LABELS = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
          "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
          "sheep", "sofa", "train", "tv"]

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
RunningMode = mp.tasks.vision.RunningMode


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else asset_path("cat.png")

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        running_mode=RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=False,
    )
    with ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp.Image.create_from_file(image_path))

    mask = np.squeeze(result.category_mask.numpy_view()).astype(np.uint8)
    print(f"Gorsel: {image_path}")
    print(f"Maske boyutu: {mask.shape[1]}x{mask.shape[0]}")
    print("Piksel dagilimi:")
    for index, count in zip(*np.unique(mask, return_counts=True)):
        name = LABELS[index] if index < len(LABELS) else str(index)
        print(f"  {index:>2} {name:<12} %{100 * count / mask.size:.1f}")

    image = cv2.imread(image_path)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    palette = np.random.RandomState(0).randint(60, 255, size=(256, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    overlay = cv2.addWeighted(image, 0.5, palette[mask], 0.5, 0)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "image_segmenter.jpg"
    cv2.imwrite(str(out_path), np.hstack([image, overlay]))
    print(f"Cikti kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
