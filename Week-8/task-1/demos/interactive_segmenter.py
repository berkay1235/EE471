"""MediaPipe Interactive Segmenter demosu: verdigin noktadaki nesneyi ayirir.

Kullanim: python demos/interactive_segmenter.py [gorsel_yolu] [x] [y]
x ve y 0-1 arasi normalize koordinatlardir (varsayilan 0.5 0.5 = gorselin ortasi).
Cikti: outputs/interactive_segmenter.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
import numpy as np
from common import OUTPUT_DIR, asset_path, download_model

MODEL_URL = ("https://huggingface.co/litert-community/MediaPipe-InteractiveSegmenter-MagicTouch/"
             "resolve/main/interactive_segmentation.task")

BaseOptions = mp.tasks.BaseOptions
InteractiveSegmenter = mp.tasks.vision.InteractiveSegmenter
InteractiveSegmenterOptions = mp.tasks.vision.InteractiveSegmenterOptions
BrushMode = mp.tasks.vision.InteractiveSegmenterBrushMode
Stroke = mp.tasks.vision.InteractiveSegmenterStroke
StrokePoint = mp.tasks.vision.InteractiveSegmenterStrokePoint


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else asset_path("cat.png")
    x = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    y = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    options = InteractiveSegmenterOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)))
    with InteractiveSegmenter.create_from_options(options) as segmenter:
        segmenter.set_image(mp.Image.create_from_file(image_path))
        strokes = [Stroke(brush_mode=BrushMode.POSITIVE, points=[StrokePoint(x=x, y=y)],
                          is_completed=True)]
        mask = np.squeeze(segmenter.segment(strokes).numpy_view())

    foreground = mask > 0.5
    print(f"Gorsel: {image_path}, nokta=({x}, {y})")
    print(f"Maske boyutu: {mask.shape[1]}x{mask.shape[0]}, on plan orani: %{100 * foreground.mean():.1f}")

    image = cv2.imread(image_path)
    foreground = cv2.resize(foreground.astype(np.uint8), (image.shape[1], image.shape[0])).astype(bool)
    overlay = image.copy()
    overlay[foreground] = (0.5 * overlay[foreground] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    center = (int(x * image.shape[1]), int(y * image.shape[0]))
    cv2.circle(overlay, center, 5, (0, 0, 255), -1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "interactive_segmenter.jpg"
    cv2.imwrite(str(out_path), np.hstack([image, overlay]))
    print(f"Cikti kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
