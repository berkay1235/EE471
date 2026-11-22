"""MediaPipe Image Embedder demosu: gorselleri vektore cevirir ve benzerliklerini hesaplar.

Kullanim: python demos/image_embedder.py [gorsel1 gorsel2 ...]
Varsayilan: assets/cat.png, assets/face-1.png, assets/pose-1.jpg
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import asset_path, download_model

MODEL_URL = ("https://huggingface.co/litert-community/MediaPipe-ImageEmbedder-MobileNet-V3-Small/"
             "resolve/main/mobilenet_v3_small.tflite")

BaseOptions = mp.tasks.BaseOptions
ImageEmbedder = mp.tasks.vision.ImageEmbedder
ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
RunningMode = mp.tasks.vision.RunningMode


def main():
    paths = sys.argv[1:] or [asset_path("cat.png"), asset_path("face-1.png"), asset_path("pose-1.jpg")]

    options = ImageEmbedderOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        running_mode=RunningMode.IMAGE,
        l2_normalize=True,
    )
    with ImageEmbedder.create_from_options(options) as embedder:
        embeddings = {}
        for path in paths:
            result = embedder.embed(mp.Image.create_from_file(path))
            embeddings[path] = result.embeddings[0]

    first = next(iter(embeddings.values()))
    print(f"Vektor boyutu: {len(first.embedding)}")
    for a, b in itertools.combinations(paths, 2):
        similarity = ImageEmbedder.cosine_similarity(embeddings[a], embeddings[b])
        print(f"Benzerlik {similarity:.3f}: {Path(a).name} <-> {Path(b).name}")


if __name__ == "__main__":
    main()
