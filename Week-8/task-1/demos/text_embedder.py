"""MediaPipe Text Embedder demosu: cumleleri vektore cevirir ve benzerliklerini hesaplar.

Kullanim: python demos/text_embedder.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import download_model

MODEL_URL = ("https://huggingface.co/litert-community/MediaPipe-TextEmbedder-Universal-Sentence-Encoder/"
             "resolve/main/universal_sentence_encoder.tflite")

BaseOptions = mp.tasks.BaseOptions
TextEmbedder = mp.tasks.text.TextEmbedder
TextEmbedderOptions = mp.tasks.text.TextEmbedderOptions

SENTENCES = [
    "It's a charming and affecting journey.",
    "What a great and fantastic trip!",
    "The stock market fell sharply today.",
]


def main():
    options = TextEmbedderOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)))

    with TextEmbedder.create_from_options(options) as embedder:
        embeddings = [embedder.embed(sentence).embeddings[0] for sentence in SENTENCES]

    print(f"Vektor boyutu: {len(embeddings[0].embedding)}")
    for i in range(len(SENTENCES)):
        for j in range(i + 1, len(SENTENCES)):
            similarity = TextEmbedder.cosine_similarity(embeddings[i], embeddings[j])
            print(f'Benzerlik {similarity:.3f}: "{SENTENCES[i]}" <-> "{SENTENCES[j]}"')


if __name__ == "__main__":
    main()
