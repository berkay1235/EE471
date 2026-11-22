"""MediaPipe Universal Embedder demosu (EmbeddingGemma 270M): metinleri vektore cevirir.

Kullanim: python demos/universal_embedder.py
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import download_model

MODEL_URL = ("https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/"
             "resolve/main/embeddinggemma-2-text-270m.litertlm")

BaseOptions = mp.tasks.BaseOptions
UniversalEmbedder = mp.tasks.retrieval.UniversalEmbedder
UniversalEmbedderOptions = mp.tasks.retrieval.UniversalEmbedderOptions

SENTENCES = [
    "A cat is sleeping on the sofa.",
    "A kitten naps on the couch.",
    "The stock market fell sharply today.",
]


def main():
    options = UniversalEmbedderOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        l2_normalize=True,
    )
    with UniversalEmbedder.create_from_options(options) as embedder:
        embeddings = [embedder.embed_text(sentence).embeddings[0] for sentence in SENTENCES]

    print(f"Vektor boyutu: {len(embeddings[0].embedding)}")
    for i, j in itertools.combinations(range(len(SENTENCES)), 2):
        similarity = UniversalEmbedder.cosine_similarity(embeddings[i], embeddings[j])
        print(f'Benzerlik {similarity:.3f}: "{SENTENCES[i]}" <-> "{SENTENCES[j]}"')


if __name__ == "__main__":
    main()
