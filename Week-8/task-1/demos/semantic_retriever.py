"""MediaPipe Semantic Retriever demosu: belgeleri indeksler ve anlama gore arar.

Kullanim: python demos/semantic_retriever.py ["kendi sorun"]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import download_model

MODEL_URL = ("https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/"
             "resolve/main/embeddinggemma-2-text-270m.litertlm")

BaseOptions = mp.tasks.BaseOptions
SemanticRetriever = mp.tasks.retrieval.SemanticRetriever
SemanticRetrieverOptions = mp.tasks.retrieval.SemanticRetrieverOptions
UniversalEmbedder = mp.tasks.retrieval.UniversalEmbedder
UniversalEmbedderOptions = mp.tasks.retrieval.UniversalEmbedderOptions

DOCUMENTS = {
    "paris": "Paris is the capital of France and is known for the Eiffel Tower.",
    "ankara": "Ankara is the capital city of Turkey.",
    "python": "Python is a popular programming language used for data science.",
    "tflite": "TensorFlow Lite runs machine learning models on mobile and edge devices.",
}
DEFAULT_QUERIES = ["Which city is the capital of France?", "What can I use to run ML models on a phone?"]


def main():
    queries = sys.argv[1:] or DEFAULT_QUERIES
    embedder_options = UniversalEmbedderOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        l2_normalize=True,
    )
    with UniversalEmbedder.create_from_options(embedder_options) as embedder:
        retriever_options = SemanticRetrieverOptions(embedder=embedder, embedding_dimension=768)
        with SemanticRetriever.create_from_options(retriever_options) as retriever:
            for doc_id, text in DOCUMENTS.items():
                retriever.insert_document(doc_id=doc_id, text=text)
            print(f"{len(DOCUMENTS)} belge indekslendi.")

            for query in queries:
                print(f'\nSorgu: "{query}"')
                for hit in retriever.retrieve(query=query, limit=2, min_similarity=0.3):
                    print(f"  {hit.score:.3f}  [{hit.id}] {hit.content}")


if __name__ == "__main__":
    main()
