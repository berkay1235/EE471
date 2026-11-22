"""MediaPipe Text Summarizer demosu: uzun metni TLDR ve madde madde ozetler.

Kullanim: python demos/text_summarizer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import download_model

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/text_summarizer/200m/latest/"
             "summarization_quant_200m_2modes.litertlm")

BaseOptions = mp.tasks.BaseOptions
TextSummarizer = mp.tasks.text.TextSummarizer
TextSummarizerOptions = mp.tasks.text.TextSummarizerOptions
TextSummarizerMode = mp.tasks.text.TextSummarizerMode

TEXT = (
    "MediaPipe is an open-source framework from Google for building machine learning pipelines. "
    "Its Tasks API lets developers run ready-made models for vision, audio and text on Android, iOS, "
    "the web and desktop with only a few lines of code. Because the models run on the device, "
    "applications work without a network connection, respond with low latency and keep user data "
    "private. Developers can also customize several of the tasks with their own data, for example by "
    "training a new image classifier or gesture recognizer with MediaPipe Model Maker."
)


def main():
    model_path = download_model(MODEL_URL)
    for mode in (TextSummarizerMode.TLDR, TextSummarizerMode.KEYPOINTS):
        options = TextSummarizerOptions(base_options=BaseOptions(model_asset_path=model_path), mode=mode)
        with TextSummarizer.create_from_options(options) as summarizer:
            result = summarizer.summarize(TEXT)
        print(f"--- {mode.name} ---")
        print(result.summary)


if __name__ == "__main__":
    main()
