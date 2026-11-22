"""MediaPipe Decision Maker demosu (EmbeddingGemma 270M): metne gore secim yapar.

Kullanim: python demos/decision_maker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import download_model

MODEL_URL = ("https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/"
             "resolve/main/embeddinggemma-2-text-270m.litertlm")

BaseOptions = mp.tasks.BaseOptions
DecisionMaker = mp.tasks.decision.DecisionMaker
DecisionMakerOptions = mp.tasks.decision.DecisionMakerOptions
ChoiceQuestion = mp.tasks.decision.ChoiceQuestion
BooleanQuestion = mp.tasks.decision.BooleanQuestion

MESSAGES = [
    "The screen freezes every time I open settings.",
    "I was charged twice for my subscription this month.",
    "Can you add a dark mode to the app?",
]


def main():
    options = DecisionMakerOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        max_num_tokens=512,
    )
    choice = ChoiceQuestion(
        criteria={
            "payments": "Card charges, receipts, or refund status.",
            "bug": "Freezes, error screens, or features that stop working.",
            "feature_request": "Suggestions for new features or improvements.",
        },
        normalize_prior=True,
    )
    urgent = BooleanQuestion(condition="The customer reports a problem that blocks them from using the app.")

    with DecisionMaker.create_from_options(options) as decision_maker:
        decision_maker.prewarm(choice)
        for message in MESSAGES:
            result = decision_maker.evaluate_choice(message, choice)
            blocking = decision_maker.evaluate_boolean(message, urgent)
            print(f'"{message}"')
            print(f"  kategori: {result.selected_key}")
            print(f"  olasiliklar: {result.probabilities}")
            print(f"  engelleyici sorun mu: {blocking.value} (p={blocking.probability_true:.2f})")


if __name__ == "__main__":
    main()
