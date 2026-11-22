"""MediaPipe Text Proofreader demosu: yazim ve dil bilgisi hatalarini duzeltir.

Kullanim: python demos/text_proofreader.py ["duzeltilecek ingilizce metin"]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import download_model

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/text_proofreader/200m/latest/"
             "proofread_quant_200m.litertlm")

BaseOptions = mp.tasks.BaseOptions
TextProofreader = mp.tasks.text.TextProofreader
TextProofreaderOptions = mp.tasks.text.TextProofreaderOptions
CorrectionType = mp.tasks.text.TextProofreaderCorrectionType

DEFAULT_TEXTS = ["Thier is a mistake here.", "I has went to the stor yesterday and buyed two apple."]


def main():
    texts = sys.argv[1:] or DEFAULT_TEXTS
    options = TextProofreaderOptions(
        base_options=BaseOptions(model_asset_path=download_model(MODEL_URL)),
        max_num_tokens=4096,
    )
    with TextProofreader.create_from_options(options) as proofreader:
        for text in texts:
            result = proofreader.proofread(text)
            print(f'Girdi : "{text}"')
            print(f'Duzeltilmis: "{result.proofread_text}"')
            for correction in result.corrections:
                if correction.type != CorrectionType.SAME:
                    print(f"  {correction.type.name}: {correction.text!r}")


if __name__ == "__main__":
    main()
