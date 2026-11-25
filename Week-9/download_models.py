from transformers import pipeline

pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
pipeline("image-classification", model="google/vit-base-patch16-224")
pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
print("Hepsi indi")