"""EE 471 - Week 9 - Task 3: Hugging Face görevleri için OOP modülü.

Her görev, ortak BaseTask sınıfından türeyen ayrı bir sınıftır. Modeller ilk
kullanımda yüklenir (lazy loading), böylece arayüz hızlı açılır.
"""
from transformers import pipeline


class BaseTask:
    """Tüm görevlerin ortak mantığı: model adı, tembel yükleme, run()."""

    task: str = ""
    default_model: str = ""

    def __init__(self, model: str | None = None):
        self.model_name = model or self.default_model
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = self._load()
        return self._pipe

    def _load(self):
        return pipeline(self.task, model=self.model_name)

    def run(self, *args, **kwargs):
        raise NotImplementedError


class SentimentAnalyzer(BaseTask):
    task = "sentiment-analysis"
    default_model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    def run(self, text: str) -> dict:
        results = self.pipe(text, top_k=None)
        if results and isinstance(results[0], list):
            results = results[0]
        return {r["label"]: float(r["score"]) for r in results}


class ZeroShotClassifier(BaseTask):
    task = "zero-shot-classification"
    default_model = "facebook/bart-large-mnli"

    def run(self, text: str, labels: str) -> dict:
        candidates = [l.strip() for l in labels.split(",") if l.strip()]
        out = self.pipe(text, candidate_labels=candidates)
        return dict(zip(out["labels"], map(float, out["scores"])))


class TextGenerator(BaseTask):
    task = "text-generation"
    default_model = "HuggingFaceTB/SmolLM2-360M"

    def run(self, prompt: str, max_words: int = 35, num_sequences: int = 2) -> str:
        max_words, n = int(max_words), int(num_sequences)
        outputs = self.pipe(prompt, max_new_tokens=int(max_words * 1.5) + 10,
                            num_return_sequences=n, do_sample=True, temperature=0.8)
        texts = [" ".join(o["generated_text"].split()[:max_words]) for o in outputs]
        return "\n\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))


class MaskFiller(BaseTask):
    task = "fill-mask"
    default_model = "distilbert/distilroberta-base"

    def run(self, text: str) -> dict:
        if "[MASK]" not in text:
            raise ValueError("Metinde [MASK] ifadesi olmalı.")
        text = text.replace("[MASK]", self.pipe.tokenizer.mask_token)
        return {r["token_str"].strip(): float(r["score"]) for r in self.pipe(text, top_k=5)}


class NamedEntityRecognizer(BaseTask):
    task = "ner"
    default_model = "dbmdz/bert-large-cased-finetuned-conll03-english"

    def _load(self):
        return pipeline(self.task, model=self.model_name, aggregation_strategy="simple")

    def run(self, text: str) -> dict:
        entities = [{"entity": e["entity_group"], "start": e["start"], "end": e["end"]}
                    for e in self.pipe(text)]
        return {"text": text, "entities": entities}


class QuestionAnswerer(BaseTask):
    task = "question-answering"
    default_model = "distilbert/distilbert-base-cased-distilled-squad"

    def run(self, question: str, context: str) -> str:
        out = self.pipe(question=question, context=context)
        return f"{out['answer']}  (score: {out['score']:.3f})"


class Summarizer(BaseTask):
    task = "summarization"
    default_model = "sshleifer/distilbart-cnn-12-6"

    def run(self, text: str) -> str:
        out = self.pipe(text, max_length=60, min_length=20, do_sample=False)
        return out[0]["summary_text"].replace(" .", ".").strip()


class Translator(BaseTask):
    task = "translation"
    default_model = "Helsinki-NLP/opus-mt-en-fr"

    def run(self, text: str) -> str:
        return self.pipe(text)[0]["translation_text"]


class ImageClassifier(BaseTask):
    task = "image-classification"
    default_model = "google/vit-base-patch16-224"

    def run(self, image) -> dict:
        if image is None:
            raise ValueError("Önce bir resim yükle.")
        return {r["label"]: float(r["score"]) for r in self.pipe(image, top_k=5)}


class SpeechRecognizer(BaseTask):
    task = "automatic-speech-recognition"
    default_model = "openai/whisper-large-v3"

    def _load(self):
        return pipeline(self.task, model=self.model_name, chunk_length_s=30)

    def run(self, audio_path: str) -> str:
        if not audio_path:
            raise ValueError("Önce bir ses dosyası yükle ya da kaydet.")
        return self.pipe(audio_path)["text"].strip()
