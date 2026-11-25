"""EE 471 - Week 9 - Task 1
Kullanım: python task1_pipelines.py <görev> [dosya_yolu]
Görevler: sentiment, zeroshot, generate, fillmask, ner, qa, summarize, translate, image, asr
"""
import sys
from transformers import pipeline

NER_SENTENCE = (
    "I am Nate, a research assistant at Izmir Institute of Technology, and currently "
    "living and working in the beautiful city İzmir in Türkiye."
)
CRISIS_TEXT = (
    "The 2008 Global Financial Crisis stands as the most severe economic collapse of the "
    "21st century, often compared to the Great Depression of the 1930s. Triggered by the "
    "bursting of the United States housing bubble, its effects rippled across the globe, "
    "leading to the collapse of major financial institutions and a deep international "
    "recession. The crisis began with the subprime mortgage market. In the early 2000s, "
    "low interest rates and a push for homeownership led banks to issue high-risk loans "
    "to borrowers with poor credit."
)


def sentiment():
    clf = pipeline("sentiment-analysis",
                   model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    sentences = [
        "I've been waiting for a modern software practices course my whole life.",
        "I hate software courses",
    ]
    for s, r in zip(sentences, clf(sentences)):
        print(f"{s!r} -> {r['label']} ({r['score']:.3f})")


def zeroshot():
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    text = "Berkshire keeps their cash reserves at an extremely high level."
    result = clf(text, candidate_labels=["finance", "politics", "sports", "technology", "health"])
    print(text)
    for label, score in zip(result["labels"], result["scores"]):
        print(f"  {label}: {score:.3f}")


def generate():
    gen = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
    prompt = "If I continue to complete all in-class exercises in this modern software practices course,"
    outputs = gen(prompt, max_new_tokens=45, num_return_sequences=2,
                  do_sample=True, temperature=0.8)
    for i, o in enumerate(outputs, 1):
        words = o["generated_text"].split()[:35]  # en fazla 35 kelime
        print(f"{i}. {' '.join(words)}\n")


def fillmask():
    unmasker = pipeline("fill-mask", model="distilbert/distilroberta-base")
    mask = unmasker.tokenizer.mask_token
    text = f"To understand generative AI, one must study {mask} well."
    for r in unmasker(text, top_k=5):
        print(f"{r['token_str'].strip():<15} {r['score']:.3f}  ->  {r['sequence']}")


def extract_entities():
    ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
                   aggregation_strategy="simple")
    ents = ner(NER_SENTENCE)
    person = next((e["word"] for e in ents if e["entity_group"] == "PER"), None)
    org = next((e["word"] for e in ents if e["entity_group"] == "ORG"), None)
    locs = [e["word"] for e in ents if e["entity_group"] == "LOC"]
    return ents, person, org, locs


def ner():
    ents, person, org, locs = extract_entities()
    for e in ents:
        print(f"{e['entity_group']:<4} {e['word']:<32} {e['score']:.3f}")
    print(f"\nKişi: {person} | Kurum: {org} | Şehir/yer: {locs}")


def qa():
    _, person, org, locs = extract_entities()
    reader = pipeline("question-answering",
                      model="distilbert/distilbert-base-cased-distilled-squad")
    checks = [
        ("What is the name of the person?", person),
        ("Which organization does the person work for?", org),
        ("Which city does the person live in?", locs[0] if locs else None),
    ]
    for question, expected in checks:
        ans = reader(question=question, context=NER_SENTENCE)["answer"]
        ok = expected is not None and (expected.lower() in ans.lower() or ans.lower() in expected.lower())
        print(f"{question}\n  QA: {ans} | NER: {expected} -> {'DOĞRULANDI' if ok else 'UYUŞMADI'}")


def summarize():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    print(summarizer(CRISIS_TEXT, max_length=60, min_length=20, do_sample=False)[0]["summary_text"])


def translate():
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
    text = ("The 2008 Global Financial Crisis stands as the most severe economic collapse "
            "of the 21st century, often compared to the Great Depression.")
    print(translator(text)[0]["translation_text"])


def image(path="sample.jpg"):
    clf = pipeline("image-classification", model="google/vit-base-patch16-224")
    for r in clf(path, top_k=3):
        print(f"{r['label']:<35} {r['score']:.3f}")


def asr(path="sample.wav"):
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                    chunk_length_s=30)
    print(pipe(path)["text"])


TASKS = {"sentiment": sentiment, "zeroshot": zeroshot, "generate": generate,
         "fillmask": fillmask, "ner": ner, "qa": qa, "summarize": summarize,
         "translate": translate, "image": image, "asr": asr}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TASKS:
        print("Kullanım: python task1_pipelines.py <görev> [dosya]\nGörevler:", ", ".join(TASKS))
        sys.exit(1)
    TASKS[sys.argv[1]](*sys.argv[2:])