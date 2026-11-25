"""EE 471 - Week 9 - Task 3: Gradio arayüzü (her görev için bir sekme)."""
import gradio as gr
from nlp_tasks import (ImageClassifier, MaskFiller, NamedEntityRecognizer,
                       QuestionAnswerer, SentimentAnalyzer, SpeechRecognizer,
                       Summarizer, TextGenerator, Translator, ZeroShotClassifier)

CRISIS = ("The 2008 Global Financial Crisis stands as the most severe economic collapse of the "
          "21st century, often compared to the Great Depression of the 1930s. Triggered by the "
          "bursting of the United States housing bubble, its effects rippled across the globe, "
          "leading to the collapse of major financial institutions and a deep international "
          "recession. The crisis began with the subprime mortgage market. In the early 2000s, "
          "low interest rates and a push for homeownership led banks to issue high-risk loans "
          "to borrowers with poor credit.")
NATE = ("I am Nate, a research assistant at Izmir Institute of Technology, and currently "
        "living and working in the beautiful city İzmir in Türkiye.")

sentiment, zeroshot, generator = SentimentAnalyzer(), ZeroShotClassifier(), TextGenerator()
masker, ner, qa = MaskFiller(), NamedEntityRecognizer(), QuestionAnswerer()
summarizer, translator = Summarizer(), Translator()
image_clf, asr = ImageClassifier(), SpeechRecognizer()

tabs = {
    "Sentiment Analysis": gr.Interface(
        sentiment.run,
        gr.Textbox(label="Text", lines=3, value="I've been waiting for a modern software practices course my whole life."),
        gr.Label(label="Sentiment")),
    "Zero-shot Classification": gr.Interface(
        zeroshot.run,
        [gr.Textbox(label="Text", lines=3, value="Berkshire keeps their cash reserves at an extremely high level."),
         gr.Textbox(label="Candidate labels (comma separated)", value="finance, politics, sports, technology, health")],
        gr.Label(label="Classes")),
    "Text Generation": gr.Interface(
        generator.run,
        [gr.Textbox(label="Incomplete sentence", lines=2,
                    value="If I continue to complete all in-class exercises in this modern software practices course,"),
         gr.Slider(10, 60, value=35, step=1, label="Max words"),
         gr.Slider(1, 3, value=2, step=1, label="Number of alternatives")],
        gr.Textbox(label="Generated text", lines=8)),
    "Mask Filling": gr.Interface(
        masker.run,
        gr.Textbox(label="Text with [MASK]", value="To understand generative AI, one must study [MASK] well."),
        gr.Label(label="Top predictions")),
    "Named Entity Recognition": gr.Interface(
        ner.run, gr.Textbox(label="Text", lines=3, value=NATE),
        gr.HighlightedText(label="Entities")),
    "Question Answering": gr.Interface(
        qa.run,
        [gr.Textbox(label="Question", value="Which organization does the person work for?"),
         gr.Textbox(label="Context", lines=3, value=NATE)],
        gr.Textbox(label="Answer")),
    "Summarization": gr.Interface(
        summarizer.run, gr.Textbox(label="Text", lines=8, value=CRISIS),
        gr.Textbox(label="Summary", lines=4)),
    "Translation (EN→FR)": gr.Interface(
        translator.run,
        gr.Textbox(label="English text", lines=3,
                   value="The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, often compared to the Great Depression."),
        gr.Textbox(label="French translation", lines=3)),
    "Image Classification": gr.Interface(
        image_clf.run, gr.Image(type="pil", label="Image"), gr.Label(label="Top classes")),
    "Speech Recognition": gr.Interface(
        asr.run, gr.Audio(type="filepath", label="Audio (upload or record)"),
        gr.Textbox(label="Transcript", lines=4)),
}

demo = gr.TabbedInterface(list(tabs.values()), list(tabs.keys()),
                          title="EE 471 - Week 9 - Hugging Face Demo")

if __name__ == "__main__":
    demo.launch()
