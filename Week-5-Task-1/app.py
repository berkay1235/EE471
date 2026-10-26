import os
import tempfile
import threading

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

# --- 1) Load secret keys from .env -------------------------------------------
load_dotenv()
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")

app = Flask(__name__)


def make_speech_config():
    """Create an Azure SpeechConfig from the values in .env."""
    if not SPEECH_KEY or not SPEECH_REGION:
        raise RuntimeError("SPEECH_KEY / SPEECH_REGION is missing in .env")
    return speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)


# --- 2) Home page ------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# --- 3) Helper: recognize a whole WAV file -----------------------------------
def recognize_file(speech_config, wav_path):
    """Continuous recognition: works for long files, not only one sentence."""
    audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    pieces = []          # recognized sentences
    errors = []          # error messages from Azure
    done = threading.Event()

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
            pieces.append(evt.result.text)

    def on_canceled(evt):
        details = evt.cancellation_details
        if details.reason == speechsdk.CancellationReason.Error:
            errors.append(details.error_details)
        done.set()  # EndOfStream also lands here when the file is finished

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(lambda evt: done.set())

    recognizer.start_continuous_recognition()
    done.wait(timeout=120)
    recognizer.stop_continuous_recognition()

    if errors:
        raise RuntimeError(errors[0])
    return " ".join(pieces)


# --- 4) Speech-to-Text endpoint ----------------------------------------------
@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio = request.files.get("audio")
    language = request.form.get("language") or "en-US"

    if audio is None:
        return jsonify(error="Please record or upload an audio file."), 400

    try:
        speech_config = make_speech_config()
    except RuntimeError as e:
        return jsonify(error=str(e)), 500

    speech_config.speech_recognition_language = language

    # The browser already converted the audio to 16 kHz mono WAV.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio.save(tmp)
        wav_path = tmp.name

    try:
        text = recognize_file(speech_config, wav_path)
    except RuntimeError as e:
        return jsonify(error=f"Recognition failed: {e}"), 500
    finally:
        os.remove(wav_path)

    if not text:
        return jsonify(error="No speech could be recognized."), 422
    return jsonify(text=text)


if __name__ == "__main__":
    app.run(debug=True, port=5001)  # 5000 is used by AirPlay on macOS
