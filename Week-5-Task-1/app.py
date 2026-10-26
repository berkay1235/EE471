import io
import os

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file

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


# --- 3) Text-to-Speech endpoint ----------------------------------------------
@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    voice = data.get("voice") or "en-US-JennyNeural"

    if not text:
        return jsonify(error="Please enter some text."), 400

    try:
        speech_config = make_speech_config()
    except RuntimeError as e:
        return jsonify(error=str(e)), 500

    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm  # .wav
    )

    # audio_config=None -> keep audio in memory instead of playing on server speakers
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return send_file(
            io.BytesIO(result.audio_data),
            mimetype="audio/wav",
            as_attachment=False,
            download_name="speech.wav",
        )

    details = result.cancellation_details
    return jsonify(error=f"Synthesis failed: {details.reason} {details.error_details}"), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)  # 5000 is used by AirPlay on macOS
