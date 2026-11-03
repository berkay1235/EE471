import os

import azure.cognitiveservices.speech as speechsdk
import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from custom_classes import Calculator

# --- 1) Load secret keys from .env -------------------------------------------
load_dotenv()
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")

app = Flask(__name__)
calculator = Calculator()

# Spoken words are reused a lot ("five", "plus"...), so keep their audio in memory
audio_cache = {}


# --- 2) Home page ------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# --- 3) Calculation (same as Week 4) -----------------------------------------
@app.route("/api/calculate", methods=["POST"])
def calculate():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        x = float(data["x"])
        y = float(data["y"])
        operation = data["operation"]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    try:
        if operation == "+":
            result = calculator.add(x, y)
        elif operation == "-":
            result = calculator.subtract(x, y)
        elif operation == "*":
            result = calculator.multiply(x, y)
        elif operation == "/":
            result = calculator.divide(x, y)
        else:
            return jsonify({"error": "Invalid operation"}), 400
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify({"result": result}), 200


# --- 4) Short-lived token for the browser's speech recognizer ----------------
# The browser listens to the microphone itself (real time), but it must never
# see SPEECH_KEY. So the server exchanges the key for a 10-minute token.
@app.route("/api/speech-token")
def speech_token():
    if not SPEECH_KEY or not SPEECH_REGION:
        return jsonify(error="SPEECH_KEY / SPEECH_REGION is missing in .env"), 500

    url = f"https://{SPEECH_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    try:
        r = requests.post(url, headers={"Ocp-Apim-Subscription-Key": SPEECH_KEY}, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        return jsonify(error=f"Could not get a speech token: {e}"), 500

    return jsonify(token=r.text, region=SPEECH_REGION)


# --- 5) Text-to-Speech: the calculator's voice -------------------------------
@app.route("/api/speak", methods=["POST"])
def speak():
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
    if not text:
        return jsonify(error="Nothing to say."), 400

    if text not in audio_cache:
        if not SPEECH_KEY or not SPEECH_REGION:
            return jsonify(error="SPEECH_KEY / SPEECH_REGION is missing in .env"), 500

        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm  # .wav
        )
        # audio_config=None -> keep audio in memory instead of playing on server speakers
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result = synthesizer.speak_text_async(text).get()

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = result.cancellation_details
            return jsonify(error=f"Synthesis failed: {details.error_details}"), 500
        audio_cache[text] = result.audio_data

    return Response(audio_cache[text], mimetype="audio/wav")


if __name__ == "__main__":
    app.run(debug=True, port=5002)  # 5000 = AirPlay on macOS, 5001 = Task 1
