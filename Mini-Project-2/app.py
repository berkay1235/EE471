import os

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from library_classes import (
    Book,
    DVD,
    ItemNotFoundError,
    ItemStateError,
    LibraryCollection,
)
from voice_parser import parse_command
from voice_speech import speech_for

# Secrets (Azure key + region) live in .env, which is listed in .gitignore
load_dotenv()
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")
SPEECH_VOICE = os.getenv("SPEECH_VOICE", "en-US-JennyNeural")


app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")

library = LibraryCollection()

library.add_item(Book("1984", "George Orwell", 328))
library.add_item(DVD("Interstellar", "Christopher Nolan", 169))
library.add_item(Book("The Hobbit", "J.R.R. Tolkien", 310))


@app.route("/api/items", methods=["GET"])
def list_items():
    available_param = request.args.get("available")
    item_type = request.args.get("type")

    available = None

    if available_param is not None:
        if available_param.lower() == "true":
            available = True
        elif available_param.lower() == "false":
            available = False
        else:
            return jsonify({"error": "Invalid available filter"}), 400

    try:
        items = library.list_items(
            available=available,
            item_type=item_type
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify([item.to_dict() for item in items]), 200


@app.route("/api/items/<int:item_id>", methods=["GET"])
def get_item(item_id):
    try:
        item = library.get_item(item_id)
    except ItemNotFoundError as error:
        return jsonify({"error": str(error)}), 404

    return jsonify(item.to_dict()), 200


@app.route("/api/items", methods=["POST"])
def create_item():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        item_type = data["type"].lower()
        title = data["title"]
        author = data["author"]

        if item_type == "book":
            page_count = data["page_count"]
            item = Book(title, author, page_count)

        elif item_type == "dvd":
            duration = data["duration"]
            item = DVD(title, author, duration)

        else:
            return jsonify({"error": "Invalid item type"}), 400

    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    library.add_item(item)

    return jsonify(item.to_dict()), 201


@app.route("/api/items/<int:item_id>", methods=["PATCH"])
def patch_item(item_id):
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        item = library.update_item(item_id, **data)

    except ItemNotFoundError as error:
        return jsonify({"error": str(error)}), 404

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(item.to_dict()), 200


@app.route("/api/items/<int:item_id>", methods=["PUT"])
def replace_item(item_id):
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        item = library.get_item(item_id)

        required_fields = {"title", "author"}

        if isinstance(item, Book):
            required_fields.add("page_count")
        elif isinstance(item, DVD):
            required_fields.add("duration")

        if not required_fields.issubset(data.keys()):
            return jsonify({"error": "Missing required fields"}), 400

        editable_fields = {
            key: value
            for key, value in data.items()
            if key in {"title", "author", "page_count", "duration"}
        }

        item = library.update_item(item_id, **editable_fields)

    except ItemNotFoundError as error:
        return jsonify({"error": str(error)}), 404

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(item.to_dict()), 200


@app.route("/api/items/<int:item_id>", methods=["DELETE"])
def delete_item(item_id):
    try:
        library.remove_item(item_id)

    except ItemNotFoundError as error:
        return jsonify({"error": str(error)}), 404

    except ItemStateError as error:
        return jsonify({"error": str(error)}), 409

    return "", 204


@app.route("/api/items/<int:item_id>/checkout", methods=["POST"])
def checkout_item(item_id):
    try:
        item = library.get_item(item_id)
        message = item.checkout()

    except ItemNotFoundError as error:
        return jsonify({"error": str(error)}), 404

    except ItemStateError as error:
        return jsonify({"error": str(error)}), 409

    return jsonify({
        "message": message,
        "item": item.to_dict()
    }), 200


@app.route("/api/items/<int:item_id>/return", methods=["POST"])
def return_item(item_id):
    try:
        item = library.get_item(item_id)
        message = item.return_item()

    except ItemNotFoundError as error:
        return jsonify({"error": str(error)}), 404

    except ItemStateError as error:
        return jsonify({"error": str(error)}), 409

    return jsonify({
        "message": message,
        "item": item.to_dict()
    }), 200


# =====================================================================
# Voice layer (Mini Project 2). Everything above is the unchanged Week 4 API.
# =====================================================================

# Spoken sentences repeat a lot ("Sorry, item 3 is already checked out."),
# so the generated audio is kept in memory.
audio_cache = {}


# --- 1) Short-lived token for the browser's speech recognizer ---------
# The browser listens to the microphone itself, but it must never see
# SPEECH_KEY. The server exchanges the key for a 10-minute token.
@app.route("/api/speech-token")
def speech_token():
    if not SPEECH_KEY or not SPEECH_REGION:
        return jsonify({"error": "SPEECH_KEY / SPEECH_REGION is missing in .env"}), 500

    url = f"https://{SPEECH_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    try:
        response = requests.post(
            url, headers={"Ocp-Apim-Subscription-Key": SPEECH_KEY}, timeout=10
        )
        response.raise_for_status()
    except requests.RequestException as error:
        return jsonify({"error": f"Could not get a speech token: {error}"}), 500

    return jsonify({"token": response.text, "region": SPEECH_REGION})


# --- 2) Text-to-Speech: the librarian's voice -------------------------
@app.route("/api/speak", methods=["POST"])
def speak():
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
    if not text:
        return jsonify({"error": "Nothing to say"}), 400

    if text not in audio_cache:
        if not SPEECH_KEY or not SPEECH_REGION:
            return jsonify({"error": "SPEECH_KEY / SPEECH_REGION is missing in .env"}), 500

        import azure.cognitiveservices.speech as speechsdk  # only needed here

        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = SPEECH_VOICE
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm  # .wav
        )
        # audio_config=None keeps the audio in memory instead of using server speakers
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result = synthesizer.speak_text_async(text).get()

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = result.cancellation_details
            return jsonify({"error": f"Synthesis failed: {details.error_details}"}), 500
        audio_cache[text] = result.audio_data

    return Response(audio_cache[text], mimetype="audio/wav")


# --- 3) Run a voice command against the REST API ----------------------
def run_command(command):
    """Call the real REST endpoint for this command and describe what happened.

    The calls go through Flask's test client, so they hit exactly the same
    routes (and status codes) as Postman or the browser would.
    """
    client = app.test_client()
    trace = []   # shown in the page and printed in the terminal (handy in the demo video)

    def call(method, path, payload=None, query=None):
        options = {}
        if payload is not None:
            options["json"] = payload
        if query:
            options["query_string"] = query

        response = client.open(path, method=method, **options)
        shown_path = path + ("?" + "&".join(f"{k}={v}" for k, v in query.items()) if query else "")
        trace.append({"method": method, "path": shown_path, "status": response.status_code})
        print(f"[voice] {method} {shown_path} -> {response.status_code}")
        return response.status_code, response.get_json(silent=True)

    intent = command.intent
    item_id = command.item_id
    status, body, item_before = 400, None, None

    if intent == "create":
        status, body = call("POST", "/api/items", command.fields)

    elif intent == "list":
        status, body = call("GET", "/api/items", query=command.filters)

    elif intent in ("get", "replace", "update", "delete", "checkout", "return") and item_id is not None:
        path = f"/api/items/{item_id}"

        if intent == "get":
            status, body = call("GET", path)
        elif intent == "replace":
            status, body = call("PUT", path, command.fields)
        elif intent == "update":
            status, body = call("PATCH", path, command.fields)
        elif intent == "delete":
            # Look the title up first so the librarian can say it after deleting
            before_status, before_body = call("GET", path)
            item_before = before_body if before_status == 200 else None
            status, body = call("DELETE", path)
        elif intent == "checkout":
            status, body = call("POST", f"{path}/checkout")
        elif intent == "return":
            status, body = call("POST", f"{path}/return")

    return {
        "transcript": command.transcript,
        "intent": intent,
        "status": status,
        "speech": speech_for(command, status, body, item_before),
        "trace": trace,
    }


@app.route("/api/voice", methods=["POST"])
def voice_command():
    data = request.get_json(silent=True) or {}
    transcript = (data.get("transcript") or "").strip()

    if not transcript:
        return jsonify({"error": "No transcript provided"}), 400

    return jsonify(run_command(parse_command(transcript))), 200


if __name__ == "__main__":
    app.run(debug=True, port=5003)  # 5000 = AirPlay on macOS, 5001/5002 = Week 5