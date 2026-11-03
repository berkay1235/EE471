import base64
import os

import requests
from flask import Flask, jsonify, render_template, request

# Where the Cog container listens (override with the COG_URL environment variable)
COG_URL = os.environ.get("COG_URL", "http://localhost:5100/predictions")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    """Receive an uploaded image, send it to the Cog container, return the prediction."""
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded"}), 400

    # Cog expects files inside the JSON body as base64 data URIs
    mime = file.mimetype or "image/jpeg"
    encoded = base64.b64encode(file.read()).decode("ascii")
    payload = {"input": {"image": f"data:{mime};base64,{encoded}"}}

    try:
        response = requests.post(COG_URL, json=payload, timeout=60)
    except requests.RequestException:
        return jsonify({"error": "Cannot reach the Cog container"}), 502

    data = response.json()
    if data.get("status") != "succeeded":
        return jsonify({"error": data.get("error", "Prediction failed")}), 502

    return jsonify(data["output"])


if __name__ == "__main__":
    app.run(port=5200, debug=True)