import base64
import os

import requests
from flask import Flask, jsonify, render_template, request

# Where the Cog containers listen. COG_URL_OPT is optional: when it is set, the page
# shows the baseline and the optimized model side by side.
MODELS = {"Baseline": os.environ.get("COG_URL_BASE", "http://localhost:5111/predictions")}
if os.environ.get("COG_URL_OPT"):
    MODELS["Optimized"] = os.environ["COG_URL_OPT"]

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", models=list(MODELS))


def ask_cog(url, payload):
    """Send one prediction request to a Cog container and return its output (or an error)."""
    try:
        response = requests.post(url, json=payload, timeout=60)
        data = response.json()
    except (requests.RequestException, ValueError):
        return {"error": "Cannot reach the Cog container"}
    if data.get("status") != "succeeded":
        return {"error": data.get("error", "Prediction failed")}
    return data["output"]


@app.route("/classify", methods=["POST"])
def classify():
    """Receive an uploaded image, ask every model, return {model name: prediction}."""
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded"}), 400

    # Cog expects files inside the JSON body as base64 data URIs
    mime = file.mimetype or "image/jpeg"
    encoded = base64.b64encode(file.read()).decode("ascii")
    payload = {"input": {"image": f"data:{mime};base64,{encoded}"}}

    return jsonify({name: ask_cog(url, payload) for name, url in MODELS.items()})


if __name__ == "__main__":
    app.run(port=5300, debug=True)
