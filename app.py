"""Flask API for semantic and OCR-assisted product search."""
from functools import lru_cache

import cv2
import numpy as np
from easyocr import Reader
from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from similarity_search import similarity_search


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}


@lru_cache(maxsize=1)
def get_ocr_reader():
    return Reader(["en"], gpu=False)


def get_top_k():
    try:
        return int(request.args.get("top_k", 5))
    except (TypeError, ValueError):
        raise ValueError("top_k must be an integer.")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/search")
def search():
    payload = request.get_json(silent=True) or {}
    query = payload.get("query", "")
    products = similarity_search(query, top_k=get_top_k())
    return jsonify({"query": query, "products": products})


@app.post("/api/ocr-search")
def ocr_search():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Upload an image using the 'file' field."}), 400

    extension = uploaded.filename.rsplit(".", 1)[-1].lower() if "." in uploaded.filename else ""
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({"error": "Unsupported image format."}), 400

    image_bytes = uploaded.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "The uploaded file is not a valid image."}), 400

    extracted_lines = get_ocr_reader().readtext(image, detail=0)
    query = " ".join(line.strip() for line in extracted_lines if line.strip())
    if not query:
        return jsonify({"error": "No readable text was found in the image."}), 422

    products = similarity_search(query, top_k=get_top_k())
    return jsonify({"query": query, "products": products})


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return jsonify({"error": "Image exceeds the 8 MB upload limit."}), 413


@app.errorhandler(ValueError)
def handle_bad_request(error):
    return jsonify({"error": str(error)}), 400


@app.errorhandler(RuntimeError)
def handle_configuration_error(error):
    return jsonify({"error": str(error)}), 503


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
