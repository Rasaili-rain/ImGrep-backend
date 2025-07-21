from flask import Blueprint, request, jsonify, Response, current_app
from werkzeug.datastructures import FileStorage
from PIL import Image
import numpy as np
import faiss

from src.config import Config

image_upload_bp = Blueprint('image_upload', __name__)

ALLOWED_EXTENSIONS: set[str] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@image_upload_bp.route("/test_upload", methods=["POST"])
def test_upload() -> tuple[Response, int]:
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400

    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "message": "user_id not provided"}), 400

    file: FileStorage = request.files['image']
    if not file or not file.filename or not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid or no file selected'}), 400

    # Load the image using PIL
    pil_image = Image.open(file.stream).convert("RGB")

    # Generate image embeddings
    feat = current_app.imgrep.encode_image(pil_image).numpy().astype("float32") # type: ignore
    feat = np.expand_dims(feat, axis=0)

    # Saving in the faiss
    index = faiss.read_index(f"{Config.FAISS_DATABASE}/{user_id}_img.faiss")
    idx = index.ntotal
    index.add(feat)
    fiass_path : str =f"{Config.FAISS_DATABASE}/{user_id}_img.faiss"
    faiss.write_index(index, fiass_path)

    return jsonify({
        "status": "ok",
        "index": idx,
        "message": f"{fiass_path} updated sucessfully with image {file.filename}"
    }), 200


@image_upload_bp.route("/test_search", methods=["POST"])
def test_search() -> tuple[Response, int]:
    payload = request.get_json()

    if "user_id" not in payload:
        return jsonify({"status": "error", "message": "user_id is required"}),400
    if "query" not in payload:
        return jsonify({"status": "error", "message": "query is required"}),400

    user_id = payload.get("user_id")
    query = payload.get("query")
    amount = payload.get("amount") if "amount" in payload else 5

    feat = current_app.imgrep.encode_text(query).numpy().astype("float32") # type: ignore
    feat = np.expand_dims(feat, axis=0)

    index = faiss.read_index(f"{Config.FAISS_DATABASE}/{user_id}_img.faiss")
    dist, indices = index.search(feat, k = amount)

    return jsonify({
        "status": "ok",
        "distances": dist[-1].tolist(),
        "indices": indices[-1].tolist()
    }), 200


