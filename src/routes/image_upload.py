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

@image_upload_bp.route("/upload-image", methods=["POST"])
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
    img_feat = current_app.imgrep.encode_image(pil_image).numpy().astype("float32") # type: ignore
    img_feat = np.expand_dims(img_feat, axis=0)

    # Run OCR
    ocr_texts = current_app.imgrep.ocr.extract_text(pil_image)
    joined_text = "\n".join(ocr_texts)
    text_feat = current_app.imgrep.encode_text(joined_text).numpy().astype("float32")
    text_feat = np.expand_dims(text_feat, axis=0)

    # Saving the image embeddings in faiss
    img_index = faiss.read_index(f"{Config.FAISS_DATABASE}/{user_id}_img.faiss")
    idx = img_index.ntotal
    img_index.add(img_feat)
    faiss_path : str =f"{Config.FAISS_DATABASE}/{user_id}_img.faiss"
    faiss.write_index(img_index, faiss_path)

    # Saving the ocr embeddings in faiss
    text_index = faiss.read_index(f"{Config.FAISS_DATABASE}/{user_id}_text.faiss")
    text_index.add(text_feat)
    faiss_path : str =f"{Config.FAISS_DATABASE}/{user_id}_text.faiss"
    faiss.write_index(text_index, faiss_path)

    return jsonify({
        "status": "ok",
        "index": idx,
        "message": f'{file.filename.split(".")[0]}',
    }), 200


@image_upload_bp.route("/search", methods=["POST"])
def test_search() -> tuple[Response, int]:
    payload = request.get_json()

    if "user_id" not in payload:
        return jsonify({"status": "error", "message": "user_id is required"}),400
    if "query" not in payload:
        return jsonify({"status": "error", "message": "query is required"}),400

    user_id = payload.get("user_id")
    query = payload.get("query")
    amount = payload.get("amount") if "amount" in payload else 5

    # Generate text embeddings of the query
    feat = current_app.imgrep.encode_text(query).numpy().astype("float32") # type: ignore
    feat = np.expand_dims(feat, axis=0)

    # Query for image
    img_index = faiss.read_index(f"{Config.FAISS_DATABASE}/{user_id}_img.faiss")
    img_dist, img_indices = img_index.search(feat, k = amount)

    # Query for ocr text
    text_index = faiss.read_index(f"{Config.FAISS_DATABASE}/{user_id}_text.faiss")
    text_dist, text_indices = text_index.search(feat, k = amount)

    IMAGE_WEIGHT = 0.7
    TEXT_WEIGHT = 0.3

    combined = {}

    # Appending image distance
    for d, i in zip(img_dist[0], img_indices[0]):
        combined[i] = combined.get(i, [0, 0])
        combined[i][0] = d  # image dist

    # Appending text distance
    for d, i in zip(text_dist[0], text_indices[0]):
        combined[i] = combined.get(i, [0, 0])
        combined[i][1] = d  # text dist

    # Weighted distance: lower is better
    final = sorted([
        (i, IMAGE_WEIGHT * v[0] + TEXT_WEIGHT * v[1])
        for i, v in combined.items()
    ], key=lambda x: x[1])

    final = final[:amount]
    indices = [int(i) for i, _ in final]
    distances = [float(d) for _, d in final]

    return jsonify({
        "status": "ok",
        "distances": distances,
        "indices": indices
    }), 200


