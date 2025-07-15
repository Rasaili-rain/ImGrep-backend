import threading
import time
from typing import Any, Dict, OrderedDict
from flask import Blueprint, request, jsonify, Response, current_app
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import os
import uuid
from PIL import Image
import numpy as np
import faiss

from src.utils.logger import logger
from src.config import Config

image_upload_bp = Blueprint('image_upload', __name__)

ALLOWED_EXTENSIONS: set[str] = {'png', 'jpg', 'jpeg', 'gif'}
MAX_JOBS: int = 64

# Simple job tracking
jobs: Dict[str, Dict[str, Any]] = OrderedDict()


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_in_background(job_id: str, file_path: str, filename: str):
    try:
        jobs[job_id]['status'] = 'processing'

        # TODO: IMGREP model
        time.sleep(100)

        jobs[job_id] = {
            'status': 'completed',
            'message': 'Processing completed',
            'filename': filename,
            'file_path': file_path
        }
        os.remove(file_path)
    except Exception as e:
        jobs[job_id] = {
            'status': 'error',
            'message': str(e),
            'filename': filename
        }


@image_upload_bp.route('/upload-image', methods=['POST'])
def upload_image() -> tuple[Response, int]:
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400

    file: FileStorage = request.files['image']
    if not file or not file.filename or not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid or no file selected'}), 400

    upload_folder: str = "uploads/"
    os.makedirs(upload_folder, exist_ok=True)

    filename: str = secure_filename(file.filename)
    extension: str = filename.rsplit('.', 1)[1].lower()
    unique_filename: str = f"{uuid.uuid4()}.{extension}"
    file_path: str = os.path.join(upload_folder, unique_filename)
    file.save(file_path)

    # Create job and start background processing
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'queued',
        'message': 'Processing started',
        'filename': unique_filename,
        'file_path': file_path
    }
    # Enforce job limit
    logger.info (jobs)
    if len(jobs) > MAX_JOBS:
        jobs.popitem()

    # Start background thread
    thread = threading.Thread(target=process_image_in_background, args=(job_id, file_path, unique_filename))
    thread.start()

    return jsonify({
        'status': 'success',
        'message': 'Image uploaded, processing in background',
        'job_id': job_id,
        'filename': unique_filename
    }), 200


@image_upload_bp.route('/status/<job_id>', methods=['GET'])
def get_status(job_id: str) -> tuple[Response, int]:
    if job_id not in jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

    return jsonify(jobs[job_id]), 200


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
    feat = current_app.imgrep.encode_image(pil_image).numpy().astype("float32")
    feat = np.expand_dims(feat, axis=0)

    # Saving in the faiss
    index = faiss.read_index(f"{Config.FAISS_DATABASE}/{user_id}_img.faiss")
    idx = index.ntotal
    index.add(feat)
    faiss.write_index(index, f"{Config.FAISS_DATABASE}/{user_id}_img.faiss")

    return jsonify({
        "status": "ok",
        "index": idx
    }), 200


@image_upload_bp.route("/test_search", methods=["POST"])
def test_search() -> tuple[Response, int]:
    payload = request.get_json()

    if "user_id" not in payload:
        return jsonify({"status": "error", "message": "user_id is required"})
    if "query" not in payload:
        return jsonify({"status": "error", "message": "query is required"})
    if "amount" not in payload:
        return jsonify({"status": "error", "message": "amount is required"})

    user_id = payload.get("user_id")
    query = payload.get("query")
    amount = payload.get("amount")

    feat = current_app.imgrep.encode_text(query).numpy().astype("float32")
    feat = np.expand_dims(feat, axis=0)

    index = faiss.read_index(f"{Config.FAISS_DATABASE}/{user_id}_img.faiss")
    dist, indices = index.search(feat, k = amount)

    return jsonify({
        "status": "ok",
        "distances": dist[-1].tolist(),
        "indices": indices[-1].tolist()
    }), 200


