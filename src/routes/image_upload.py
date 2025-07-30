from flask import Blueprint, request, jsonify, Response, current_app
from werkzeug.datastructures import FileStorage
from PIL import Image
import numpy as np
import faiss

from src.config import Config
from src.db import Ocr, get_db_session

image_upload_bp = Blueprint('image_upload', __name__)
ALLOWED_EXTENSIONS: set[str] = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@image_upload_bp.route("/upload-image", methods=["POST"])
def upload() -> tuple[Response, int]:
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
    faiss.normalize_L2(img_feat)

    # Run OCR
    ocr_texts = current_app.imgrep.ocr.extract_text(pil_image)
    joined_ocr_texts = " ".join(ocr_texts).lower()

    # Saving the image embeddings in faiss
    img_index = faiss.read_index(f"{Config.FAISS_DATABASE}/{user_id}_img.faiss")
    idx = img_index.ntotal
    img_index.add(img_feat)
    faiss_path : str =f"{Config.FAISS_DATABASE}/{user_id}_img.faiss"
    faiss.write_index(img_index, faiss_path)

    # Saving the ocr text in the db only if there is a text
    if len(joined_ocr_texts) > 0:
        session = get_db_session()
        new_ocr = Ocr(user_id=user_id, faiss_id=str(idx), text=joined_ocr_texts)
        session.add(new_ocr)
        session.commit()
        session.close()

    return jsonify({
        "status": "ok",
        "index": idx,
        "message": f'{file.filename.split(".")[0]}',
    }), 200
