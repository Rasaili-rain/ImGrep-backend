from datetime import date, datetime
from flask import Blueprint, request, jsonify, Response, current_app
from sqlalchemy import Date
from werkzeug.datastructures import FileStorage
from PIL import Image
import numpy as np
import faiss

from src.config import Config
from src.db import ImageTable, get_db_session
from src.utils.logger import logger


image_upload_bp = Blueprint('image_upload', __name__)
ALLOWED_EXTENSIONS: set[str] = {'png', 'jpg', 'jpeg', 'avif'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_date(date_str: str) -> date:
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
    except:
        logger.warning("fallback to todays date")
        return date.today()

def parse_float(value_str: str) -> float | None:
    if not value_str:
        return None
    try:
        return float(value_str)
    except:
        return None


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

    latitude = parse_float(request.form.get("latitude", ""))
    longitude = parse_float(request.form.get("longitude", ""))
    created_at = parse_date(request.form.get("created_at", ""))

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


    # Saving the ocr text in the db even when ocr text = ""
    session = get_db_session()
    new_image = ImageTable(
        user_id=user_id,
        faiss_id=str(idx),
        text=joined_ocr_texts,
        created_at=created_at,
        latitude=latitude,
        longitude=longitude,
        description = "" #TODO
    )
    session.add(new_image)
    session.commit()
    session.close()


    return jsonify({
        "status": "ok",
        "index": idx,
        "message": f'{file.filename.split(".")[0]}',
    }), 200


