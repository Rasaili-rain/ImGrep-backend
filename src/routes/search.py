from rapidfuzz.fuzz import partial_ratio
from flask import Blueprint, request, jsonify, Response, current_app
import numpy as np
import faiss
from sqlalchemy import or_

from src.config import Config
from src.db import Ocr, get_db_session

search_bp = Blueprint("search", __name__)

@search_bp.route("/search", methods=["POST"])
def search() -> tuple[Response, int]:
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

    all_results = []

    #######################
    #    Text to Image    #
    #######################

    img_index = faiss.read_index(f"{Config.FAISS_DATABASE}/{user_id}_img.faiss")
    img_dist, img_indices = img_index.search(feat, k = amount)

    # Normalize distances to similarities
    max_dist = np.max(img_dist[0]) + 1e-5
    image_scores = { str(idx): 1 - (dist / max_dist) for idx, dist in zip(img_indices[0], img_dist[0]) }

    for faiss_id, img_score in image_scores.items():
        all_results.append((faiss_id, img_score))

    #######################
    #      OCR SEARCH     #
    #######################

    session = get_db_session()
    tokens = query.lower().split()

    # Query the ocr texts
    filters = [Ocr.text.ilike(f"%{token}%") for token in tokens]
    ocr_results = (
        session.query(Ocr)
        .filter(Ocr.user_id == user_id)
        .filter(or_(*filters))
        .all()
    )

    for ocr in ocr_results:
        ocr_score = 1 # partial_ratio(query.lower(), ocr.text.lower()) / 100.0
        all_results.append((ocr.faiss_id, ocr_score))

    all_results.sort(key=lambda x: x[1], reverse=True)

    best_result = {}
    for faiss_id, score in all_results:
        if faiss_id not in best_result or score > best_result[faiss_id]:
            best_result[faiss_id] = score

    print(best_result)
    final_result = list(best_result.items())

    distances = [float(score)    for _,        score in final_result]
    indices =   [int(faiss_id) for faiss_id, _     in final_result]
    print(indices)

    return jsonify({
        "status": "ok",
        "distances": distances,
        "indices": indices
    }), 200


