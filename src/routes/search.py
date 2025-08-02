from datetime import datetime
from rapidfuzz.fuzz import token_set_ratio
from flask import Blueprint, request, jsonify, Response, current_app
import numpy as np
import faiss
import json

from src.config import Config
from src.db import ImageTable, get_db_session
import src.imgrep.date_time_parser.date_time_parser as dt

search_bp = Blueprint("search", __name__)


def datetime_match_boost(all_results, images_from_db, datetime_ranges):
    if not datetime_ranges or not datetime_ranges.datetime_ranges:
        return all_results

    for image in images_from_db:
        if not image.created_at:
            continue

        image_time = image.created_at
        for time_range in datetime_ranges.datetime_ranges:
            try:
                start_time = datetime.fromisoformat(time_range.start_datetime)
                end_time = datetime.fromisoformat(time_range.end_datetime)
                if start_time <= image_time <= end_time:
                    if image.faiss_id in all_results:
                        all_results[image.faiss_id] += Config.DATE_TIME_BOOST_AMOUNT
            except (ValueError, AttributeError):
                continue

    return all_results


@search_bp.route("/search", methods=["POST"])
def search() -> tuple[Response, int]:
    payload = request.get_json()

    if "user_id" not in payload:
        return jsonify({"status": "error", "message": "user_id is required"}), 400
    if "query" not in payload:
        return jsonify({"status": "error", "message": "query is required"}), 400

    user_id = payload.get("user_id")
    query = payload.get("query")
    amount = payload.get("amount") if "amount" in payload else 5

    # --- Parse the date from the query --- #
    extractor = dt.DateTimeRangeExtractor()
    datetime_ranges = extractor.extract_datetime_ranges(query)
    print(extractor.to_json(datetime_ranges))
    # -- --- --- -- #

    # Generate text embeddings of the query
    feat = current_app.imgrep.encode_text(query).numpy().astype("float32")  # type: ignore
    feat = np.expand_dims(feat, axis=0)
    faiss.normalize_L2(feat)

    all_results = {}

    #######################
    #    Text to Image    #
    #######################

    img_index = faiss.read_index(f"{Config.FAISS_DATABASE}/{user_id}_img.faiss")
    img_dist, img_indices = img_index.search(feat, k=amount)

    # Converting from numpy to python list
    img_dist = img_dist.tolist()
    img_indices = img_indices.tolist()

    # Clean the indices and distance as there can be -1 indices and large distances
    cleaned_indices = []
    cleaned_dists = []

    for idx, dist in zip(img_indices[0], img_dist[0]):
        if idx != 1 and dist < 1e10:
            cleaned_indices.append(idx)
            cleaned_dists.append(dist)

    img_indices = [cleaned_indices]
    img_dist = [cleaned_dists]

    # Normalize distances to similarities
    max_dist = np.max(img_dist[0])
    image_scores = {
        str(idx): 1 - (dist / max_dist)
        for idx, dist in zip(img_indices[0], img_dist[0])
    }

    for faiss_id, img_score in image_scores.items():
        all_results[faiss_id] = img_score

    #######################
    #      OCR SEARCH     #
    #######################

    session = get_db_session()
    user_images = session.query(ImageTable).filter(ImageTable.user_id == user_id).all()

    # Apply datetime boost
    all_results = datetime_match_boost(all_results, user_images, datetime_ranges)

    # Apply OCR scoring
    for image in user_images:
        ocr_score = token_set_ratio(query.lower(), image.text.lower()) / 100.0
        if image.faiss_id in all_results:
            all_results[image.faiss_id] += Config.OCR_WEIGHT * ocr_score
        else:
            all_results[image.faiss_id] = ocr_score

    # Sorting the result
    print(json.dumps(all_results, indent=4))
    final_result = sorted(all_results.items(), key=lambda x: x[1], reverse=True)

    distances = [float(score) for _, score in final_result]
    indices = [int(faiss_id) for faiss_id, _ in final_result]

    return jsonify({"status": "ok", "distances": distances, "indices": indices}), 200
