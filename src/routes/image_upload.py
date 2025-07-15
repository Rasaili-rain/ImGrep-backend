import threading
import time
from typing import Any, Dict, OrderedDict
from flask import Blueprint, request, jsonify, Response
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import os
import uuid
from src.utils.logger import logger

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
    