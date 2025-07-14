from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid

image_upload_bp = Blueprint('image_upload', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@image_upload_bp.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400

    file = request.files['image']

    if not file or not file.filename or not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid or no file selected'}), 400

    upload_folder = "uploads/"
    os.makedirs(upload_folder, exist_ok=True)

    filename = secure_filename(file.filename)
    extension = filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4()}.{extension}"
    file_path = os.path.join(upload_folder, unique_filename)
    file.save(file_path)

    return jsonify({
        'status': 'success',
        'message': f'Image received successfully to {file_path}',
        'filename': unique_filename
    }), 200