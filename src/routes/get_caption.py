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

get_caption = Blueprint('get_caption', __name__)


@get_caption.route('/get-caption', methods=['GET'])
def caption():
    payload = request.get_json()
    faiss_id = payload.get('faiss_id')

    session = get_db_session()
    caption = session.query(ImageTable).filter(ImageTable.faiss_id == faiss_id).one()
    
    return jsonify({"caption": caption.description})

