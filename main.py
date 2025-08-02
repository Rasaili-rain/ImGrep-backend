import os
from flask import Flask, Response, jsonify
from sqlalchemy import create_engine


# Models
from src.imgrep.imgrep import ImGrep
from src.imgrep.ocr.ocr import OCR
from src.imgrep.caption_model.captioner import ImageCaptioner

# Routes
from src.routes.image_upload import image_upload_bp
from src.routes.search import search_bp
from src.routes.user import user_bp

# Extras
from src.config import Config
from src.utils.logger import logger


os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

app: Flask = Flask(__name__)
app.config.from_object(Config)

# Check Neon Postgres database connection
def check_db_connection() -> None:
    try:
        engine = create_engine(Config.DATABASE_URL)
        with engine.connect():
            logger.info("Remote db : Neon Postgres database connected")
    except Exception as e:
        logger.error(f"Remote db : Neon Postgres connection failed: {str(e)}")
        raise


# Loading the models
logger.info("Loading ImGrep Model")
# NOTE(slok): Download these files from drive (pinned in discord)
app.imgrep = ImGrep(
    "assets/vocabs.json", "assets/best_model.pt",
    "assets/ocr_weights.pth", "assets/craft_mlt_25k.pth",
    "assets/captioner_weights.pth.tar", "assets/vocab.pkl"
) # type: ignore
logger.info("Loaded ImGrep Model")

# Register blueprints
app.register_blueprint(image_upload_bp, url_prefix="/api")
app.register_blueprint(user_bp, url_prefix="/api")
app.register_blueprint(search_bp, url_prefix="/api")


@app.route("/test")
def hello_world() -> str:
    return "hello world"


# Global error handler
# NOTE(slok): Removing error handler cuz it makes harder to debug
# @app.errorhandler(Exception)
# def handle_error(error: Exception) -> tuple[Response, int]:
#     logger.error(f"Unexpected error: {str(error)}")
#     return jsonify({"status": "error", "message": "An unexpected error occurred"}), 500


# Run via: uv run main.py
if __name__ == "__main__":
    logger.debug("Starting Imgrep backend")
    check_db_connection()
    if Config.DEBUG:
        logger.info(f"Running Imgrep Backend in {Config.SERVER_IP}:{Config.PORT} with hot reloading")
    app.run(debug=Config.DEBUG, host=Config.SERVER_IP, port=Config.PORT)
