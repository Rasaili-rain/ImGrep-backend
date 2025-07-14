from flask import Flask, Response, jsonify
from sqlalchemy import create_engine
from src.routes.image_upload import image_upload_bp
from src.routes.user import user_bp
from src.config import Config
from src.utils.logger import logger

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

# Register blueprints
app.register_blueprint(image_upload_bp, url_prefix='/api')
app.register_blueprint(user_bp, url_prefix='/api') 

# Global error handler
@app.errorhandler(Exception)
def handle_error(error: Exception) -> tuple[Response, int]:
    logger.error(f"Unexpected error: {str(error)}")
    return jsonify({"status": "error", "message": "An unexpected error occurred"}), 500

# Run via: uv run src/main.py
if __name__ == '__main__':
    logger.debug("Starting Imgrep backend")
    check_db_connection()  
    if Config.DEBUG:
        logger.info(f"Running Imgrep Backend in {Config.SERVER_IP}:{Config.PORT} with hot reloading")
    app.run(debug=Config.DEBUG, host=Config.SERVER_IP, port=Config.PORT)
