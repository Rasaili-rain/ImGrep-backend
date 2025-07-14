from flask import Flask, Response, jsonify
from routes.image_upload import image_upload_bp
from config import Config
from utils.logger import logger

app: Flask = Flask(__name__)
app.config.from_object(Config)

logger.info("Starting Imgrep backend")

# Register blueprints
app.register_blueprint(image_upload_bp, url_prefix='/api')

# Global error handler
@app.errorhandler(Exception)
def handle_error(error: Exception) -> tuple[Response, int]:
    logger.error(f"Unexpected error: {str(error)}")
    return jsonify({"status": "error", "message": "An unexpected error occurred"}), 500

# Run via: uv run src/main.py
if __name__ == '__main__':
    if Config.DEBUG:
        logger.debug(f"Running Imgrep Backend in {Config.SERVER_IP}:{Config.PORT} with hot reloading")
    app.run(debug=Config.DEBUG, host=Config.SERVER_IP, port=Config.PORT)
