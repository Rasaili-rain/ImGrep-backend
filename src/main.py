from flask import Flask
from routes.image_upload import  image_upload_bp
from config import Config
from utils.logger import logger

app = Flask(__name__)
app.config.from_object(Config)

# Log app startup
logger.info("Starting Imgrep backend")

# Register blueprints
app.register_blueprint(image_upload_bp, url_prefix='/api')

# Log errors
@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unexpected error: {str(error)}")
    return {"status": "error", "message": "An unexpected error occurred"}, 500


# run via : uv run src/main.py

if __name__ == '__main__':
    if Config.DEBUG == True:
        logger.debug(f"Running Imgrep Backend in {Config.SERVER_IP}:{Config.PORT} with hot reloading")
    app.run(debug=Config.DEBUG, host=Config.SERVER_IP, port=Config.PORT)