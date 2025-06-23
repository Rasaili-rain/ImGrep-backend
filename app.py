from flask import Flask
from src.models import db
from src.apitest import bp as apitest_bp
import os

app = Flask(__name__)

# Configure DB
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Init extensions
db.init_app(app)

# Register Blueprints
app.register_blueprint(apitest_bp)

# Create DB tables
with app.app_context():
    db.create_all()
