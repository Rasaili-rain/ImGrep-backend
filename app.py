import os
from flask import Flask
from dotenv import load_dotenv

from src.models import *
from src.apitest import bp as apitest_bp
from src.api.feed import bp as feed_bp
from src.api.init import bp as init_bp

# Loading the .env file
load_dotenv()

app = Flask(__name__)

# Configure DB
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Init extensions
db.init_app(app)

# Register Blueprints
app.register_blueprint(apitest_bp)
app.register_blueprint(feed_bp)
app.register_blueprint(init_bp)

# Create DB tables
with app.app_context():
  db.create_all()
