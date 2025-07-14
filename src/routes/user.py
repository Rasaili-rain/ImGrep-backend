from flask import Blueprint, jsonify, Response
from uuid import uuid4, UUID as UUIDType
from sqlalchemy import create_engine, Column, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from typing import Tuple
from config import Config
from utils.logger import logger

user_bp: Blueprint = Blueprint('user', __name__)

# SQLAlchemy setup
Base = declarative_base()

class User(Base):
    __tablename__: str = 'users'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

def get_db_session() -> Session:
    try:
        # Create engine with Neon
        engine = create_engine(Config.DATABASE_URL, echo=Config.DEBUG)
        # Create tables if they don't exist
        Base.metadata.create_all(engine)
        # Create session
        Session = sessionmaker(bind=engine)
        return Session()
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

@user_bp.route('/user/new', methods=['POST'])
def create_user() -> Tuple[Response, int]:
    try:
        session: Session = get_db_session()
        
        
        # Create new user
        user_id: UUIDType = uuid4()
        new_user: User = User(id=user_id)
        session.add(new_user)
        session.commit()
        
        session.close()
        
        return jsonify({
            "status": "success",
            "user_id": str(user_id),
            "message": "User created successfully"
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to create user"
        }), 500