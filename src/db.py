from sqlalchemy import create_engine, Column, UUID, Integer, Text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from uuid import uuid4

from src.config import Config
from src.utils.logger import logger

Base = declarative_base()


def get_db_session() -> Session:
    try:
        engine = create_engine(Config.DATABASE_URL, echo=Config.DEBUG)
        Session = sessionmaker(bind=engine)
        return Session()
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise


##########################
#         MODELS         #
##########################

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)


class Ocr(Base):
    __tablename__ = "ocr"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Text, nullable=False)
    faiss_id = Column(Text, nullable=False)
    text = Column(Text, nullable=False)
