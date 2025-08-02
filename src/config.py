import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DEBUG: bool = True
    PORT: int = 5000
    SERVER_IP: str = "0.0.0.0"

    TOKENIZER_LENGTH = 20
    EMBEDDING_DIM = 256

    FAISS_DATABASE: str = "faiss_db"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "use the .env please")


    ###############################
    #      WEIGHT ADJUSTMENTS     #
    ###############################

    OCR_WEIGHT: float = 0.3
    DATE_TIME_BOOST_AMOUNT : float = 0.5  #needs to be dialed (Slok??)
