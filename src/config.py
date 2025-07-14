import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG:bool = True
    PORT:int =  5000
    SERVER_IP:str = '0.0.0.0'

    DATABASE_URL: str = os.getenv('DATABASE_URL', 'use the .env please')
    #