from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG:bool = True
    PORT:int =  5000
    SERVER_IP:str = '0.0.0.0'