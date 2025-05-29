import os
from dotenv import load_dotenv

# Cargar variables de entorno desde `.env`
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../config/.env"))

MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
MYSQL_NAME = os.getenv("DB_NAME")
