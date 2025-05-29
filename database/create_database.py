import mysql.connector
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../config/.env"))

MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
MYSQL_NAME = os.getenv("DB_NAME")

# Conectar a MySQL con el usuario que ya tiene permisos
conn = mysql.connector.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD
)
cursor = conn.cursor()

# ✅ Crear base de datos si no existe
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_NAME}")
cursor.execute(f"USE {MYSQL_NAME}")

# ✅ Crear estructura primero
script_files = ["database/db_structure.sql"]
for script_file in script_files:
    with open(script_file, "r") as file:
        sql_script = file.read()
        for statement in sql_script.split(";"):
            if statement.strip():
                try:
                    cursor.execute(statement)
                except mysql.connector.Error as e:
                    print(f"⚠️ Error ejecutando sentencia: {statement.strip()[:100]}...\n{e}")

conn.commit()

# ✅ Insertar datos iniciales
script_files = ["database/data_insertion.sql"]
for script_file in script_files:
    with open(script_file, "r") as file:
        sql_script = file.read()
        for statement in sql_script.split(";"):
            if statement.strip():
                try:
                    cursor.execute(statement)
                except mysql.connector.Error as e:
                    print(f"⚠️ Error ejecutando sentencia: {statement.strip()[:100]}...\n{e}")

conn.commit()
cursor.close()
conn.close()

print("✅ Base de datos creada y configurada correctamente.")
