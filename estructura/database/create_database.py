import mysql.connector
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
MYSQL_NAME = os.getenv("DB_NAME")

# Conectar a la base de datos
conn = mysql.connector.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_NAME
)
cursor = conn.cursor()

# Primero eliminar datos si existen (para evitar duplicados)
cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = %s AND table_name = 'alert'", (MYSQL_NAME,))
if cursor.fetchone()[0] > 0:
    cursor.execute("DELETE FROM alert")
    cursor.execute("DELETE FROM machine")
    conn.commit()

# Crear estructura e insertar datos
script_files = ["db_structure.sql", "data_insertion.sql"]

for script_file in script_files:
    with open(script_file, "r") as file:
        sql_script = file.read()
        for statement in sql_script.split(";"):
            if statement.strip():
                try:
                    cursor.execute(statement)
                except Exception as e:
                    print(f"⚠️ Error ejecutando sentencia: {statement.strip()[:100]}...\n{e}")

conn.commit()
cursor.close()
conn.close()

print("✅ Base de datos creada correctamente.")
