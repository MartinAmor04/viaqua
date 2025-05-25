import mysql.connector
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
MYSQL_NAME = os.getenv("DB_NAME")

# Conectar a la base de datos
conexion = mysql.connector.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_NAME
)
cursor = conexion.cursor()

# Leer contenido del archivo Base64
text = "../pub_sub/encoded.txt"
with open(text, "r") as audio_as_text:
    base64_audio = audio_as_text.read().strip()  # quitar saltos de línea si hay

# Verificar que machine_id = 1 existe
cursor.execute("SELECT COUNT(*) FROM machine WHERE id = 1")
if cursor.fetchone()[0] == 0:
    raise ValueError("❌ Error: No hay máquinas con ID = 1. Inserta las máquinas primero.")

# Insertar alertas
query = "INSERT INTO alert (machine_id, date_time, alert_type, audio_record, estado) VALUES (%s, NOW(), %s, %s, %s)"
valores = [
    (1, "No clasificado", base64_audio, "Pendiente"),
    (1, "Fallo eléctrico", base64_audio, "Pendiente")
]

cursor.executemany(query, valores)
conexion.commit()

print("✅ Alertas insertadas correctamente.")

cursor.close()
conexion.close()

