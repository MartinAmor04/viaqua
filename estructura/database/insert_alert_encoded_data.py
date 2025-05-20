import mysql.connector

import os
from dotenv import load_dotenv


# Load env variables
load_dotenv() 

MYSQL_HOST= os.getenv("DB_HOST")
MYSQL_USER= os.getenv("DB_USER")
MYSQL_PASSWORD= os.getenv("DB_PASSWORD")
MYSQL_NAME= os.getenv("DB_NAME")


conexion= mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_NAME)

cursor = conexion.cursor()

# Leer el contenido del archivo Base64
text = "../pub_sub/encoded.txt"
with open(text, "r") as audio_as_text:
    base64_audio = audio_as_text.read()

# Query SQL para insertar
query = "INSERT INTO alert (machine_id, date_time, alert_type, audio_record) VALUES (%s, NOW(), %s, %s)"
valores = [
    (1, "No clasificado", base64_audio),
    (1,"Fallo eléctrico", base64_audio)
]
machine_id
# Ejecutar la consulta
cursor.executemany(query, valores)

# Guardar cambios
conexion.commit()
print("Datos insertados correctamente.")

# Cerrar conexión
cursor.close()
conexion.close()
