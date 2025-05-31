import mysql.connector
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
MYSQL_NAME = os.getenv("DB_NAME")

conexion = mysql.connector.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_NAME
)
cursor = conexion.cursor()

text = "audio_record.txt"
with open(text, "r") as audio_as_text:
    base64_audio = audio_as_text.read().strip()  # quitar saltos de línea si hay

cursor.execute("SELECT COUNT(*) FROM machine WHERE id = 2")
if cursor.fetchone()[0] == 0:
    raise ValueError("❌ Error: No hay máquinas con ID = 2. Inserta las máquinas primero.")

query = "INSERT INTO alert (machine_id, date_time, alert_type, audio_record, estado) VALUES (%s, %s, %s, %s, %s)"
valores = [
    (1, "2025-01-07 21:12:37", "Rodamientos", base64_audio, "Pendiente"),
    (1, "2025-01-06 18:32:51", "Rodamientos", base64_audio, "Arreglada"),
    (3, "2025-01-27 20:14:38", "Rodamientos", base64_audio, "Pendiente"),
    (3, "2025-01-22 14:17:46", "Fallo de Fase", base64_audio, "Pendiente"),
    (7, "2025-01-10 08:43:02", "Sobrecalentamiento", base64_audio, "En revisión"),
    (9, "2025-01-02 22:32:39", "Sobrecalentamiento", base64_audio, "En revisión"),
    (3, "2025-01-15 19:56:03", "Fallo de Fase", base64_audio, "En revisión"),
    (9, "2025-01-02 23:50:35", "Fallo de Fase", base64_audio, "En revisión"),
    (6, "2025-02-15 00:28:05", "Sobrecalentamiento", base64_audio, "En revisión"),
    (1, "2025-02-09 22:50:54", "Rodamientos", base64_audio, "Pendiente"),
    (5, "2025-02-22 12:26:48", "Fallo mecánico", base64_audio, "En revisión"),
    (7, "2025-02-18 05:50:37", "Fallo eléctrico", base64_audio, "Pendiente"),
    (4, "2025-02-09 23:48:49", "Fallo de Fase", base64_audio, "En revisión"),
    (2, "2025-02-14 07:40:12", "Rodamientos", base64_audio, "Pendiente"),
    (7, "2025-02-12 05:21:05", "Válvula dañada", base64_audio, "En revisión"),
    (5, "2025-02-24 22:00:44", "Sobrecalentamiento", base64_audio, "En revisión"),
    (8, "2025-02-01 09:50:27", "Rodamientos", base64_audio, "Arreglada"),
    (8, "2025-02-28 17:18:01", "Rodamientos", base64_audio, "Pendiente"),
    (6, "2025-03-26 11:13:34", "Sobrecalentamiento", base64_audio, "En revisión"),
    (5, "2025-03-25 14:45:15", "Fallo mecánico", base64_audio, "En revisión"),
    (10, "2025-03-13 04:37:10", "Fallo eléctrico", base64_audio, "En revisión"),
    (3, "2025-03-19 17:00:26", "Rodamientos", base64_audio, "Arreglada"),
    (5, "2025-03-06 04:17:13", "Sobrecalentamiento", base64_audio, "Pendiente"),
    (3, "2025-03-17 23:53:16", "Rodamientos", base64_audio, "Arreglada"),
    (5, "2025-03-08 09:58:55", "Fallo eléctrico", base64_audio, "Arreglada"),
    (10, "2025-03-26 22:01:36", "Fallo mecánico", base64_audio, "En revisión"),
    (10, "2025-03-15 04:58:08", "Sobrecalentamiento", base64_audio, "Pendiente"),
    (1, "2025-03-28 11:32:41", "Fallo de Fase", base64_audio, "En revisión"),
    (8, "2025-04-21 13:33:48", "Sobrecalentamiento", base64_audio, "En revisión"),
    (4, "2025-04-26 17:32:25", "Sobrecalentamiento", base64_audio, "Pendiente"),
    (8, "2025-04-05 11:24:15", "Rodamientos", base64_audio, "Pendiente"),
    (4, "2025-04-10 21:35:50", "Sobrecalentamiento", base64_audio, "Pendiente"),
    (2, "2025-04-15 09:52:20", "Fallo de Fase", base64_audio, "Arreglada"),
    (10, "2025-04-20 00:01:53", "Fallo eléctrico", base64_audio, "En revisión"),
    (6, "2025-04-16 16:44:08", "Fallo mecánico", base64_audio, "En revisión"),
    (7, "2025-04-06 15:37:39", "Fallo eléctrico", base64_audio, "Pendiente"),
    (9, "2025-04-01 04:22:01", "Rodamientos", base64_audio, "Pendiente"),
    (3, "2025-04-18 10:41:04", "Fallo de Fase", base64_audio, "Arreglada"),
    (1, "2025-04-28 11:32:48", "Sobrecalentamiento", base64_audio, "Arreglada"),
    (8, "2025-04-28 23:16:29", "Fallo de Fase", base64_audio, "Arreglada"),
    (8, "2025-04-15 09:43:53", "Fallo de Fase", base64_audio, "Pendiente"),
    (5, "2025-04-24 10:45:31", "Fallo mecánico", base64_audio, "Arreglada"),
    (8, "2025-04-10 13:17:47", "Fallo de Fase", base64_audio, "Arreglada"),
    (1, "2025-05-26 08:46:11", "Sobrecalentamiento", base64_audio, "Pendiente"),
    (6, "2025-05-26 14:24:27", "Válvula dañada", base64_audio, "Pendiente"),
    (2, "2025-05-01 00:49:50", "Sobrecalentamiento", base64_audio, "Pendiente"),
    (3, "2025-05-16 19:07:33", "Sobrecalentamiento", base64_audio, "Arreglada"),
    (9, "2025-05-19 07:24:22", "Sobrecalentamiento", base64_audio, "Pendiente"),
    (8, "2025-05-11 03:14:10", "Rodamientos", base64_audio, "En revisión"),
    (5, "2025-05-07 15:15:23", "Fallo mecánico", base64_audio, "Pendiente"),
    (3, "2025-05-26 04:26:36", "Rodamientos", base64_audio, "En revisión"),
    (7, "2025-05-05 15:28:51", "Sobrecalentamiento", base64_audio, "Pendiente"),
    (3, "2025-05-05 09:41:33", "Rodamientos", base64_audio, "Arreglada"),
    (3, "2025-05-26 04:17:50", "Fallo de Fase", base64_audio, "Pendiente"),
    (9, "2025-05-19 23:24:55", "Sobrecalentamiento", base64_audio, "En revisión"),
    (7, "2025-05-20 09:33:55", "Válvula dañada", base64_audio, "En revisión"),
    (9, "2025-05-10 18:52:23", "Rodamientos", base64_audio, "Pendiente"),
    (7, "2025-05-02 16:11:59", "Sobrecalentamiento", base64_audio, "Pendiente"),
    (1, "2025-05-13 11:31:54", "Fallo de Fase", base64_audio, "Pendiente"),
    (2, "2025-05-20 07:01:56", "Fallo de Fase", base64_audio, "Pendiente"),
    (10, "2025-05-07 17:04:53", "Sobrecalentamiento", base64_audio, "En revisión"),
    (3, "2025-05-06 09:38:16", "Rodamientos", base64_audio, "Arreglada"),
    (10, "2025-05-27 05:28:48", "Sobrecalentamiento", base64_audio, "Pendiente"),
    (8, "2025-06-21 19:39:56", "Rodamientos", base64_audio, "Arreglada"),
    (10, "2025-06-08 04:18:05", "Sobrecalentamiento", base64_audio, "Pendiente"),
    (6, "2025-06-13 03:20:13", "Fallo mecánico", base64_audio, "En revisión"),
    (7, "2025-06-10 12:33:37", "Fallo eléctrico", base64_audio, "Pendiente"),
    (4, "2025-06-20 18:41:23", "Fallo de Fase", base64_audio, "En revisión"),
    (2, "2025-06-02 05:08:52", "Rodamientos", base64_audio, "Arreglada"),
    (6, "2025-06-24 07:42:58", "Fallo eléctrico", base64_audio, "En revisión"),
    (4, "2025-06-16 10:47:49", "Sobrecalentamiento", base64_audio, "En revisión"),
    (5, "2025-06-11 20:35:02", "Fallo eléctrico", base64_audio, "Pendiente"),
    (8, "2025-06-16 11:21:11", "Rodamientos", base64_audio, "Arreglada"),
    (8, "2025-06-20 04:38:30", "Rodamientos", base64_audio, "Pendiente"),
    (9, "2025-06-26 14:05:23", "Sobrecalentamiento", base64_audio, "Pendiente"),
    (7, "2025-06-26 11:52:05", "Fallo eléctrico", base64_audio, "Arreglada"),
    (4, "2025-06-26 21:43:20", "Fallo de Fase", base64_audio, "En revisión"),
    (3, "2025-06-19 04:31:49", "Fallo de Fase", base64_audio, "Pendiente"),
    (1, "2025-06-04 03:02:17", "Sobrecalentamiento", base64_audio, "En revisión"),
    (3, "2025-06-21 13:43:30", "Fallo de Fase", base64_audio, "Arreglada"),
]

# Insertar en lotes de 10
batch_size = 10
for i in range(0, len(valores), batch_size):
    cursor.executemany(query, valores[i:i+batch_size])
    conexion.commit()

print("✅ Alertas insertadas correctamente por lotes.")

cursor.close()
conexion.close()