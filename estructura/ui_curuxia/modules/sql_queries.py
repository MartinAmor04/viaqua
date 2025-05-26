# db_functions.py
import mysql.connector
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
MYSQL_NAME = os.getenv("DB_NAME")

def get_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_NAME
    )

def get_alerts(estado_filtro="Activas"):
    """
    Obtiene alertas filtradas por estado.
    - 'Pendiente' → Solo alertas pendientes
    - 'En revisión' → Solo alertas en revisión
    - 'Arreglada' → Solo alertas arregladas
    - 'Activas' (por defecto) → Pendiente + En revisión
    - 'Todas' → Todas las alertas
    """
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cursor:
            query = """
                SELECT 
                    alert.id,
                    machine.id as machine_id,
                    machine.public_id,
                    machine.machine_type,
                    alert.date_time,
                    machine.place,
                    alert.alert_type,
                    alert.estado,
                    alert.audio_record
                FROM alert
                JOIN machine ON alert.machine_id = machine.id
            """
            
            filtros = {
                "Pendiente": "WHERE alert.estado = 'Pendiente'",
                "En revisión": "WHERE alert.estado = 'En revisión'",
                "Arreglada": "WHERE alert.estado = 'Arreglada'",
                "Activas": "WHERE alert.estado IN ('Pendiente', 'En revisión')",
                "Todas": ""  # No aplica filtros
            }

            query += filtros.get(estado_filtro, filtros["Activas"])
            cursor.execute(query)
            return cursor.fetchall()

def edit_alert(alert_id, new_alert_type):
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("UPDATE alert SET alert_type = %s WHERE id = %s;", (new_alert_type, alert_id))
                conn.commit()
    except mysql.connector.Error as err:
        print(f"Error al editar la alerta: {err}")

def add_alert(machine_id, encoded_audio):
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO alert (machine_id, date_time, alert_type, audio_record) VALUES (%s, NOW(), %s, %s);", (machine_id, 'No clasificado', encoded_audio))
                conn.commit()
    except mysql.connector.Error as err:
        print(f"Error al agregar la alerta: {err}")

def get_machine(machine_id):
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM machine WHERE id = %s;", (machine_id,))
                return cursor.fetchone()
    except mysql.connector.Error as err:
        print(f"Error al obtener la máquina: {err}")
