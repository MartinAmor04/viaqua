import mysql.connector
import pandas as pd
from settings import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_NAME

def get_connection():
    """Establece conexión con la base de datos."""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_NAME
        )
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Error de conexión a la base de datos: {err}")
        return None

def get_alerts(estado_filtro="Pendiente"):
    """Obtiene alertas filtradas por estado y las devuelve en formato DataFrame."""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()  # Retorna un DataFrame vacío si no hay conexión

    try:
        with conn.cursor(dictionary=True) as cursor:
            query = """
                SELECT 
                    alert.id AS ID,
                    machine.id AS machine_id,
                    machine.public_id AS Máquina,
                    machine.machine_type AS Tipo,
                    alert.date_time AS Fecha_hora,
                    machine.place AS Ubicación,
                    alert.alert_type AS Tipo_avería,
                    alert.estado AS Estado,
                    alert.audio_record AS Audio
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

            query += filtros.get(estado_filtro, filtros["Pendiente"])
            cursor.execute(query)
            alertas = cursor.fetchall()

            if not alertas:
                print(f"⚠️ No se encontraron alertas con estado '{estado_filtro}'.")

            return alertas

    except mysql.connector.Error as err:
        print(f"❌ Error en la consulta SQL: {err}")
        return pd.DataFrame()
    finally:
        conn.close()

def edit_alert(alert_id, new_alert_type, new_estado):
    """Edita el tipo de avería y el estado de una alerta específica."""
    conn = get_connection()
    if conn is None:
        return False

    try:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE alert SET alert_type = %s, estado = %s WHERE id = %s;", 
                           (new_alert_type, new_estado, alert_id))
            conn.commit()
            return True
    except mysql.connector.Error as err:
        print(f"❌ Error al editar la alerta: {err}")
        return False
    finally:
        conn.close()

def add_alert(machine_id, encoded_audio):
    """Inserta una nueva alerta con audio procesado."""
    conn = get_connection()
    if conn is None:
        return False

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO alert (machine_id, date_time, alert_type, audio_record) VALUES (%s, NOW(), %s, %s);", 
                (machine_id, 'No clasificado', encoded_audio)
            )
            conn.commit()
            return True
    except mysql.connector.Error as err:
        print(f"❌ Error al agregar la alerta: {err}")
        return False
    finally:
        conn.close()
