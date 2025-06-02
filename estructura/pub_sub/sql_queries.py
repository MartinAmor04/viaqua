# db_functions.py
import mysql.connector
import os
from dotenv import load_dotenv


# Load env variables
load_dotenv() 

MYSQL_HOST= os.getenv("DB_HOST")
MYSQL_USER= os.getenv("DB_USER")
MYSQL_PASSWORD= os.getenv("DB_PASSWORD")
MYSQL_NAME= os.getenv("DB_NAME")


def get_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_NAME
    )

def get_alerts():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"SELECT alert.id as machine.id, alert.machine_id, machine.public_id, alert.date_time, machine.machine_type, alert.audio_record,  machine.place, machine.power,  alert.alert_type FROM alert JOIN machine ON alert.machine_id=machine.id;")
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

def edit_alert(alert_id, new_alert_type):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE alert SET alert_type = %s WHERE id = %s;", (new_alert_type, alert_id))
    conn.commit()
    cursor.close()
    conn.close()

def add_alert(machine_id, encoded_audio):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO alert (machine_id, date_time, alert_type, audio_record) VALUES (%s, NOW(), %s, %s);", (machine_id, 'No clasificado', encoded_audio))
    conn.commit()
    cursor.close()
    conn.close()
    print('Alerta insertada')

def get_machine(machine_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"SELECT id, public_id, place, machine_type FROM machine WHERE id={machine_id};")
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result
