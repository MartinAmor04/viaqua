# db_functions.py
import mysql.connector
import os

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

def get_data():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"SELECT machine_id, public_id, date_time, machine_type, audio_record,  place, power,  alert_type FROM alert JOIN machine ON alert.machine_id=machine_id;")
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
print(get_data())