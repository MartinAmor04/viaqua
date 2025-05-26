import psycopg2
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

PG_HOST = os.getenv("DB_HOST")
PG_USER = os.getenv("DB_USER")
PG_PASSWORD = os.getenv("DB_PASSWORD")
PG_NAME = os.getenv("DB_NAME")

def get_connection():
    print('HOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOLA', PG_HOST)

    return psycopg2.connect(
        host=PG_HOST,
        user=PG_USER,
        port=5432,
        password=PG_PASSWORD,
        dbname=PG_NAME
    )

def get_alerts():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            alert.id AS id, 
            machine_id, 
            public_id, 
            date_time, 
            machine_type, 
            audio_record,  
            place, 
            power,  
            alert_type 
        FROM alert 
        JOIN machine ON alert.machine_id = machine.machine_id;
    """)
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
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
    cursor.execute("""
        INSERT INTO alert (machine_id, date_time, alert_type, audio_record) 
        VALUES (%s, NOW(), %s, %s);
    """, (machine_id, 'No clasificado', encoded_audio))
    conn.commit()
    cursor.close()
    conn.close()
