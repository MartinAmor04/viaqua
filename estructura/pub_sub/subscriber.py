import paho.mqtt.client as mqtt
import ssl
import time
from email.mime.text import MIMEText
import smtplib
from dotenv import load_dotenv
import os
import json

import sys
sys.path.append('../ui_curuxia/modules')
from sql_queries import add_alert, get_machine


# === VARIABLES DE ENTORNO ===
load_dotenv() 
HIVE_USER = os.getenv("MQ_HIVE_USER")
HIVE_PASSWORD=os.getenv("MQ_HIVE_PASSWORD")
HIVE_BROKER=os.getenv("MQ_HIVE_BROKER")

EMAIL_PASSWORD=os.getenv("EMAIL_PASSWORD")
EMAIL_SENDER=os.getenv("EMAIL_SENDER")
EMAIL_RECEIVER=os.getenv("EMAIL_RECEIVER")

# === CONFIGURACIÓN DE HIVEMQ CLOUD ===
MQTT_BROKER = HIVE_BROKER
MQTT_PORT = 8883
MQTT_TOPIC = "audio/alerts"
MQTT_USER = HIVE_USER
MQTT_PASSWORD = HIVE_PASSWORD

# === CALLBACKS ===
def on_connect(client, userdata, flags, rc):
    print(f"Conectado al broker con código: {rc}")
    client.subscribe(MQTT_TOPIC)
    print(f"Suscrito al topic: {MQTT_TOPIC}")

def on_message(client, userdata, msg):
    mensaje = msg.payload.decode()
    mensaje_dict = json.loads(mensaje)
    machine_id = mensaje_dict.get("machine_id")
    audio_string = mensaje_dict.get("audio_record")
    machine_data=get_machine(machine_id)
    print(machine_data)
    print(f"Mensaje recibido en {msg.topic}")
    msg = f"""
    Hola,
    CuruxIA ha detectado un fallo.
    * ID MÁQUINA: {machine_data[0]}
    * TIPO: {machine_data[3]}
    * LUGAR: {machine_data[2]}
    Para más información consulta tu CuruxIA APP-
    """ 
    send_email("CuruxIA: fallo en máquina.", msg)
    add_alert(machine_id, audio_string)

# === SEND MESSAGE FUNCTION ===
def send_email(subject, body):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = EMAIL_SENDER
    receiver_email = EMAIL_RECEIVER
    password = EMAIL_PASSWORD

    msg = MIMEText(body, "plain") 
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
    
# === CLIENTE MQTT ===
client = mqtt.Client()
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.tls_set()  # TLS por defecto

client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT)
client.loop_start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    client.loop_stop()
