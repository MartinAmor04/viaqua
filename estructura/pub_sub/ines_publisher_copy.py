import ssl
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os
import base64
import numpy as np
import wave
import subprocess
import io


# === VARIABLES DE ENTORNO ===
load_dotenv()
HIVE_USER = os.getenv("MQ_HIVE_USER")
HIVE_PASSWORD=os.getenv("MQ_HIVE_PASSWORD")
HIVE_BROKER=os.getenv("MQ_HIVE_BROKER")

# === CONFIGURACIÓN DE HIVEMQ CLOUD ===
MQTT_BROKER = HIVE_BROKER
MQTT_PORT = 8883
MQTT_TOPIC = "audio/alerts"
MQTT_USER = HIVE_USER
MQTT_PASSWORD = HIVE_PASSWORD


def audio_to_base64(audio_file):
    encoded_audio = base64.b64encode(audio_file).decode("utf-8")
    return encoded_audio

def get_machine_id(path='machine.conf'):
    with open(path, 'r') as f:
        for line in f:
            if 'public_id=' in line:
                return line.strip().split('=')[1]
    raise ValueError("No se encontró 'public_id=' en el archivo")

def record_audio():
    return 'audio_probas ines'


def send_audio_message(audio_string):
    # === CALLBACKS ===
    def on_connect(client, userdata, flags, rc):
        print("Conectado al broker con código:", rc)
        machine_id = get_machine_id()
        audio_string = userdata["audio_string"]
        message = f'{{ "machine_id":"{machine_id}", "audio_record":"{audio_string}" }}'
        client.publish(MQTT_TOPIC, message)
        print("Mensaje enviado.")
        client.disconnect()
    
    client = mqtt.Client(userdata={"audio_string": audio_string})
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.on_connect = on_connect

    client.connect(MQTT_BROKER, MQTT_PORT)
    client.loop_forever()

def send_alert():
    wav_audio=record_audio()
    send_audio_message(wav_audio)

send_alert()