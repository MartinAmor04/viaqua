import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os
import base64
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



def audio_to_base64(audio_path):
    with open(audio_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
    return encoded_audio

def get_machine_id(path='machine.conf'):
    with open(path, 'r') as f:
        for line in f:
            if 'public_id=' in line:
                return line.strip().split('=')[1]
    raise ValueError("No se encontró 'public_id=' en el archivo")

# === CALLBACKS ===
def on_connect(client, userdata, flags, rc):
    print("Conectado al broker con código:", rc)
    audio_string=audio_to_base64('sample-3s.wav')
    machine_id=get_machine_id()
    message = f'{{ "machine_id":"{machine_id}", "audio_record":"{audio_string}" }}'    
    client.publish(MQTT_TOPIC, message)
    print("Mensaje enviado.")
    client.disconnect()

# === CLIENTE MQTT ===
client = mqtt.Client()
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.tls_set() 
client.on_connect = on_connect

client.connect(MQTT_BROKER, MQTT_PORT)
client.loop_forever()