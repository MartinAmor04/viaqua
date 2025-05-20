import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os

# === VARIABLES DE ENTORNO ===
load_dotenv()
HIVE_USER = os.getenv("MQ_HIVE_USER")
HIVE_PASSWORD=os.getenv("MQ_HIVE_PASSWORD")

# === CONFIGURACIÓN DE HIVEMQ CLOUD ===
MQTT_BROKER = "0adbb214459a4128995884f3f492115b.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "audio/alerts"
MQTT_USER = HIVE_USER
MQTT_PASSWORD = HIVE_PASSWORD


MENSAJE = "¡Hola desde el publisher MQTT en HiveMQ!"

# === CALLBACKS ===
def on_connect(client, userdata, flags, rc):
    print("Conectado al broker con código:", rc)
    client.publish(MQTT_TOPIC, MENSAJE)
    print("Mensaje enviado.")
    client.disconnect()

# === CLIENTE MQTT ===
client = mqtt.Client()
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.tls_set()  # TLS por defecto
client.on_connect = on_connect

print("Conectando y enviando mensaje...")
client.connect(MQTT_BROKER, MQTT_PORT)
client.loop_forever()