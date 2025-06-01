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
    print("[INFO] Grabando muestra de audio...")
    cmd = [
        'arecord',
        '-D', 'plughw:1',
        '-c1',              # mono
        '-r', str(48000),
        '-f', 'S32_LE',
        '-t', 'wav',
        '-d', str(int(10)),
        '-q'                # silencio en consola
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw_audio = proc.stdout.read()  # Lee toda la salida
    proc.wait()

    # Leer WAV desde bytes
    # wav_file = io.BytesIO(raw_audio)
    # with wave.open(wav_file, 'rb') as wf:
    #     frames = wf.readframes(wf.getnframes())
    #     audio_np = np.frombuffer(frames, dtype=np.int32)

    return raw_audio


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
    audio_string=audio_to_base64(wav_audio)
    send_audio_message(audio_string)

send_alert()