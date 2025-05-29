import paho.mqtt.client as mqtt
import ssl
import time
import json
import os
import base64
import io
import joblib
import librosa
import numpy as np
from email.mime.text import MIMEText
import smtplib
from dotenv import load_dotenv

import sys
# Ajusta esta ruta según tu estructura real. Aquí asumo que estás ejecutando
# "subscriber_with_prediction.py" desde una carpeta junto a "ui_curuxia/"
sys.path.append(os.path.join(os.path.dirname(__file__), "../ui_curuxia/modules"))
from sql_queries import add_alert, get_machine

# ============================
# CARGA DE VARIABLES DE ENTORNO
# ============================
load_dotenv()
HIVE_USER     = os.getenv("MQ_HIVE_USER")
HIVE_PASSWORD = os.getenv("MQ_HIVE_PASSWORD")
HIVE_BROKER   = os.getenv("MQ_HIVE_BROKER")

EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_SENDER   = os.getenv("EMAIL_SENDER")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# ============================
# PARÁMETROS MQTT
# ============================
MQTT_BROKER   = HIVE_BROKER
MQTT_PORT     = 8883
MQTT_TOPIC    = "audio/alerts"
MQTT_USER     = HIVE_USER
MQTT_PASSWORD = HIVE_PASSWORD

# ============================
# PARÁMETROS PARA CLASIFICACIÓN
# ============================

# Ruta al modelo previamente entrenado y guardado con joblib
VOTING_MODEL_PATH = "voting_classifier_model.joblib"

# Número de coeficientes MFCC que extrajimos en el entrenamiento
N_MFCC = 13

# ============================
# UTILIDADES DE AUDIO
# ============================

def extract_features_from_bytes(audio_bytes: bytes, sr_target: int = None, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Decodifica los bytes WAV en memoria y extrae MFCCs:
    - Carga con librosa desde un BytesIO
    - Usa n_mfcc coeficientes
    - Devuelve un vector float32 de longitud 2 * n_mfcc (media y desviación de cada coeficiente)
    """
    try:
        # librosa.load acepta un file-like object si se pasa BytesIO
        wav_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(wav_buffer, sr=sr_target)  # sr_target=None para conservar tasa original
        # Aseguramos que la señal no esté vacía
        if y is None or len(y) == 0:
            raise ValueError("La señal cargada está vacía.")

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        features = np.concatenate([mfccs_mean, mfccs_std]).astype(np.float32)
        return features
    except Exception as e:
        print(f"[ERROR] Al extraer MFCCs: {e}")
        # Devolvemos un vector de ceros por si acaso (dim = 2 * n_mfcc)
        return np.zeros(2 * n_mfcc, dtype=np.float32)

# ============================
# CARGA DEL MODELO DE CLASIFICACIÓN
# ============================
try:
    clf = joblib.load(VOTING_MODEL_PATH)
    print(f"[INFO] Modelo cargado: {VOTING_MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo de clasificación: {e}")
    clf = None  # Lo marcamos como None para detectar más abajo

# ============================
# FUNCIONES PARA ENVÍO DE CORREO
# ============================
def send_email(subject: str, body: str):
    """
    Envía un correo con subject y body usando credenciales de Gmail (STARTTLS).
    """
    smtp_server = "smtp.gmail.com"
    smtp_port   = 587
    sender = EMAIL_SENDER
    receiver = EMAIL_RECEIVER
    password = EMAIL_PASSWORD

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"[INFO] Email enviado a {receiver}")
    except Exception as e:
        print(f"[ERROR] No se pudo enviar el correo: {e}")

# ============================
# CALLBACKS MQTT
# ============================
def on_connect(client, userdata, flags, rc):
    print(f"[INFO] Conectado al broker MQTT con código {rc}")
    client.subscribe(MQTT_TOPIC)
    print(f"[INFO] Suscrito al tópico: {MQTT_TOPIC}")

def on_message(client, userdata, msg):
    """
    Cada vez que llegue un mensaje al tópico 'audio/alerts', se ejecuta:
     - Decodificar JSON
     - Obtener machine_id y audio_record
     - Obtener datos de la máquina desde BD
     - Predecir tipo de avería (si el modelo está cargado)
     - Enviar email con la información + predicción
     - Guardar la alerta en BD con add_alert(...)
    """
    try:
        payload_str = msg.payload.decode("utf-8")
        data = json.loads(payload_str)
    except Exception as e:
        print(f"[ERROR] No se pudo decodificar JSON de MQTT: {e}")
        return

    machine_id   = data.get("machine_id")
    audio_b64    = data.get("audio_record", "")
    if machine_id is None or audio_b64 == "":
        print("[WARNING] Falta 'machine_id' o 'audio_record' en mensaje")
        return

    # 1) Obtener info de la máquina (ID, public_id, place, machine_type, etc.)
    machine_data = get_machine(machine_id)
    if machine_data is None:
        print(f"[WARNING] No se encontró la máquina con ID {machine_id}")
        # Puedes optar por igual guardar la alerta sin predicción
        machine_info_str = f"ID: {machine_id} (no hallada en BD)"
    else:
        # Asumimos que get_machine devuelve algo tipo (id, public_id, place, machine_type, ...)
        # Ajusta según tu schema real
        _, public_id, place, machine_type = machine_data[0], machine_data[1], machine_data[2], machine_data[3]
        machine_info_str = f"ID interna: {machine_id}\nPublic ID: {public_id}\nLugar: {place}\nTipo de máquina: {machine_type}"

    # 2) Decodificar Base64 a bytes brutos de WAV
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        print(f"[ERROR] No se pudo decodificar Base64: {e}")
        return

    # 3) Predecir el tipo de avería si el modelo está disponible
    predicted_label = "desconocido"
    confidence_str  = ""
    if clf is not None:
        feats = extract_features_from_bytes(audio_bytes, sr_target=None, n_mfcc=N_MFCC)
        # El clasificador espera un array 2D: shape (1, num_features)
        try:
            pred = clf.predict(feats.reshape(1, -1))[0]
            predicted_label = str(pred)
            # Si el clasificador soporta predict_proba, calculamos confianza
            if hasattr(clf, "predict_proba"):
                proba = np.max(clf.predict_proba(feats.reshape(1, -1)))
                confidence_str = f" (confianza ≈ {proba:.2f})"
            print(f"[INFO] Predicción de avería: {predicted_label}{confidence_str}")
        except Exception as e:
            print(f"[ERROR] Falló predict(): {e}")
    else:
        print("[WARNING] El modelo de clasificación no está cargado. Se omite predicción.")

    # 4) Construir el cuerpo del email
    email_body = f"""
Hola,

CuruxIA ha detectado un fallo en la máquina:

{machine_info_str}

Tipo de avería predicho: {predicted_label}{confidence_str}

Para más información, consulta tu CuruxIA APP.
"""
    send_email("CuruxIA: fallo en máquina detectado", email_body)

    # 5) Guardar la alerta en la base de datos (add_alert almacena machine_id + audio_record)
    #    Podrías modificar add_alert para que reciba también predicted_label si deseas registrarlo en BD.
    add_alert(machine_id, audio_b64)
    print(f"[INFO] Alerta guardada en BD para machine_id={machine_id}")

# ============================
# INICIALIZACIÓN DEL CLIENTE MQTT
# ============================
client = mqtt.Client()
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.tls_set(cert_reqs=ssl.CERT_REQUIRED)

client.on_connect = on_connect
client.on_message = on_message

print(f"[INFO] Conectando a {MQTT_BROKER}:{MQTT_PORT} …")
client.connect(MQTT_BROKER, MQTT_PORT)
client.loop_start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[INFO] Detenido por Ctrl+C")
    client.loop_stop()
