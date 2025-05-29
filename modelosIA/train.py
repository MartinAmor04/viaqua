import argparse
import numpy as np
import librosa
import wave
import subprocess
import io
import os
import logging
import pickle

# Configuración de hilos para TensorFlow
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# ----------------------------- CONFIGURACIÓN GLOBAL -----------------------------

N_MELS = 128
FIXED_FRAMES = 128
DURATION = 3.0  # segundos de grabación
SAMPLING_RATE = 48000  # Hz

# Para que FIXED_FRAMES concuerde con mel-spectrogram, calculamos hop_length:
HOP_LENGTH = int((SAMPLING_RATE * DURATION) / FIXED_FRAMES)

TRAIN_SAMPLES = 100
MODEL_CHECKPOINT = 'ae_best.keras'
TFLITE_MODEL_PATH = 'autoencoder_model.tflite'

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def preprocess_signal(signal):
    """Convierte la señal mono en un vector de características (mel-spectrograma aplanado)."""
    s = signal.astype(np.float32)
    if np.max(np.abs(s)) > 0:
        s /= (np.max(np.abs(s)) + 1e-6)

    S = librosa.feature.melspectrogram(
        y=s,
        sr=SAMPLING_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=2048
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Asegurar FIXED_FRAMES columnas
    if S_dB.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - S_dB.shape[1]
        S_dB = np.pad(S_dB, ((0, 0), (0, pad_width)), mode='constant')
    else:
        S_dB = S_dB[:, :FIXED_FRAMES]

    mn, mx = S_dB.min(), S_dB.max()
    norm = (S_dB - mn) / (mx - mn + 1e-6)
    return norm.flatten().astype(np.float32)


def record_audio():
    """Graba audio usando arecord y devuelve numpy.array float32."""
    cmd = [
        'arecord',
        '-D', 'plughw:1',
        '-c1',
        '-r', str(SAMPLING_RATE),
        '-f', 'S16_LE',
        '-t', 'wav',
        '-d', str(int(DURATION)),
        '-q'
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw, err = proc.communicate(timeout=int(DURATION) + 2)
        if proc.returncode != 0:
            raise RuntimeError(f"arecord error: {err.decode().strip()}")
        wav = wave.open(io.BytesIO(raw), 'rb')
        raw_frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32)
        return audio
    except Exception as e:
        logging.error(f"Falló la grabación de audio: {e}")
        return np.zeros(int(SAMPLING_RATE * DURATION), dtype=np.float32)


def autoencoder_model(input_dim):
    """Define y compila un autoencoder totalmente conectado."""
    inp = Input(shape=(input_dim,))
    x = Dense(224, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(112, activation='relu')(x)
    x = BatchNormalization()(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)

    bottleneck = Dense(24, activation='relu')(x)

    x = Dense(64, activation='relu')(bottleneck)
    x = BatchNormalization()(x)

    x = Dense(112, activation='relu')(x)
    x = BatchNormalization()(x)

    x = Dense(224, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    output = Dense(input_dim, activation='sigmoid')(x)

    model = Model(inp, output)
    lr = 3.205205517146071e-04
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model


def convertir_a_tflite(model, path):
    """Convierte el modelo Keras a TensorFlow Lite con optimizaciones."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    try:
        tflite_model = converter.convert()
        with open(path, 'wb') as f:
            f.write(tflite_model)
        logging.info(f"Modelo convertido y guardado en: {path}")
    except Exception as e:
        logging.error(f"Error al convertir a TFLite: {e}")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de autoencoder y grabación de señales sanas.")
    parser.add_argument('--batch_size', type=int, default=8, help='Tamaño de batch para entrenamiento del autoencoder.')
    args = parser.parse_args()

    input_dim = N_MELS * FIXED_FRAMES

    logging.info(f"Recopilando {TRAIN_SAMPLES} audios sanos para entrenamiento de autoencoder...")

    X = []
    healthy_signals = []

    # 1) Recolección de señales sanas y extracción de mel-spectrogramas
    for i in range(TRAIN_SAMPLES):
        sig = record_audio()
        healthy_signals.append(sig)
        X_proc = preprocess_signal(sig)
        X.append(X_proc)
        rms_val = np.sqrt(np.mean(sig ** 2))
        logging.info(f"  - Muestra {i + 1}/{TRAIN_SAMPLES} (RMS={rms_val:.2f})")

    X = np.array(X, dtype=np.float32)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)

    # 2) Definición y entrenamiento del autoencoder
    model = autoencoder_model(input_dim)
    cb_ckpt = ModelCheckpoint(MODEL_CHECKPOINT, monitor='val_loss', save_best_only=True, verbose=0)
    cb_es = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
    cb_rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=8, min_lr=1e-6, verbose=1)

    logging.info("Entrenando autoencoder...")
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=250,
        batch_size=args.batch_size,
        callbacks=[cb_ckpt, cb_es, cb_rlr],
        verbose=2
    )

    # 3) Cargar pesos óptimos y convertir a TFLite
    model.load_weights(MODEL_CHECKPOINT)
    convertir_a_tflite(model, TFLITE_MODEL_PATH)

    # 4) Guardar señales sanas en disco para la fase de extracción de baseline
    with open('healthy_signals.pkl', 'wb') as f:
        pickle.dump(healthy_signals, f)
    logging.info("Señales sanas guardadas en 'healthy_signals.pkl'")

    logging.info("Entrenamiento del autoencoder completado. Artefactos guardados:\n"
                 f" • {MODEL_CHECKPOINT}\n"
                 f" • {TFLITE_MODEL_PATH}")


if __name__ == '__main__':
    main()
