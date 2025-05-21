import argparse
import numpy as np
import librosa
import wave
import subprocess
import io
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURACIÓN ---
N_MELS = 128
FIXED_FRAMES = 128
DURATION = 3.0
SAMPLING_RATE = 48000
BATCH_AUDIOS = 10
WINDOW_SIZE = 100
PERCENTILE = 99
MIN_UPDATES_IN_WINDOW = 5
TFLITE_MODEL_PATH = 'autoencoder_model.tflite'

# --- PREPROCESADO ---
def preprocess_signal(signal):
    signal = signal.astype(np.float16)
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    S = librosa.feature.melspectrogram(y=signal, sr=SAMPLING_RATE, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)

    if S_dB.shape[1] < FIXED_FRAMES:
        pad = FIXED_FRAMES - S_dB.shape[1]
        S_dB = np.pad(S_dB, ((0, 0), (0, pad)), mode='constant')
    else:
        S_dB = S_dB[:, :FIXED_FRAMES]

    mn, mx = S_dB.min(), S_dB.max()
    S_norm = (S_dB - mn) / (mx - mn + 1e-6)
    return S_norm.astype(np.float16)

# --- AUDIO ---
def record_audio():
    print("[INFO] Grabando muestra de audio...")
    cmd = [
        'arecord', '-D', 'plughw:1', '-c1', '-r', str(SAMPLING_RATE),
        '-f', 'S32_LE', '-t', 'wav', '-d', str(int(DURATION)), '-q'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw_audio = proc.stdout.read()
    proc.wait()

    wav_file = io.BytesIO(raw_audio)
    with wave.open(wav_file, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio_np = np.frombuffer(frames, dtype=np.int16)
    return audio_np

# --- MODELO ---
def autoencoder_model(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inp)
    x = Dense(16, activation='relu')(x)
    bottleneck = Dense(8, activation='relu')(x)
    x = Dense(16, activation='relu')(bottleneck)
    x = Dense(64, activation='relu')(x)
    decoded = Dense(input_dim, activation='sigmoid')(x)
    model = Model(inp, decoded)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def convertir_a_tflite(model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print(f'[INFO] Modelo convertido y guardado como {path}')

def clasificar_danio(pct):
    if pct == 0:
        return "Totalmente sano"
    elif pct <= 25:
        return "Ligeramente dañado"
    elif pct <= 50:
        return "Parcialmente dañado"
    elif pct <= 75:
        return "Muy dañado"
    else:
        return "Roto"

# --- FLUJO PRINCIPAL ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.1)
    args = parser.parse_args()

    input_dim = N_MELS * FIXED_FRAMES
    model = autoencoder_model(input_dim)

    if not os.path.exists(TFLITE_MODEL_PATH):
        print('[INFO] Entrenando modelo...')
        while True:
            X_batch = []
            for _ in range(BATCH_AUDIOS):
                sig = record_audio()
                feat = preprocess_signal(sig).flatten()
                X_batch.append(feat)
            X_batch = np.array(X_batch)
            X_train, X_val = train_test_split(X_batch, test_size=0.2, random_state=42)

            early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
            history = model.fit(
                X_train, X_train,
                validation_data=(X_val, X_val),
                epochs=10,
                batch_size=args.batch_size,
                verbose=2,
                callbacks=[early]
            )

            val_loss = min(history.history['val_loss'])
            if val_loss < args.threshold:
                print('[INFO] Umbral alcanzado.')
                break
        convertir_a_tflite(model, TFLITE_MODEL_PATH)

    # --- INFERENCIA TFLITE ---
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    errores_buffer = []
    err_max = 1.0
    print('[INFO] Iniciando inferencia con TFLite...')
    while True:
        sig = record_audio()
        X_test = preprocess_signal(sig).flatten().astype(np.float16)[np.newaxis, :]
        interpreter.set_tensor(input_details[0]['index'], X_test)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        err = np.mean((X_test - output_data) ** 2)

        if err > args.threshold:
            errores_buffer.append(err)
            if len(errores_buffer) > WINDOW_SIZE:
                errores_buffer.pop(0)

        if len(errores_buffer) >= MIN_UPDATES_IN_WINDOW:
            nuevo_err_max = float(np.percentile(errores_buffer, PERCENTILE))
            if abs(nuevo_err_max - err_max) / (err_max + 1e-6) > 0.05:
                err_max = nuevo_err_max
                print(f'[INFO] err_max actualizado por percentil {PERCENTILE}: {err_max:.5f}')

        if err <= args.threshold:
            pct = 0.0
        else:
            denom = max(err_max - args.threshold, 1e-6)
            pct = np.clip((err - args.threshold) / denom, 0, 1) * 100

        estado = clasificar_danio(pct)
        print(f'Error: {err:.5f} | Daño estimado: {pct:6.2f}% | Estado: {estado}')

if __name__ == '__main__':
    main()
