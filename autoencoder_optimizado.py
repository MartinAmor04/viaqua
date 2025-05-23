import argparse
import numpy as np
import librosa
import wave
import time
import subprocess
import io
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN ---
N_MELS = 128
FIXED_FRAMES = 128
DURATION = 3.0
SAMPLING_RATE = 48000

TRAIN_SAMPLES       = 100      # audios para entrenamiento
CALIBRATION_SAMPLES = 20       # audios sanos para calibrar
MODEL_CHECKPOINT    = 'ae_best.keras'
TFLITE_MODEL_PATH   = 'autoencoder_model.tflite'

# Margen para alerta temprana (porcentaje del rango inicial)
ALERT_MARGIN = 0.1  # 10%

# --- PREPROCESADO ---
def preprocess_signal(signal):
    s = signal.astype(np.float32)
    s /= (np.max(np.abs(s)) + 1e-6)
    S = librosa.feature.melspectrogram(y=s, sr=SAMPLING_RATE, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)
    if S_dB.shape[1] < FIXED_FRAMES:
        pad = FIXED_FRAMES - S_dB.shape[1]
        S_dB = np.pad(S_dB, ((0,0),(0,pad)), mode='constant')
    else:
        S_dB = S_dB[:, :FIXED_FRAMES]
    mn, mx = S_dB.min(), S_dB.max()
    norm = (S_dB - mn) / (mx - mn + 1e-6)
    return norm.flatten().astype(np.float32)

# --- AUDIO ---
def record_audio():
    cmd = [
        'arecord','-D','plughw:1','-c1','-r',str(SAMPLING_RATE),
        '-f','S32_LE','-t','wav','-d',str(int(DURATION)),'-q'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw = proc.stdout.read()
    proc.wait()
    wav = wave.open(io.BytesIO(raw), 'rb')
    frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16)

# --- MODELO ---
def autoencoder_model(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inp)
    x = Dense(32, activation='relu')(x)
    bottleneck = Dense(16, activation='relu')(x)
    x = Dense(32, activation='relu')(bottleneck)
    x = Dense(64, activation='relu')(x)
    dec = Dense(input_dim, activation='sigmoid')(x)
    model = Model(inp, dec)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def convertir_a_tflite(model, path):
    model.save(MODEL_CHECKPOINT)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print(f'[INFO] Modelo convertido a {path}')

# --- FLUJO PRINCIPAL ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    input_dim = N_MELS * FIXED_FRAMES

    # 1) Recoger TRAIN_SAMPLES audios sanos para entrenamiento
    print(f'[INFO] Recopilando {TRAIN_SAMPLES} audios sanos para entrenamiento...')
    X = []
    for i in range(TRAIN_SAMPLES):
        sig = record_audio()
        X.append(preprocess_signal(sig))
        print(f'  - {i+1}/{TRAIN_SAMPLES}')
    X = np.array(X)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)

    # 2) Definir y entrenar el autoencoder (único paso)
    model = autoencoder_model(input_dim)
    cb_ckpt = ModelCheckpoint(MODEL_CHECKPOINT, monitor='val_loss', save_best_only=True, verbose=1)
    cb_es   = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    cb_rlr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)

    print('[INFO] Entrenando autoencoder…')
    model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=100,
        batch_size=args.batch_size,
        callbacks=[cb_ckpt, cb_es, cb_rlr],
        verbose=2
    )

    # Cargamos mejor peso y convertimos a TFLite
    model.load_weights(MODEL_CHECKPOINT)
    convertir_a_tflite(model, TFLITE_MODEL_PATH)

    # 3) Calibración inicial con audios sanos
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    inp_d, out_d = interpreter.get_input_details()[0], interpreter.get_output_details()[0]

    errs = []
    print(f'[INFO] Calibrando con {CALIBRATION_SAMPLES} audios sanos…')
    for _ in range(CALIBRATION_SAMPLES):
        sig = record_audio()
        Xc  = preprocess_signal(sig)[None, :].astype(np.float32)
        interpreter.set_tensor(inp_d['index'], Xc)
        interpreter.invoke()
        recon = interpreter.get_tensor(out_d['index'])
        errs.append(np.mean((Xc - recon)**2))

    err_min = float(np.mean(errs))                     # error típico sano
    err_max = float(np.percentile(errs, 99))            # límite superior sano
    anom_thr = err_min + ALERT_MARGIN * (err_max - err_min)  # alerta temprana
    print(f'[INFO] err_min={err_min:.6f}, err_max={err_max:.6f}, anom_thr={anom_thr:.6f}')

    # 4) Monitorización continua
    print('[INFO] Monitorización iniciada… (Ctrl+C para parar)')
    def clasificar(p):
        if p == 0:      return "Totalmente sano"
        if p <= 25:     return "Ligeramente dañado"
        if p <= 50:     return "Parcialmente dañado"
        if p <= 75:     return "Muy dañado"
        return "Roto"

    while True:
        sig = record_audio()
        Xc  = preprocess_signal(sig)[None, :].astype(np.float32)
        interpreter.set_tensor(inp_d['index'], Xc)
        interpreter.invoke()
        recon = interpreter.get_tensor(out_d['index'])
        err = np.mean((Xc - recon)**2)

        # Actualizamos extremos de forma monotónica
        err_min = min(err_min, err)
        if err > err_max:
            err_max = err
            print(f'[INFO] Nuevo err_max histórico: {err_max:.6f}')

        # Cálculo de porcentaje
        denom = max(err_max - err_min, 1e-6)
        pct = float(np.clip((err - err_min) / denom * 100, 0, 100))

        estado = clasificar(pct)
        print(f'Error: {err:.6f} | Daño: {pct:6.2f}% | Estado: {estado}')
        if err > anom_thr and pct < 100:
            print('[ALERTA] Desviación leve – revisar pronto.')

        time.sleep(5)

if __name__ == '__main__':
    main()
