import argparse
import numpy as np
import librosa
import wave
import subprocess
import io
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURACIÓN ---
N_MELS = 128                # Resolución en frecuencia (bandas Mel)
FIXED_FRAMES = 128          # Resolución en tiempo (nº de "ventanas" fijas)
DURATION = 3.0              # Duración de cada muestra en segundos
SAMPLING_RATE = 48000       # Frecuencia de muestreo en Hz
BATCH_AUDIOS = 10           # Muestras por lote de entrenamiento
WINDOW_SIZE = 100           # Número de errores recientes que guardamos
PERCENTILE = 95             # Percentil que usaremos para definir err_max
MIN_UPDATES_IN_WINDOW = 5   # Mínimo de valores en buffer para empezar a calcular percentil

# --- PREPROCESADO ---
def preprocess_signal(signal):
    """
    Convierte una señal de audio cruda en un espectrograma Mel normalizado.

    Parameters
    ----------
    signal : np.ndarray
        Señal de audio en formato mono, con valores enteros.

    Returns
    -------
    np.ndarray
        Espectrograma Mel normalizado en el rango [0, 1], con forma (N_MELS, FIXED_FRAMES).
    """
    signal = signal.astype(np.float32)
    signal = signal / np.max(np.abs(signal) + 1e-6)  # Normalización segura
    S = librosa.feature.melspectrogram(y=signal, sr=SAMPLING_RATE, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)

    if S_dB.shape[1] < FIXED_FRAMES:
        pad = FIXED_FRAMES - S_dB.shape[1]
        S_dB = np.pad(S_dB, ((0, 0), (0, pad)), mode='constant')
    else:
        S_dB = S_dB[:, :FIXED_FRAMES]

    mn, mx = S_dB.min(), S_dB.max()
    S_norm = (S_dB - mn) / (mx - mn + 1e-6)
    return S_norm.astype(np.float32)

# --- CAPTURA DE AUDIO ---
def record_audio():
    """
    Graba una muestra de audio en tiempo real usando `arecord` y la convierte a un array NumPy.

    Returns
    -------
    np.ndarray
        Señal de audio capturada como arreglo de enteros de 32 bits.
    """
    print("[INFO] Grabando muestra de audio...")
    cmd = [
        'arecord',
        '-D', 'plughw:2',
        '-c1',
        '-r', str(SAMPLING_RATE),
        '-f', 'S32_LE',
        '-t', 'wav',
        '-d', str(int(DURATION)),
        '-q'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw_audio = proc.stdout.read()
    proc.wait()

    wav_file = io.BytesIO(raw_audio)
    with wave.open(wav_file, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio_np = np.frombuffer(frames, dtype=np.int32)
    return audio_np

# --- MODELO ---
def autoencoder_model(input_dim):
    """
    Construye un autoencoder denso para detección de anomalías en espectrogramas.

    Parameters
    ----------
    input_dim : int
        Dimensión de entrada (N_MELS * FIXED_FRAMES).

    Returns
    -------
    keras.Model
        Autoencoder compilado listo para entrenamiento.
    """
    inp = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inp)
    x = Dense(128, activation='relu')(inp)
    x = Dense(32, activation='relu')(x)
    bottleneck = Dense(8, activation='relu')(x)
    x = Dense(32, activation='relu')(bottleneck)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(inp)
    decoded = Dense(input_dim, activation='sigmoid')(x)
    model = Model(inp, decoded)
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    return model

# --- FLUJO PRINCIPAL ---
def main():
    """
    Función principal: entrena un autoencoder en tiempo real con audio grabado,
    y luego realiza inferencia continua para detectar anomalías acústicas.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='Tamaño del batch para entrenamiento.')
    parser.add_argument('--threshold', type=float, default=0.1, help='Umbral de error para detección de anomalías.')
    args = parser.parse_args()

    input_dim = N_MELS * FIXED_FRAMES
    model = autoencoder_model(input_dim)
    print('[INFO] Entrenando hasta que val_loss sea menor que', args.threshold, '...')

    sample_idx = 0
    while True:
        X_batch = []
        for _ in range(BATCH_AUDIOS):
            sig = record_audio()
            feat = preprocess_signal(sig).flatten()
            X_batch.append(feat)

        X_batch = np.array(X_batch)
        X_train, X_val = train_test_split(X_batch, test_size=0.2, random_state=42)

        early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
        history = model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=10,
            batch_size=args.batch_size,
            verbose=2,
            callbacks=[early]
        )

        val_loss = history.history['val_loss'][0]
        loss = history.history['loss'][0]
        mae = history.history['mae'][0]
        print(f'Época {sample_idx} -> loss: {loss:.6f}, val_loss: {val_loss:.6f}, mae: {mae:.6f}')
        sample_idx += 1

        if val_loss < args.threshold:
            print('[INFO] Umbral alcanzado. Pasando a predicción...')
            break

    errores_buffer = []
    err_max = 1.0
    print('[INFO] Iniciando predicción en tiempo real...')
    while True:
        sig = record_audio()
        X_test = preprocess_signal(sig).flatten()[np.newaxis, ...]
        rec = model.predict(X_test, verbose=2)
        err = np.mean((X_test - rec) ** 2)
        # Actualización diámica de daño
        if err > args.threshold:
            errores_buffer.append(err)
            # Si excedemos el tamaño del buffer, sacamos el más viejo
            if len(errores_buffer) > WINDOW_SIZE:
                errores_buffer.pop(0)
                
        # Si tenemos suficientes valores en el buffer, calculamos err_max como percentil 95
        if len(errores_buffer) >= MIN_UPDATES_IN_WINDOW:
            nuevo_err_max = np.percentile(errores_buffer, PERCENTILE)
            # Solo actualizamos si el percentil es mayor al err_max actual
            if nuevo_err_max < err_max:
                err_max = float(nuevo_err_max)
                print(f'[INFO] err_max actualizado por percentil {PERCENTILE}: {err_max:.5f}')
        
        # Cálculo del % de daño usando err_max
        if err <= args.threshold:
            pct = 0.0
        else:
            denom = max(err_max - args.threshold, 1e-6)
            pct = np.clip((err - args.threshold) / denom, 0, 1) * 100
        
        print(f'Error: {err:.5f} | Daño estimado: {pct:6.2f}%')

if __name__ == '__main__':
    main()
