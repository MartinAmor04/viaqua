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
N_MELS = 128
FIXED_FRAMES = 128
DURATION = 3.0
SAMPLING_RATE = 48000
BATCH_AUDIOS = 10

# --- PREPROCESADO ---
def preprocess_signal(signal):
    """
    Convierte una señal de audio en un espectrograma mel-normalizado de tamaño fijo.

    Args:
        signal (np.ndarray): Señal de audio en forma de array de enteros.

    Returns:
        np.ndarray: Espectrograma mel normalizado y ajustado a dimensiones fijas.
    """
    signal = signal.astype(np.float32)
    signal = signal / np.max(np.abs(signal) + 1e-6)  # Previene división por cero
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
    Graba audio durante una duración fija usando `arecord` y lo convierte en un array NumPy.

    Returns:
        np.ndarray: Señal de audio grabada en formato entero de 32 bits.
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
    Construye un modelo de autoencoder simple completamente conectado.

    Args:
        input_dim (int): Dimensión de entrada del autoencoder.

    Returns:
        tensorflow.keras.models.Model: Modelo compilado del autoencoder.
    """
    inp = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inp)
    x = Dense(32, activation='relu')(x)
    bottleneck = Dense(8, activation='relu')(x)
    x = Dense(32, activation='relu')(bottleneck)
    x = Dense(128, activation='relu')(x)
    decoded = Dense(input_dim, activation='sigmoid')(x)
    model = Model(inp, decoded)
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    return model

# --- FLUJO PRINCIPAL ---
def main():
    """
    Entrena un autoencoder con muestras de audio en tiempo real.
    Una vez alcanzado el umbral de pérdida en validación, pasa a predicción continua
    para estimar el nivel de daño acústico en nuevas muestras de audio.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='Tamaño del batch de entrenamiento')
    parser.add_argument('--threshold', type=float, default=0.1, help='Umbral de pérdida para detener entrenamiento')
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

    print('[INFO] Iniciando predicción en tiempo real...')
    while True:
        sig = record_audio()
        X_test = preprocess_signal(sig).flatten()[np.newaxis, ...]
        rec = model.predict(X_test, verbose=0)
        err = np.mean((X_test - rec) ** 2)
        pct = min(err / args.threshold, 1.0) * 100
        print(f'Error: {err:.5f}, Daño estimado: {pct:.2f}%')

if __name__ == '__main__':
    main()
