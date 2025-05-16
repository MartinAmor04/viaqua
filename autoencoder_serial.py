import argparse
import numpy as np
import librosa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import serial

# --- CONFIGURACIÓN ---
N_MELS = 128
FIXED_FRAMES = 128
DURATION = 3.0          # segundos de cada muestra
SAMPLING_RATE = 8000    # frecuencia de muestreo de Arduino

# --- PREPROCESADO ---
def preprocess_signal(signal):
    # Normalizar señal a rango [-1,1]
    signal = signal.astype(np.float32) / 1023.0 * 2 - 1
    # Espectrograma Mel
    S = librosa.feature.melspectrogram(y=signal, sr=SAMPLING_RATE, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)
    # Padding/truncado
    if S_dB.shape[1] < FIXED_FRAMES:
        pad = FIXED_FRAMES - S_dB.shape[1]
        S_dB = np.pad(S_dB, ((0,0),(0,pad)), mode='constant')
    else:
        S_dB = S_dB[:,:FIXED_FRAMES]
    # Normalizar [0,1]
    S_min, S_max = S_dB.min(), S_dB.max()
    S_norm = (S_dB - S_min) / (S_max - S_min + 1e-6)
    return S_norm[:,:,np.newaxis]

# --- MODELO ---
def build_conv_autoencoder(input_shape=(128,128,1)):
    inp = Input(shape=input_shape)
    # Encoder
    x = Conv2D(32,(3,3),activation='relu',padding='same')(inp)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(16,(3,3),activation='relu',padding='same')(x)
    encoded = MaxPooling2D((2,2),padding='same')(x)
    # Decoder
    x = Conv2DTranspose(16,(3,3),strides=2,activation='relu',padding='same')(encoded)
    x = Conv2DTranspose(32,(3,3),strides=2,activation='relu',padding='same')(x)
    decoded = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)
    model = Model(inp, decoded)
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    return model

# --- LECTURA DESDE SERIAL ---
def read_samples_from_serial(ser, num_samples):
    data = []
    while len(data) < num_samples:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        try:
            value = int(line)
            data.append(value)
        except ValueError:
            continue
    return np.array(data)

# --- FLUJO PRINCIPAL ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True, help='Puerto serial (e.g. COM3 o /dev/ttyUSB0)')
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.1, help='Valor de val_loss para detener entrenamiento')
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baudrate, timeout=1)
    model = build_conv_autoencoder()

    print('[INFO] Entrenando hasta val_loss <', args.threshold, '...')
    while True:
        # Leer un lote de muestras
        X = []
        for _ in range(args.batch_size):
            sig = read_samples_from_serial(ser, int(SAMPLING_RATE * DURATION))
            X.append(preprocess_signal(sig))
        X = np.stack(X, axis=0)
        # Entrenar 1 época con validación interna
        history = model.fit(X, X, epochs=1, batch_size=args.batch_size, validation_split=0.2, verbose=1)
        val_loss = history.history['val_loss'][0]
        print(f'val_loss: {val_loss:.4f}')
        if val_loss < args.threshold:
            print('[INFO] Umbral alcanzado. Pasando a predicción...')
            break

    # Predicciones en tiempo real
    print('[INFO] Iniciando predicción en tiempo real:')
    while True:
        sig = read_samples_from_serial(ser, int(SAMPLING_RATE * DURATION))
        X_test = preprocess_signal(sig)[np.newaxis, ...]
        rec = model.predict(X_test, verbose=0)
        err = np.mean((X_test - rec)**2)
        # Mapeo 0->10%, thresh->90%, err > thresh->>=90%
        pct = 10 + min(err/args.threshold, 1.0) * 80
        print(f'Error: {err:.5f}, Daño: {pct:.2f}%')

if __name__ == '__main__':
    main()
