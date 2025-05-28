import argparse
import numpy as np
import librosa
import wave
import subprocess
import io
import os
import tensorflow as tf
import pickle
import time

# Librería para reducción de ruido
import noisereduce as nr

# Desactivar logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ---------------------------- PARÁMETROS GLOBALES -----------------------------

N_MELS = 128
FIXED_FRAMES = 128
DURATION = 3.0          # segundos de grabación
SAMPLING_RATE = 48000   # Hz
HOP_LENGTH = int((SAMPLING_RATE * DURATION) / FIXED_FRAMES)

TFLITE_MODEL_PATH = 'autoencoder_model.tflite'
ANOMALY_SCORE_THRESHOLD = 0.5  # Score normalizado ≥ this → anomalía

CALIBRATION_SAMPLES = 5       # Número de grabaciones iniciales para calibrar baseline


def preprocess_signal(signal):
    """Convierte la señal mono en vector de mel-spectrograma aplanado."""
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

    if S_dB.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - S_dB.shape[1]
        S_dB = np.pad(S_dB, ((0, 0), (0, pad_width)), mode='constant')
    else:
        S_dB = S_dB[:, :FIXED_FRAMES]

    mn, mx = S_dB.min(), S_dB.max()
    return ((S_dB - mn) / (mx - mn + 1e-6)).flatten().astype(np.float32)


def reduce_noise(signal):
    """
    Aplica reducción de ruido espectral utilizando noisereduce.
    Asume que 'signal' está en formato float32 y a la tasa SAMPLING_RATE.
    """
    s = signal.astype(np.float32)
    if np.max(np.abs(s)) > 0:
        s /= (np.max(np.abs(s)) + 1e-6)

    reduced = nr.reduce_noise(
        y=s,
        sr=SAMPLING_RATE,
        prop_decrease=1.0,
        stationary=False
    )
    max_orig = np.max(np.abs(signal)) + 1e-6
    return (reduced * max_orig).astype(np.float32)


def record_audio():
    """Graba audio con arecord, aplica reducción de ruido y lo devuelve como array de floats."""
    cmd = [
        'arecord', '-D', 'plughw:1', '-c1', '-r', str(SAMPLING_RATE),
        '-f', 'S16_LE', '-t', 'wav', '-d', str(int(DURATION)), '-q'
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw, err = proc.communicate(timeout=int(DURATION) + 2)
        if proc.returncode != 0:
            raise RuntimeError(err.decode().strip())

        wav = wave.open(io.BytesIO(raw), 'rb')
        frames = wav.readframes(wav.getnframes())
        signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

        # Si la señal es muy silenciosa, devolvemos ceros
        if np.sqrt(np.mean(signal**2)) < 1e-6:
            return np.zeros(int(SAMPLING_RATE * DURATION), dtype=np.float32)

        # Aplicar reducción de ruido
        denoised = reduce_noise(signal)
        return denoised

    except Exception as e:
        print(f"Error grabación: {e}")
        return np.zeros(int(SAMPLING_RATE * DURATION), dtype=np.float32)


class ProgressivePredictor:
    def __init__(self, healthy_signals):
        # Inicializamos ae_mean y max_error_seen a partir de healthy_signals.pkl
        self.ae_mean = 0.0
        self.max_error_seen = 0.0
        self._compute_baseline(healthy_signals)

    def _compute_baseline(self, healthy_signals):
        """Calcula la media de errores AE y el máximo sobre señales sanas."""
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=4)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]

        errors = []
        for sig in healthy_signals:
            X = preprocess_signal(sig).reshape(1, -1)
            interpreter.set_tensor(inp['index'], X)
            interpreter.invoke()
            recon = interpreter.get_tensor(out['index'])
            err = float(np.mean((X - recon) ** 2))
            errors.append(err)

        self.ae_mean = float(np.mean(errors))
        self.max_error_seen = float(np.max(errors))
        print(f"Baseline AE mean: {self.ae_mean:.6f}")
        print(f"Baseline AE max : {self.max_error_seen:.6f}")

    def predict(self, signal):
        """
        Calcula el error de reconstrucción AE, actualiza dinámicamente el max_error_seen
        y devuelve (score_normalizado, ae_error, is_anomaly).
        """
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=4)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]

        X = preprocess_signal(signal).reshape(1, -1)
        interpreter.set_tensor(inp['index'], X)
        interpreter.invoke()
        recon = interpreter.get_tensor(out['index'])
        ae_err = float(np.mean((X - recon) ** 2))

        # Si aparece un error mayor al que conocíamos, actualizamos max_error_seen
        if ae_err > self.max_error_seen:
            self.max_error_seen = ae_err

        denom = max(self.max_error_seen - self.ae_mean, 1e-8)
        raw_score = (ae_err - self.ae_mean) / denom
        score = float(np.clip(raw_score, 0.0, 1.0))

        is_anomaly = (score >= ANOMALY_SCORE_THRESHOLD)
        return score, ae_err, is_anomaly


def main():
    parser = argparse.ArgumentParser(description="Monitorización progresiva con AE, reducción de ruido y calibración inicial")
    parser.add_argument('--interval', type=int, default=3, help='Segundos entre mediciones')
    args = parser.parse_args()

    # 1) Cargar señales sanas para baseline teórico
    try:
        with open('healthy_signals.pkl', 'rb') as f:
            healthy_signals = pickle.load(f)
    except Exception as e:
        print(f"Error cargando 'healthy_signals.pkl': {e}")
        return

    # 2) Instanciar predictor y calcular baseline a partir de healthy_signals
    predictor = ProgressivePredictor(healthy_signals)

    # 3) Fase de calibración “real” (usar solo si estás seguro de que estos CALIBRATION_SAMPLES son audios sanos)
    print(f"⚙️  Comenzando fase de calibración con {CALIBRATION_SAMPLES} grabaciones (audios sanos) …")
    calibration_errors = []
    for i in range(CALIBRATION_SAMPLES):
        sig = record_audio()
        # Si la señal es muy baja, repetimos este ciclo sin contarla
        if np.sqrt(np.mean(sig**2)) < 1e-6:
            print(f"  {i+1}/{CALIBRATION_SAMPLES}: Señal muy débil, repitiendo …")
            time.sleep(args.interval)
            i -= 1
            continue

        # Calculamos AE error sin marcar anomalía todavía
        score_tmp, ae_err_tmp, _ = predictor.predict(sig)
        calibration_errors.append(ae_err_tmp)
        print(f"  {i+1}/{CALIBRATION_SAMPLES} calibrado: AE_err={ae_err_tmp:.6f}")
        time.sleep(args.interval)

    # Ajustamos baseline “real” con los errores medidos
    new_mean = float(np.mean(calibration_errors))
    new_max = float(np.max(calibration_errors))
    predictor.ae_mean = new_mean
    predictor.max_error_seen = new_max
    print(f"✅ Baseline recalibrado: AE mean={new_mean:.6f}, AE max={new_max:.6f}")
    print("🔧 INICIANDO MONITORIZACIÓN PROGRESIVA (con reducción de ruido y calibración real)\n")

    # 4) Bucle de monitorización normal
    count = 0
    anomalies = 0
    try:
        while True:
            count += 1
            sig = record_audio()
            if np.sqrt(np.mean(sig**2)) < 1e-6:
                print(f"#{count}: Señal muy débil, saltando")
                time.sleep(args.interval)
                continue

            score, ae_err, is_anomaly = predictor.predict(sig)
            status = "🚨 ANOMALÍA" if is_anomaly else "✅ Normal"
            if is_anomaly:
                anomalies += 1
            print(f"#{count}: AE_err={ae_err:.6f}, Score={score:.3f} → {status}")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n🛑 MONITORIZACIÓN FINALIZADA: {count} mediciones, {anomalies} anomalías")


if __name__ == '__main__':
    main()
