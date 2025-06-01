import argparse
import numpy as np
import librosa
import wave
import subprocess
import io
import os
import logging
import tensorflow as tf
import pickle
import warnings

# Configuración de hilos para TensorFlow
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'

from scipy import stats
from collections import deque
import noisereduce as nr

# ----------------------------- CONFIGURACIÓN GLOBAL -----------------------------
warnings.filterwarnings("ignore")
N_MELS = 128
FIXED_FRAMES = 128
DURATION = 3.0  # segundos de grabación
SAMPLING_RATE = 48000  # Hz

# Para que FIXED_FRAMES concuerde con mel-spectrogram, calculamos hop_length:
HOP_LENGTH = int((SAMPLING_RATE * DURATION) / FIXED_FRAMES)

CALIBRATION_SAMPLES = 15
TFLITE_MODEL_PATH = 'autoencoder_model.tflite'
WINDOW_SIZE = 8
TREND_THRESHOLD = 0.02
ANOMALY_CONSECUTIVE = 2
MIN_DETECTION_THRESHOLD = 0.15

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def preprocess_signal(signal):
    """Mismo preprocess que en train.py: mel-spectrograma aplanado."""
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


class ImprovedFaultDetector:
    def __init__(self):
        # Estos atributos serán cargados desde detector_baseline.pkl
        self.error_history = deque(maxlen=WINDOW_SIZE)
        self.ae_error_stats = {}
        self.scalers = {}
        self.anomaly_detectors = {}
        self.baseline_stats = {}
        self.feature_names = []
        self.frequency_buffer = deque(maxlen=20)
        self.energy_buffer = deque(maxlen=20)
        self.spectral_buffer = deque(maxlen=20)
        self.dynamic_threshold = 0.0
        self.detection_sensitivity = 1.0
        self.consecutive_anomalies = 0

    def extract_enhanced_features(self, signal, sr=SAMPLING_RATE):
        # Misma implementación que en extract_baseline.py
        signal = nr.reduce_noise(y=signal, sr=sr)
        if np.max(np.abs(signal)) > 0:
            signal = signal / (np.max(np.abs(signal)) + 1e-8)

        nperseg = 1024
        freqs, psd = welch(signal, fs=sr, nperseg=nperseg)
        bands = np.array([
            [0,    60],
            [60,  250],
            [250, 2000],
            [2000,6000],
            [6000,8000]
        ])
        features = {}
        for name, (low, high) in zip(self.feature_names[:5], bands):
            mask = (freqs >= low) & (freqs < high)
            features[name] = float(np.mean(psd[mask])) if np.any(mask) else 0.0

        zcr_values = librosa.feature.zero_crossing_rate(signal, frame_length=1024, hop_length=512)[0]
        features["zcr"] = float(np.mean(zcr_values))

        rms_values = librosa.feature.rms(y=signal, frame_length=1024, hop_length=512)[0]
        rms_mean = float(np.mean(rms_values))
        features["rms_mean"] = rms_mean

        centroid_values = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=1024, hop_length=512)[0]
        features["spectral_centroid_mean"] = float(np.mean(centroid_values))

        try:
            f0_values = librosa.yin(signal, fmin=50, fmax=sr/2, frame_length=1024, hop_length=512)
            f0_values = f0_values[np.isfinite(f0_values) & (f0_values > 0)]
            features["fundamental_frequency"] = float(np.mean(f0_values)) if len(f0_values) > 0 else 0.0
        except Exception:
            features["fundamental_frequency"] = 0.0

        peak = float(np.max(np.abs(signal)) + 1e-12)
        features["crest_factor"] = peak / (rms_mean + 1e-12)
        features["thd"] = 0.0
        features["periodicity_strength"] = 0.0

        ordered = {name: features.get(name, 0.0) for name in self.feature_names}
        return ordered

    def calculate_anomaly_score(self, signal, autoencoder_error):
        features = self.extract_enhanced_features(signal)
        vector = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
        vector = np.nan_to_num(vector, nan=0.0, posinf=1e6, neginf=-1e6)

        normalized = self.scalers['features'].transform(vector)

        iso_raw = self.anomaly_detectors['isolation_forest'].decision_function(normalized)[0]
        ell_raw = self.anomaly_detectors['elliptic_envelope'].decision_function(normalized)[0]

        iso_prob = float(self.scalers['iso'].transform([[-iso_raw]])[0][0])
        ell_prob = float(self.scalers['ell'].transform([[-ell_raw]])[0][0])

        # Autoencoder: mapeo basado en percentiles
        ae_prob = 0.0
        if self.ae_error_stats:
            stats_ae = self.ae_error_stats
            if autoencoder_error > stats_ae['percentile_95']:
                ae_prob = 0.9
            elif autoencoder_error > stats_ae['percentile_90']:
                ae_prob = 0.7
            elif autoencoder_error > stats_ae['percentile_75']:
                ae_prob = 0.4
            else:
                z = (autoencoder_error - stats_ae['mean']) / (stats_ae['std'] + 1e-8)
                ae_prob = np.clip((z - 0.5) / 2, 0, 1)

        med = self.baseline_stats['feature_medians']
        mad = self.baseline_stats['feature_mads']
        robust_dist = np.median(np.abs((normalized[0] - med) / mad))
        robust_prob = np.clip((robust_dist - 1) / 3, 0, 1)

        freq = features.get('fundamental_frequency', 0.0)
        energy = features.get('rms_mean', 0.0)
        spectral = features.get('spectral_centroid_mean', 0.0)

        self.frequency_buffer.append(freq)
        self.energy_buffer.append(energy)
        self.spectral_buffer.append(spectral)

        temporal_prob = 0.0
        if len(self.frequency_buffer) >= 10:
            def var_ratio(buffer):
                arr = np.array(buffer)
                recent = arr[-5:]
                baseline = arr[:-5]
                if len(baseline) < 5 or np.var(baseline) < 1e-6:
                    return 0.0
                return float(np.var(recent) / (np.var(baseline) + 1e-8))

            vr = max(
                var_ratio(self.frequency_buffer),
                var_ratio(self.energy_buffer),
                var_ratio(self.spectral_buffer)
            )
            temporal_prob = np.clip((vr - 1.5) / 3, 0, 1)

        score = (
            0.3 * iso_prob +
            0.25 * ell_prob +
            0.2 * robust_prob +
            0.15 * ae_prob +
            0.1 * temporal_prob
        ) * self.detection_sensitivity
        score = min(1.0, score)

        details = {
            'isolation': iso_prob,
            'elliptic': ell_prob,
            'robust_distance': robust_prob,
            'autoencoder': ae_prob,
            'temporal_change': temporal_prob,
            'ae_raw': autoencoder_error,
            'fundamental_freq': freq,
            'spectral_centroid': spectral,
            'rms_energy': energy,
            'crest_factor': features.get('crest_factor', 0.0),
            'thd': features.get('thd', 0.0),
            'periodicity': features.get('periodicity_strength', 0.0)
        }

        return score, details

    def detect_trend(self):
        if len(self.error_history) < 6:
            return False, 0.0

        recent_history = list(self.error_history)[-6:]
        x = np.arange(len(recent_history))
        y = np.array(recent_history)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        is_trending_up = (
            (slope > TREND_THRESHOLD) and
            (r_value > 0.5) and
            (p_value < 0.1)
        )
        return is_trending_up, slope

    def predict_failure_risk(self, current_score, trend_slope):
        risk_factors = []

        if current_score > 0.6:
            risk_factors.append(("Anomalía crítica detectada", 0.95))
        elif current_score > 0.4:
            risk_factors.append(("Anomalía alta detectada", 0.75))
        elif current_score > 0.25:
            risk_factors.append(("Anomalía moderada detectada", 0.55))
        elif current_score > MIN_DETECTION_THRESHOLD:
            risk_factors.append(("Anomalía leve detectada", 0.35))

        if trend_slope > 0.08:
            risk_factors.append(("Tendencia de deterioro rápida", 0.85))
        elif trend_slope > 0.04:
            risk_factors.append(("Tendencia de deterioro moderada", 0.6))
        elif trend_slope > TREND_THRESHOLD:
            risk_factors.append(("Tendencia de deterioro leve", 0.4))

        if self.consecutive_anomalies >= ANOMALY_CONSECUTIVE:
            consecutive_factor = min(0.9, 0.5 + 0.1 * self.consecutive_anomalies)
            risk_factors.append((f"{self.consecutive_anomalies} anomalías consecutivas", consecutive_factor))

        if current_score < self.dynamic_threshold * 0.5:
            self.dynamic_threshold = max(MIN_DETECTION_THRESHOLD,
                                         self.dynamic_threshold - self.threshold_adaptation_rate)
        elif current_score > self.dynamic_threshold:
            self.dynamic_threshold = min(0.8,
                                         self.dynamic_threshold + self.threshold_adaptation_rate)

        if not risk_factors:
            total_risk = 0.0
            risk_level = "NORMAL"
            action = "Sistema funcionando correctamente"
        else:
            weights = [w for (_, w) in risk_factors]
            mean_w = np.mean(weights)
            bonus = 0.2 * (np.max(weights) - mean_w)
            total_risk = min(1.0, mean_w + bonus)

            if total_risk >= 0.8:
                risk_level = "CRÍTICO"
                action = "🚨 PARAR MÁQUINA - Inspección inmediata requerida"
            elif total_risk >= 0.6:
                risk_level = "ALTO"
                action = "⚠️ Programar mantenimiento urgente (24-48h)"
            elif total_risk >= 0.4:
                risk_level = "MEDIO"
                action = "📋 Inspección programada - Monitoreo intensivo"
            elif total_risk >= 0.2:
                risk_level = "BAJO"
                action = "🔎 Monitoreo continuo - Seguimiento de tendencia"
            else:
                risk_level = "MÍNIMO"
                action = "✅ Continuar operación normal"

        return total_risk, risk_level, action, risk_factors


def main():
    parser = argparse.ArgumentParser(description="Monitorización de fallas en tiempo real.")
    parser.add_argument('--monitor_interval', type=int, default=3, help='Segundos entre mediciones.')
    args = parser.parse_args()

    # Cargar detector con baseline
    with open('detector_baseline.pkl', 'rb') as f:
        detector = pickle.load(f)

    # Configurar intérprete TFLite
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    logging.info("🔧 INICIANDO MONITORIZACIÓN DE FALLAS - MODO PREDICT 🔧")
    logging.info(f"▪ Sensibilidad: {detector.detection_sensitivity:.1f}")
    logging.info(f"▪ Umbral dinámico: {detector.dynamic_threshold:.3f}")
    logging.info(f"▪ Intervalo de monitoreo: {args.monitor_interval}s")
    logging.info(f"▪ Número de características extraídas: {len(detector.feature_names)}")
    logging.info("🎯 Umbrales de alerta:")
    logging.info("   • CRÍTICO: >0.6 (Parar máquina)")
    logging.info("   • ALTO: 0.4 - 0.6 (Mantenimiento urgente)")
    logging.info("   • MEDIO: 0.25 - 0.4 (Inspección programada)")
    logging.info("   • BAJO: 0.15 - 0.25 (Monitoreo intensivo)")
    logging.info("[Presiona Ctrl+C para detener]")

    measurement_count = 0
    anomaly_count = 0
    last_risk_level = "NORMAL"

    try:
        while True:
            measurement_count += 1
            logging.info(f"🔍 --- MEDICIÓN #{measurement_count} ---")

            sig = record_audio()
            X_proc = preprocess_signal(sig).reshape(1, -1).astype(np.float32)

            # Error autoencoder en TFLite
            interpreter.set_tensor(input_details['index'], X_proc)
            try:
                interpreter.invoke()
                recon = interpreter.get_tensor(output_details['index'])
                ae_error = float(np.mean((X_proc - recon) ** 2))
            except Exception as e:
                logging.error(f"Error en inferencia TFLite: {e}")
                ae_error = 0.0

            # Cálculo de score de anomalía
            anomaly_score, details = detector.calculate_anomaly_score(sig, ae_error)
            detector.error_history.append(anomaly_score)

            # Detección de tendencia
            is_trending, trend_slope = detector.detect_trend()

            # Anomalías consecutivas
            if anomaly_score > MIN_DETECTION_THRESHOLD:
                detector.consecutive_anomalies += 1
                anomaly_count += 1
            else:
                detector.consecutive_anomalies = 0

            # Predicción de riesgo
            risk_score, risk_level, action, risk_factors = detector.predict_failure_risk(
                anomaly_score, trend_slope
            )

            # Impresión de resultados principales
            logging.info(f"SCORE ANOMALÍA: {anomaly_score:.3f} (Umbral dinámico: {detector.dynamic_threshold:.3f})")
            logging.info(f"ERROR AUTOENCODER: {ae_error:.6f}")

            logging.info("SCORES INDIVIDUALES:")
            logging.info(f"   • Isolation Forest: {details['isolation']:.3f}")
            logging.info(f"   • Elliptic Envelope: {details['elliptic']:.3f}")
            logging.info(f"   • Distancia Robusta: {details['robust_distance']:.3f}")
            logging.info(f"   • Autoencoder: {details['autoencoder']:.3f}")
            logging.info(f"   • Cambio Temporal: {details['temporal_change']:.3f}")

            logging.info("CARACTERÍSTICAS DEL AUDIO:")
            logging.info(f"   • Freq. Fundamental: {details['fundamental_freq']:.1f} Hz")
            logging.info(f"   • Centroide Espectral: {details['spectral_centroid']:.1f} Hz")
            logging.info(f"   • Energía RMS: {details['rms_energy']:.1f}")
            logging.info(f"   • Factor de Cresta: {details['crest_factor']:.2f}")
            logging.info(f"   • Distorsión Armónica: {details['thd']:.3f}")
            logging.info(f"   • Periodicidad: {details['periodicity']:.3f}")

            if is_trending:
                logging.info(f"📈 TENDENCIA DETECTADA: Pendiente +{trend_slope:.4f} por medición")

            if detector.consecutive_anomalies > 0:
                logging.info(f"⚡ ANOMALÍAS CONSECUTIVAS: {detector.consecutive_anomalies}")

            risk_emojis = {
                "CRÍTICO": "🚨",
                "ALTO": "⚠️",
                "MEDIO": "📋",
                "BAJO": "🔎",
                "MÍNIMO": "✅",
                "NORMAL": "✅"
            }
            emoji = risk_emojis.get(risk_level, "❓")
            logging.info(f"{emoji} EVALUACIÓN DE RIESGO: NIVEL={risk_level} ({risk_score:.2f}) | ACCIÓN={action}")

            if risk_factors:
                logging.info("FACTORES DETECTADOS:")
                for fact, wt in risk_factors:
                    logging.info(f"   • {fact} (peso={wt:.2f})")

            if risk_level == "CRÍTICO":
                logging.warning("🚨🚨🚨 ALERTA CRÍTICA: REVISAR MÁQUINA INMEDIATAMENTE 🚨🚨🚨")
            elif risk_level == "ALTO" and last_risk_level != "ALTO":
                logging.warning("⚠️ NUEVA ALERTA: Programar mantenimiento urgente ⚠️")
            elif detector.consecutive_anomalies >= ANOMALY_CONSECUTIVE:
                logging.warning(f"⚠️ ATENCIÓN: {detector.consecutive_anomalies} anomalías consecutivas detectadas")

            anomaly_rate = (anomaly_count / measurement_count) * 100
            logging.info("ESTADÍSTICAS DE SESIÓN:")
            logging.info(f"   • Medidas totales: {measurement_count}")
            logging.info(f"   • Anomalías detectadas: {anomaly_count} ({anomaly_rate:.1f}%)")
            logging.info(f"   • Umbral dinámico actual: {detector.dynamic_threshold:.3f}")

            last_risk_level = risk_level
            time.sleep(args.monitor_interval)

    except KeyboardInterrupt:
        logging.info("\n" + "=" * 80)
        logging.info("🛑 MONITORIZACIÓN DETENIDA POR EL USUARIO")
        logging.info("=" * 80)
        logging.info("RESUMEN FINAL:")
        logging.info(f"   • Total mediciones: {measurement_count}")
        logging.info(f"   • Anomalías detectadas: {anomaly_count}")
        logging.info(f"   • Tasa de anomalías: {(anomaly_count / measurement_count * 100):.1f}%")
        logging.info(f"   • Último nivel de riesgo: {last_risk_level}")
        logging.info(f"   • Umbral final: {detector.dynamic_threshold:.3f}")

        if anomaly_count > 0:
            logging.info("💡 RECOMENDACIONES:")
            rate = anomaly_count / measurement_count
            if rate > 0.3:
                logging.info("   • Alta tasa de anomalías detectada")
                logging.info("   • Considerar inspección de la máquina")
                logging.info("   • Revisar condiciones de operación")
            elif rate > 0.1:
                logging.info("   • Anomalías moderadas detectadas")
                logging.info("   • Continuar monitoreo frecuente")
                logging.info("   • Programar mantenimiento preventivo")
            else:
                logging.info("   • Pocas anomalías detectadas")
                logging.info("   • Sistema funcionando correctamente")
                logging.info("   • Mantener monitoreo regular")

        logging.info("\n✅ Sistema finalizado correctamente")


if __name__ == '__main__':
    main()
