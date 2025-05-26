import numpy as np
import pickle
import logging
import os

# Configuración de hilos para TensorFlow
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'

import tensorflow as tf
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from scipy import stats
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import librosa
from scipy.signal import welch
import noisereduce as nr

# Constantes globales
N_MELS = 128
FIXED_FRAMES = 128
DURATION = 3.0  # segundos de grabación
SAMPLING_RATE = 48000  # Hz
HOP_LENGTH = int((SAMPLING_RATE * DURATION) / FIXED_FRAMES)

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


def extract_enhanced_features_helper(args):
    """
    Función auxiliar para ProcessPoolExecutor: 
    recibe (signal, feature_names) y devuelve lista de 12 valores en orden fijo.
    """
    signal, feature_names = args

    # 1) Denoise
    denoised = nr.reduce_noise(y=signal, sr=SAMPLING_RATE)

    # 2) Normalizar a [-1,1]
    if np.max(np.abs(denoised)) > 0:
        denoised = denoised / (np.max(np.abs(denoised)) + 1e-8)

    # 3) PSD con Welch
    nperseg = 1024
    freqs, psd = welch(denoised, fs=SAMPLING_RATE, nperseg=nperseg)

    bands = np.array([
        [0,    60],
        [60,  250],
        [250, 2000],
        [2000,6000],
        [6000,8000]
    ])
    band_names = ["sub_bajo_power", "bajo_power", "medio_power", "alto_power", "muy_alto_power"]
    features = {}

    # Potencia en cada banda
    for name, (low, high) in zip(band_names, bands):
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            features[name] = float(np.mean(psd[mask]))
        else:
            features[name] = 0.0

    # 4) ZCR medio
    zcr_values = librosa.feature.zero_crossing_rate(denoised, frame_length=1024, hop_length=512)[0]
    features["zcr"] = float(np.mean(zcr_values))

    # 5) RMS medio
    rms_values = librosa.feature.rms(y=denoised, frame_length=1024, hop_length=512)[0]
    rms_mean = float(np.mean(rms_values))
    features["rms_mean"] = rms_mean

    # 6) Centroide espectral medio
    centroid_values = librosa.feature.spectral_centroid(y=denoised, sr=SAMPLING_RATE, n_fft=1024, hop_length=512)[0]
    features["spectral_centroid_mean"] = float(np.mean(centroid_values))

    # 7) Frecuencia fundamental (YIN)
    try:
        f0_values = librosa.yin(denoised, fmin=50, fmax=SAMPLING_RATE/2, frame_length=1024, hop_length=512)
        f0_values = f0_values[np.isfinite(f0_values) & (f0_values > 0)]
        features["fundamental_frequency"] = float(np.mean(f0_values)) if len(f0_values) > 0 else 0.0
    except Exception:
        features["fundamental_frequency"] = 0.0

    # 8) Crest factor
    peak = float(np.max(np.abs(denoised)) + 1e-12)
    features["crest_factor"] = peak / (rms_mean + 1e-12)

    # 9) THD y periodicidad (no implementados)
    features["thd"] = 0.0
    features["periodicity_strength"] = 0.0

    # Devolver solo valores en el orden de feature_names
    return [features.get(fn, 0.0) for fn in feature_names]


class ImprovedFaultDetector:
    def __init__(self):
        # Historial de scores de anomalía (ventana deslizante)
        self.error_history = deque(maxlen=WINDOW_SIZE)
        # Estadísticas Baseline para AE
        self.baseline_ae_errors = []
        self.ae_error_stats = {}

        # Scalers y detectores
        self.scalers = {
            'features': None,
            'iso': None,
            'ell': None
        }
        self.anomaly_detectors = {
            'isolation_forest': None,
            'elliptic_envelope': None
        }

        # Stats robustas de características normales (medianas y MAD)
        self.baseline_stats = {
            'feature_medians': None,
            'feature_mads': None
        }
        # Nombres fijos de características (orden explícito)
        self.feature_names = [
            "sub_bajo_power",
            "bajo_power",
            "medio_power",
            "alto_power",
            "muy_alto_power",
            "zcr",
            "rms_mean",
            "spectral_centroid_mean",
            "fundamental_frequency",
            "crest_factor",
            "thd",
            "periodicity_strength"
        ]

        # Buffers para análisis temporal (varianzas)
        self.frequency_buffer = deque(maxlen=20)
        self.energy_buffer = deque(maxlen=20)
        self.spectral_buffer = deque(maxlen=20)

        # Umbrales adaptativos
        self.dynamic_threshold = 0.2
        self.threshold_adaptation_rate = 0.05
        self.detection_sensitivity = 1.5
        self.consecutive_anomalies = 0

    def setup_baseline(self, healthy_signals):
        logging.info("Estableciendo línea base con detección mejorada...")

        # 1) Cargar intérprete TFLite para calcular errores AE
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=4)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        # 2) Calcular errores de reconstrucción AE para cada señal
        baseline_ae_errors = []
        for idx, sig in enumerate(healthy_signals):
            X_proc = preprocess_signal(sig).reshape(1, -1).astype(np.float32)
            interpreter.set_tensor(input_details['index'], X_proc)
            interpreter.invoke()
            recon = interpreter.get_tensor(output_details['index'])
            ae_error = float(np.mean((X_proc - recon) ** 2))
            baseline_ae_errors.append(ae_error)
            if (idx + 1) % 10 == 0:
                logging.info(f"  - Calculados {idx + 1}/{len(healthy_signals)} errores AE")

        self.baseline_ae_errors = baseline_ae_errors
        median = np.median(baseline_ae_errors)
        mad = np.median(np.abs(baseline_ae_errors - median)) * 1.4826
        self.ae_error_stats = {
            'mean': median,
            'std': mad,
            'percentile_75': np.percentile(baseline_ae_errors, 75),
            'percentile_90': np.percentile(baseline_ae_errors, 90),
            'percentile_95': np.percentile(baseline_ae_errors, 95)
        }

        # 3) Extraer características mejoradas en paralelo con ProcessPoolExecutor
        logging.info("Extrayendo características mejoradas de señales sanas (paralelizado)...")
        args_list = [(sig, self.feature_names) for sig in healthy_signals]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(extract_enhanced_features_helper, args_list))

        # Convertir resultados a numpy array
        features_array = np.array(results, dtype=np.float32)

        # 4) Normalizar y entrenar detectores
        self.scalers['features'] = RobustScaler().fit(features_array)
        normalized_features = self.scalers['features'].transform(features_array)

        iso_clf = IsolationForest(
            contamination=0.01, random_state=42, n_estimators=100, max_samples=512, n_jobs=-1
        )
        iso_clf.fit(normalized_features)
        self.anomaly_detectors['isolation_forest'] = iso_clf

        ell_clf = EllipticEnvelope(contamination=0.005, random_state=42, support_fraction=0.7)
        ell_clf.fit(normalized_features)
        self.anomaly_detectors['elliptic_envelope'] = ell_clf

        iso_scores = iso_clf.decision_function(normalized_features).reshape(-1, 1)
        ell_scores = ell_clf.decision_function(normalized_features).reshape(-1, 1)
        iso_scores_inv = -iso_scores
        ell_scores_inv = -ell_scores

        self.scalers['iso'] = MinMaxScaler(feature_range=(0, 1)).fit(iso_scores_inv)
        self.scalers['ell'] = MinMaxScaler(feature_range=(0, 1)).fit(ell_scores_inv)

        medians = np.median(normalized_features, axis=0)
        mads = np.median(np.abs(normalized_features - medians), axis=0)
        mads = np.where(mads < 1e-6, 1e-6, mads)
        self.baseline_stats['feature_medians'] = medians
        self.baseline_stats['feature_mads'] = mads

        # 5) Calcular umbral dinámico inicial (usando las primeras 10 señales)
        logging.info("Calculando umbral dinámico inicial...")
        initial_scores = []
        for sig in healthy_signals[:10]:
            vals = extract_enhanced_features_helper((sig, self.feature_names))
            vector = np.array(vals, dtype=np.float32).reshape(1, -1)
            normalized = self.scalers['features'].transform(vector)

            iso_raw = iso_clf.decision_function(normalized)[0]
            ell_raw = ell_clf.decision_function(normalized)[0]

            iso_prob = float(self.scalers['iso'].transform([[-iso_raw]])[0][0])
            ell_prob = float(self.scalers['ell'].transform([[-ell_raw]])[0][0])

            robust_dist = np.median(np.abs((normalized[0] - medians) / mads))
            robust_prob = np.clip((robust_dist - 1) / 3, 0, 1)

            ae_prob = 0.0
            temporal_prob = 0.0

            score = (
                0.3 * iso_prob +
                0.25 * ell_prob +
                0.2 * robust_prob +
                0.15 * ae_prob +
                0.1 * temporal_prob
            ) * self.detection_sensitivity
            initial_scores.append(score)

        self.dynamic_threshold = float(np.mean(initial_scores) + 2 * np.std(initial_scores))
        logging.info(f"Umbral dinámico inicial: {self.dynamic_threshold:.3f}")
        logging.info(f"Línea base establecida con {len(healthy_signals)} muestras sanas.")

    def save(self, path):
        """Guarda la instancia del detector calibrado a disco."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)


def main():
    # Cargar señales sanas grabadas en train.py
    with open('healthy_signals.pkl', 'rb') as f:
        healthy_signals = pickle.load(f)

    detector = ImprovedFaultDetector()
    detector.setup_baseline(healthy_signals)
    detector.save('detector_baseline.pkl')
    logging.info("Detector calibrado y guardado en 'detector_baseline.pkl'")


if __name__ == '__main__':
    main()
