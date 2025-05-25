import argparse
import numpy as np
import librosa
import wave
import time
import subprocess
import io
from concurrent.futures import ThreadPoolExecutor
import os
import logging

# Configuración de hilos para TensorFlow
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from scipy import stats
from collections import deque
from scipy.signal import welch

# ----------------------------- CONFIGURACIÓN GLOBAL -----------------------------

N_MELS = 128
FIXED_FRAMES = 128
DURATION = 3.0  # segundos de grabación
SAMPLING_RATE = 48000  # Hz

# Para que FIXED_FRAMES concuerde con mel-spectrogram, calculamos hop_length:
HOP_LENGTH = int((SAMPLING_RATE * DURATION) / FIXED_FRAMES)

TRAIN_SAMPLES = 50
CALIBRATION_SAMPLES = 15
MODEL_CHECKPOINT = 'ae_best.keras'
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

    def extract_enhanced_features(self, signal, sr=SAMPLING_RATE):
        """
        Extrae características robustas de la señal de audio:
         - PSD en bandas definidas
         - ZCR medio
         - RMS medio
         - Centroide espectral medio
         - Frecuencia fundamental (YIN)
         - Crest factor
         - THD y periodicidad (0 si no implementados)
        Retorna diccionario con claves en el orden de self.feature_names.
        """
        # 1) Normalizar señal a [-1, 1]
        if np.max(np.abs(signal)) > 0:
            signal = signal / (np.max(np.abs(signal)) + 1e-8)
        else:
            signal = signal

        nperseg = 1024
        freqs, psd = welch(signal, fs=sr, nperseg=nperseg)
        bands = np.array([
            [0, 60],
            [60, 250],
            [250, 2000],
            [2000, 6000],
            [6000, 8000]
        ])
        band_names = ["sub_bajo_power", "bajo_power", "medio_power", "alto_power", "muy_alto_power"]
        features = {}

        freqs = np.asarray(freqs)
        psd = np.asarray(psd)
        for name, (low, high) in zip(band_names, bands):
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                band_power = float(np.mean(psd[mask]))
            else:
                band_power = 0.0
            features[name] = band_power

        # 2) ZCR medio
        zcr_values = librosa.feature.zero_crossing_rate(signal, frame_length=1024, hop_length=512)[0]
        features["zcr"] = float(np.mean(zcr_values))

        # 3) RMS medio
        rms_values = librosa.feature.rms(y=signal, frame_length=1024, hop_length=512)[0]
        rms_mean = float(np.mean(rms_values))
        features["rms_mean"] = rms_mean

        # 4) Centroide espectral medio
        centroid_values = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=1024, hop_length=512)[0]
        features["spectral_centroid_mean"] = float(np.mean(centroid_values))

        # 5) Frecuencia fundamental (YIN)
        try:
            f0_values = librosa.yin(signal, fmin=50, fmax=sr/2, frame_length=1024, hop_length=512)
            f0_values = f0_values[np.isfinite(f0_values) & (f0_values > 0)]
            if len(f0_values) > 0:
                features["fundamental_frequency"] = float(np.mean(f0_values))
            else:
                features["fundamental_frequency"] = 0.0
        except Exception:
            features["fundamental_frequency"] = 0.0

        # 6) Crest factor
        peak = float(np.max(np.abs(signal)) + 1e-12)
        features["crest_factor"] = peak / (rms_mean + 1e-12)

        # 7) THD y periodicidad: no implementados, se mantienen 0.0
        features["thd"] = 0.0
        features["periodicity_strength"] = 0.0

        # Aseguramos orden consistente
        ordered = {name: features.get(name, 0.0) for name in self.feature_names}
        return ordered

    def setup_baseline(self, healthy_signals, ae_errors=None):
        """
        Configura la línea base:
         - Calcula estadísticas robustas de errores de AE (si se pasan)
         - Extrae características de señales sanas y ajusta RobustScaler
         - Entrena IsolationForest y EllipticEnvelope
         - Calibra MinMaxScalers para scores de los detectores
         - Calcula estadísticas de medianas y MAD para distancia robusta
         - Calcula umbral dinámico inicial
        """
        logging.info("Estableciendo línea base con detección mejorada...")

        if ae_errors is not None and len(ae_errors) > 0:
            self.baseline_ae_errors = ae_errors
            median = np.median(ae_errors)
            mad = np.median(np.abs(ae_errors - median)) * 1.4826  # MAD aproximando std
            self.ae_error_stats = {
                'mean': median,
                'std': mad,
                'percentile_75': np.percentile(ae_errors, 75),
                'percentile_90': np.percentile(ae_errors, 90),
                'percentile_95': np.percentile(ae_errors, 95)
            }
            logging.info(
                f"Stats AE robustas - Mediana: {median:.6f}, MAD: {mad:.6f}, P90: {self.ae_error_stats['percentile_90']:.6f}"
            )
        else:
            logging.warning("No se proporcionaron errores de AE para calibrar estadísticas de AE.")

        # Extracción de características en paralelo
        logging.info("Extrayendo características de señales sanas...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            features_list = list(executor.map(self.extract_enhanced_features, healthy_signals))

        features_array = np.array([list(f.values()) for f in features_list], dtype=np.float32)

        # Normalización robusta de características
        self.scalers['features'] = RobustScaler().fit(features_array)
        normalized_features = self.scalers['features'].transform(features_array)

        # Entrenamiento de detectores de anomalías
        logging.info("Entrenando Isolation Forest...")
        iso_clf = IsolationForest(
            contamination=0.01, random_state=42, n_estimators=100, max_samples='auto', n_jobs=-1
        )
        iso_clf.fit(normalized_features)
        self.anomaly_detectors['isolation_forest'] = iso_clf

        logging.info("Entrenando Elliptic Envelope...")
        ell_clf = EllipticEnvelope(contamination=0.005, random_state=42, support_fraction=0.7)
        ell_clf.fit(normalized_features)
        self.anomaly_detectors['elliptic_envelope'] = ell_clf

        # Calibración de MinMaxScaler para iso_score y ell_score
        iso_scores = iso_clf.decision_function(normalized_features).reshape(-1, 1)
        ell_scores = ell_clf.decision_function(normalized_features).reshape(-1, 1)

        # Invertimos signo para que puntuación alta signifique más anómalo
        iso_scores_inv = -iso_scores
        ell_scores_inv = -ell_scores

        self.scalers['iso'] = MinMaxScaler(feature_range=(0, 1)).fit(iso_scores_inv)
        self.scalers['ell'] = MinMaxScaler(feature_range=(0, 1)).fit(ell_scores_inv)

        # Estadísticas robustas de características
        medians = np.median(normalized_features, axis=0)
        mads = np.median(np.abs(normalized_features - medians), axis=0)
        # Evitamos ceros en mads
        mads = np.where(mads < 1e-6, 1e-6, mads)
        self.baseline_stats['feature_medians'] = medians
        self.baseline_stats['feature_mads'] = mads

        # Cálculo del umbral dinámico inicial usando primeras 10 señales sanas
        logging.info("Calculando umbral dinámico inicial...")
        initial_scores = []
        for signal in healthy_signals[:10]:
            features = self.extract_enhanced_features(signal)
            vector = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
            normalized = self.scalers['features'].transform(vector)

            iso_raw = iso_clf.decision_function(normalized)[0]
            ell_raw = ell_clf.decision_function(normalized)[0]

            iso_prob = float(self.scalers['iso'].transform([[-iso_raw]])[0][0])
            ell_prob = float(self.scalers['ell'].transform([[-ell_raw]])[0][0])

            # Distancia robusta
            robust_dist = np.median(np.abs((normalized[0] - medians) / mads))
            robust_prob = np.clip((robust_dist - 1) / 3, 0, 1)

            # Autoencoder (ninguno en esta fase)
            ae_prob = 0.0

            # Temporal (0 en baseline)
            temporal_prob = 0.0

            # Score combinado con pesos predeterminados
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

    def calculate_anomaly_score(self, signal, autoencoder_error):
        """
        Calcula el score de anomalía combinando:
         - Probabilidad de IsolationForest (iso_prob)
         - Probabilidad de EllipticEnvelope (ell_prob)
         - Distancia robusta en espacio de características (robust_prob)
         - Score basado en error de autoencoder (ae_prob)
         - Cambio temporal de varianzas (temporal_prob)
        Retorna (score, detalles_dict).
        """
        features = self.extract_enhanced_features(signal)
        vector = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
        vector = np.nan_to_num(vector, nan=0.0, posinf=1e6, neginf=-1e6)

        normalized = self.scalers['features'].transform(vector)

        # Scores crudos de detectores
        iso_raw = self.anomaly_detectors['isolation_forest'].decision_function(normalized)[0]
        ell_raw = self.anomaly_detectors['elliptic_envelope'].decision_function(normalized)[0]

        # Transformación MinMax a [0,1]: invertimos signo para que más negativo = más anómalo
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
                # Z-score robusto
                z = (autoencoder_error - stats_ae['mean']) / (stats_ae['std'] + 1e-8)
                ae_prob = np.clip((z - 0.5) / 2, 0, 1)

        # Distancia robusta en espacio de características
        med = self.baseline_stats['feature_medians']
        mad = self.baseline_stats['feature_mads']
        robust_dist = np.median(np.abs((normalized[0] - med) / mad))
        robust_prob = np.clip((robust_dist - 1) / 3, 0, 1)

        # Análisis temporal de varianzas
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

        # Score combinado
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
        """
        Detecta tendencia ascendente en los últimos WINDOW_SIZE scores:
         - Requiere al menos 6 muestras para ser fiable.
         - Retorna (is_trending_up: bool, slope: float).
        """
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
        """
        Combina diversos factores (score actual, tendencia, anomalías consecutivas)
        para asignar un nivel de riesgo y acción recomendada.
        Retorna (total_risk, risk_level, action, risk_factors_list).
        """
        risk_factors = []

        # Basado en current_score
        if current_score > 0.6:
            risk_factors.append(("Anomalía crítica detectada", 0.95))
        elif current_score > 0.4:
            risk_factors.append(("Anomalía alta detectada", 0.75))
        elif current_score > 0.25:
            risk_factors.append(("Anomalía moderada detectada", 0.55))
        elif current_score > MIN_DETECTION_THRESHOLD:
            risk_factors.append(("Anomalía leve detectada", 0.35))

        # Basado en tendencia
        if trend_slope > 0.08:
            risk_factors.append(("Tendencia de deterioro rápida", 0.85))
        elif trend_slope > 0.04:
            risk_factors.append(("Tendencia de deterioro moderada", 0.6))
        elif trend_slope > TREND_THRESHOLD:
            risk_factors.append(("Tendencia de deterioro leve", 0.4))

        # Anomalías consecutivas
        if self.consecutive_anomalies >= ANOMALY_CONSECUTIVE:
            consecutive_factor = min(0.9, 0.5 + 0.1 * self.consecutive_anomalies)
            risk_factors.append((f"{self.consecutive_anomalies} anomalías consecutivas", consecutive_factor))

        # Adaptar umbral dinámico
        if current_score < self.dynamic_threshold * 0.5:
            self.dynamic_threshold = max(MIN_DETECTION_THRESHOLD,
                                         self.dynamic_threshold - self.threshold_adaptation_rate)
        elif current_score > self.dynamic_threshold:
            self.dynamic_threshold = min(0.8,
                                         self.dynamic_threshold + self.threshold_adaptation_rate)

        # Calcular riesgo total
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
                action = "👁️ Monitoreo continuo - Seguimiento de tendencia"
            else:
                risk_level = "MÍNIMO"
                action = "✅ Continuar operación normal"

        return total_risk, risk_level, action, risk_factors


def preprocess_signal(signal):
    """
    Convierte la señal mono en un vector de características para el autoencoder:
     - Normaliza a [-1,1]
     - Calcula mel-spectrogram con hop_length fijo para tener FIXED_FRAMES
     - Convierte a dB y normaliza a [0,1]
     - Hace flatten y retorna float32
    """
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
    """
    Graba audio usando arecord y devuelve numpy.array float32:
     - Usa S16_LE para que coincida con dtype=np.int16
     - Maneja excepciones si arecord falla o dispositivo no existe
    """
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
        # Retornar silencio para que el sistema continue (aunque con baja calidad)
        return np.zeros(int(SAMPLING_RATE * DURATION), dtype=np.float32)


def autoencoder_model(input_dim):
    """
    Define y compila un autoencoder completamente conectado:
     - Encoder: 224 → 112 → 64 → 24
     - Decoder simétrico
     - Activación 'sigmoid' al final para salida [0,1]
     - Optimizer Adam con learning_rate ~3.2e-4
    """
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
    """
    Convierte el modelo Keras a TensorFlow Lite con optimizaciones:
     - Optimizations DEFAULT
     - Guarda en archivo indicado por 'path'
    """
    # No es necesario guardar de nuevo el checkpoint si ya se cargó
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
    parser = argparse.ArgumentParser(description="Detección temprana de fallas en máquinas basado en audio.")
    parser.add_argument('--batch_size', type=int, default=8, help='Tamaño de batch para entrenamiento del autoencoder.')
    parser.add_argument('--monitor_interval', type=int, default=3, help='Segundos entre mediciones en modo monitorización.')
    parser.add_argument('--sensitivity', type=float, default=1.5, help='Factor de sensibilidad (1.0 a 3.0).')
    args = parser.parse_args()

    input_dim = N_MELS * FIXED_FRAMES
    detector = ImprovedFaultDetector()
    detector.detection_sensitivity = args.sensitivity

    logging.info(f"Sistema de detección mejorado - Sensibilidad inicial: {args.sensitivity:.1f}")
    logging.info(f"Recopilando {TRAIN_SAMPLES} audios sanos para entrenamiento de autoencoder...")

    X = []
    healthy_signals = []

    # 1) Recolección de señales sanas
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
        epochs=120,
        batch_size=args.batch_size,
        callbacks=[cb_ckpt, cb_es, cb_rlr],
        verbose=2
    )

    # 3) Cargar pesos óptimos y convertir a TFLite
    model.load_weights(MODEL_CHECKPOINT)
    convertir_a_tflite(model, TFLITE_MODEL_PATH)

    # 4) Calcular errores de línea base del autoencoder
    logging.info("Calculando errores de línea base del autoencoder...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    baseline_ae_errors = []
    for idx, sig in enumerate(healthy_signals):
        X_proc = preprocess_signal(sig).reshape(1, -1).astype(np.float32)
        interpreter.set_tensor(input_details['index'], X_proc)
        try:
            interpreter.invoke()
            recon = interpreter.get_tensor(output_details['index'])
            ae_error = float(np.mean((X_proc - recon) ** 2))
        except Exception as e:
            logging.error(f"Error en inferencia TFLite en muestra {idx + 1}: {e}")
            ae_error = 0.0
        baseline_ae_errors.append(ae_error)
        if (idx + 1) % 10 == 0:
            logging.info(f"  - Procesadas {idx + 1}/{len(healthy_signals)} muestras para AE")

    # 5) Configuración del detector con línea base
    detector.setup_baseline(healthy_signals, baseline_ae_errors)

    # 6) Calibración con señales adicionales
    logging.info(f"Calibrando sistema con {CALIBRATION_SAMPLES} audios adicionales...")
    calibration_scores = []
    for i in range(CALIBRATION_SAMPLES):
        logging.info(f"Grabando audio de calibración {i + 1}/{CALIBRATION_SAMPLES}...")
        sig = record_audio()
        X_proc = preprocess_signal(sig).reshape(1, -1).astype(np.float32)

        # Error AE
        interpreter.set_tensor(input_details['index'], X_proc)
        try:
            interpreter.invoke()
            recon = interpreter.get_tensor(output_details['index'])
            ae_error = float(np.mean((X_proc - recon) ** 2))
        except Exception as e:
            logging.error(f"Error en inferencia TFLite durante calibración: {e}")
            ae_error = 0.0

        score, details = detector.calculate_anomaly_score(sig, ae_error)
        calibration_scores.append(score)
        detector.error_history.append(score)

        logging.info(
            f"  - Score={score:.3f} | AE_error={ae_error:.6f} | "
            f"Freq={details['fundamental_freq']:.1f}Hz | RMS={details['rms_energy']:.1f}"
        )

    cal_mean = float(np.mean(calibration_scores))
    cal_std = float(np.std(calibration_scores))
    cal_max = float(np.max(calibration_scores))
    logging.info(f"Calibración completa - Media={cal_mean:.3f}, Std={cal_std:.3f}, Max={cal_max:.3f}")

    # Ajuste de sensibilidad si es necesario
    if cal_mean > 0.3:
        logging.warning("Scores de calibración altos. Se ajusta sensibilidad.")
        detector.detection_sensitivity = max(1.0, detector.detection_sensitivity - 0.3)
        logging.info(f"Nueva sensibilidad: {detector.detection_sensitivity:.1f}")

    # 7) Inicio de monitorización en tiempo real
    logging.info("=" * 80)
    logging.info("🔧 INICIANDO MONITORIZACIÓN DE FALLAS - MODO MEJORADO 🔧")
    logging.info("=" * 80)
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
    logging.info("=" * 80)

    measurement_count = 0
    anomaly_count = 0
    last_risk_level = "NORMAL"

    try:
        while True:
            measurement_count += 1
            logging.info(f"🔍 --- MEDICIÓN #{measurement_count} ---")

            # Grabar y procesar audio
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

            # Scores individuales
            logging.info("SCORES INDIVIDUALES:")
            logging.info(f"   • Isolation Forest: {details['isolation']:.3f}")
            logging.info(f"   • Elliptic Envelope: {details['elliptic']:.3f}")
            logging.info(f"   • Distancia Robusta: {details['robust_distance']:.3f}")
            logging.info(f"   • Autoencoder: {details['autoencoder']:.3f}")
            logging.info(f"   • Cambio Temporal: {details['temporal_change']:.3f}")

            # Características clave del audio
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

            # Nivel de riesgo y acción
            risk_emojis = {
                "CRÍTICO": "🚨",
                "ALTO": "⚠️",
                "MEDIO": "📋",
                "BAJO": "👁️",
                "MÍNIMO": "✅",
                "NORMAL": "✅"
            }
            emoji = risk_emojis.get(risk_level, "❓")
            logging.info(f"{emoji} EVALUACIÓN DE RIESGO: NIVEL={risk_level} ({risk_score:.2f}) | ACCIÓN={action}")

            if risk_factors:
                logging.info("FACTORES DETECTADOS:")
                for fact, wt in risk_factors:
                    logging.info(f"   • {fact} (peso={wt:.2f})")

            # Alertas especiales
            if risk_level == "CRÍTICO":
                logging.warning("🚨🚨🚨 ALERTA CRÍTICA: REVISAR MÁQUINA INMEDIATAMENTE 🚨🚨🚨")
            elif risk_level == "ALTO" and last_risk_level != "ALTO":
                logging.warning("⚠️ NUEVA ALERTA: Programar mantenimiento urgente ⚠️")
            elif detector.consecutive_anomalies >= ANOMALY_CONSECUTIVE:
                logging.warning(f"⚠️ ATENCIÓN: {detector.consecutive_anomalies} anomalías consecutivas detectadas")

            # Estadísticas de sesión
            anomaly_rate = (anomaly_count / measurement_count) * 100
            logging.info("ESTADÍSTICAS DE SESIÓN:")
            logging.info(f"   • Medidas totales: {measurement_count}")
            logging.info(f"   • Anomalías detectadas: {anomaly_count} ({anomaly_rate:.1f}%)")
            logging.info(f"   • Umbral dinámico actual: {detector.dynamic_threshold:.3f}")

            last_risk_level = risk_level
            time.sleep(args.monitor_interval)

    except KeyboardInterrupt:
        # Resumen final al detener el monitoreo
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
