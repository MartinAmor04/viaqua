import argparse
import numpy as np
import librosa
import wave
import time
import subprocess
import io
from concurrent.futures import ThreadPoolExecutor
import os
os.environ['TF_NUM_INTRAOP_THREADS'] ='4'
os.environ['TF_NUM_INTEROP_THREADS'] ='4'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from scipy import stats
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN OPTIMIZADA ---
N_MELS = 128
FIXED_FRAMES = 128
DURATION = 3.0
SAMPLING_RATE = 48000

TRAIN_SAMPLES = 50
CALIBRATION_SAMPLES = 15
MODEL_CHECKPOINT = 'ae_best.keras'
TFLITE_MODEL_PATH = 'autoencoder_model.tflite'

# Parámetros más sensibles para detección temprana
WINDOW_SIZE = 8
TREND_THRESHOLD = 0.02  # Más sensible a cambios
ANOMALY_CONSECUTIVE = 2  # Reducido para detección más rápida
MIN_DETECTION_THRESHOLD = 0.15  # Umbral mínimo más bajo


class ImprovedFaultDetector:
    def __init__(self):
        self.error_history = deque(maxlen=WINDOW_SIZE)
        self.feature_history = deque(maxlen=WINDOW_SIZE)
        self.baseline_stats = {}
        self.scalers = {}
        self.anomaly_detectors = {}
        self.consecutive_anomalies = 0
        self.baseline_ae_errors = []
        self.ae_error_stats = {}

        # Umbrales adaptativos más sensibles
        self.dynamic_threshold = 0.2
        self.threshold_adaptation_rate = 0.05
        self.detection_sensitivity = 1.5  # Factor de sensibilidad

        # Buffers para análisis temporal
        self.frequency_buffer = deque(maxlen=20)
        self.energy_buffer = deque(maxlen=20)
        self.spectral_buffer = deque(maxlen=20)

    def extract_enhanced_features(signal, sr=16000):
        nperseg = 1024

        # Calcular PSD solo una vez
        freqs, psd = welch(signal, fs=sr, nperseg=nperseg)

        # Bandas predefinidas
        bands = np.array([
            [0, 60],
            [60, 250],
            [250, 2000],
            [2000, 6000],
            [6000, 8000]
        ])
        band_names = ["sub_bajo", "bajo", "medio", "alto", "muy_alto"]

        features = {}

        # Convertimos freqs y psd en arrays numpy para indexación eficiente
        freqs = np.asarray(freqs)
        psd = np.asarray(psd)

        for name, (low, high) in zip(band_names, bands):
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                band_power = np.mean(psd[mask])
            else:
                band_power = 0.0
            features[f"{name}_power"] = band_power

        # ZCR eficiente (evita signo nulo)
        signs = np.sign(signal)
        signs[signs == 0] = 1
        zcr = np.mean(signs[1:] != signs[:-1])

        # RMS
        rms = np.sqrt(np.mean(np.square(signal), dtype=np.float32))

        features["zcr"] = zcr
        features["rms"] = rms

        return features

    def setup_baseline(self, healthy_signals, ae_errors=None):
        """Establece línea base más robusta"""
        print("[INFO] Estableciendo línea base con detección mejorada...")

        if ae_errors is not None:
            self.baseline_ae_errors = ae_errors
            # Usar estimadores robustos para evitar outliers
            robust_mean = np.median(ae_errors)
            robust_std = np.median(np.abs(ae_errors - robust_mean)) * 1.4826  # MAD

            self.ae_error_stats = {
                'mean': robust_mean,
                'std': robust_std,
                'percentile_75': np.percentile(ae_errors, 75),
                'percentile_90': np.percentile(ae_errors, 90),
                'percentile_95': np.percentile(ae_errors, 95)
            }
            print(f"[INFO] Stats AE robustas - Mediana: {robust_mean:.6f}, "
                  f"MAD: {robust_std:.6f}, P90: {self.ae_error_stats['percentile_90']:.6f}")

        # Extraer características mejoradas
        all_features=[]
        # Extraer características mejoradas
        with ThreadPoolExecutor(max_workers=4) as executor:
            features_list = list(executor.map(self.extract_enhanced_features, healthy_signals))
        all_features = [list(feats.values()) for feats in features_list]

        features_array = np.array(all_features)

        # Usar RobustScaler para ser menos sensible a outliers
        self.scalers['features'] = RobustScaler()
        normalized_features = self.scalers['features'].fit_transform(features_array)

        # Detectores de anomalías más sensibles
        self.anomaly_detectors['isolation_forest'] = IsolationForest(
            contamination=0.1,  # Más permisivo para capturar más anomalías
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        self.anomaly_detectors['isolation_forest'].fit(normalized_features)

        self.anomaly_detectors['elliptic_envelope'] = EllipticEnvelope(
            contamination=0.1,
            random_state=42,
            support_fraction=0.7  # Más restrictivo en el soporte
        )
        self.anomaly_detectors['elliptic_envelope'].fit(normalized_features)

        # Estadísticas robustas de línea base
        self.baseline_stats = {
            'feature_medians': np.median(normalized_features, axis=0),
            'feature_mads': np.median(np.abs(normalized_features - np.median(normalized_features, axis=0)), axis=0),
            'feature_names': list(self.extract_enhanced_features(healthy_signals[0]).keys())
        }

        # Establecer umbral dinámico inicial
        initial_scores = []
        for i in range(min(10, len(healthy_signals))):
            features = self.extract_enhanced_features(healthy_signals[i])
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            normalized = self.scalers['features'].transform(feature_vector)

            iso_score = self.anomaly_detectors['isolation_forest'].decision_function(normalized)[0]
            ell_score = self.anomaly_detectors['elliptic_envelope'].decision_function(normalized)[0]

            combined_score = max(0, (-iso_score + 0.2) / 0.8) * 0.5 + max(0, (-ell_score + 1) / 4) * 0.5
            initial_scores.append(combined_score)

        self.dynamic_threshold = np.mean(initial_scores) + 2 * np.std(initial_scores)
        print(f"[INFO] Umbral dinámico inicial: {self.dynamic_threshold:.3f}")

        print(f"[INFO] Línea base establecida con {len(healthy_signals)} muestras")

    def calculate_anomaly_score(self, signal, autoencoder_error):
        """Cálculo de anomalía mejorado con mayor sensibilidad"""
        features = self.extract_enhanced_features(signal)
        feature_vector = np.array(list(features.values())).reshape(1, -1)

        # Manejar valores NaN o infinitos
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)

        # Normalizar características
        normalized_features = self.scalers['features'].transform(feature_vector)

        # Detectores de anomalías con scoring mejorado
        isolation_score = self.anomaly_detectors['isolation_forest'].decision_function(normalized_features)[0]
        elliptic_score = self.anomaly_detectors['elliptic_envelope'].decision_function(normalized_features)[0]

        # Convertir a probabilidades con mayor sensibilidad
        isolation_prob = max(0, min(1, (-isolation_score + 0.1) / 0.6))  # Más sensible
        elliptic_prob = max(0, min(1, (-elliptic_score + 0.5) / 2.5))  # Más sensible

        # Score de autoencoder mejorado
        ae_prob = 0
        if hasattr(self, 'ae_error_stats') and self.ae_error_stats:
            # Usar percentiles en lugar de z-score para mayor robustez
            if autoencoder_error > self.ae_error_stats['percentile_95']:
                ae_prob = 0.9
            elif autoencoder_error > self.ae_error_stats['percentile_90']:
                ae_prob = 0.7
            elif autoencoder_error > self.ae_error_stats['percentile_75']:
                ae_prob = 0.4
            else:
                # Z-score suavizado
                z_score = (autoencoder_error - self.ae_error_stats['mean']) / (self.ae_error_stats['std'] + 1e-8)
                ae_prob = max(0, min(1, (z_score - 0.5) / 2))  # Más sensible

        # Distancia robusta usando MAD
        baseline_median = self.baseline_stats['feature_medians']
        baseline_mad = self.baseline_stats['feature_mads']
        baseline_mad = np.where(baseline_mad == 0, 1e-6, baseline_mad)

        robust_distance = np.median(np.abs((normalized_features[0] - baseline_median) / baseline_mad))
        robust_prob = max(0, min(1, (robust_distance - 1) / 3))  # Más sensible

        # Detectar cambios temporales
        temporal_prob = 0
        current_freq = features.get('fundamental_frequency', 0)
        current_energy = features.get('rms_mean', 0)
        current_spectral = features.get('spectral_centroid_mean', 0)

        # Agregar a buffers
        self.frequency_buffer.append(current_freq)
        self.energy_buffer.append(current_energy)
        self.spectral_buffer.append(current_spectral)

        # Detectar cambios significativos en las últimas mediciones
        if len(self.frequency_buffer) >= 5:
            freq_var = np.var(list(self.frequency_buffer)[-5:])
            energy_var = np.var(list(self.energy_buffer)[-5:])
            spectral_var = np.var(list(self.spectral_buffer)[-5:])

            # Normalizar variaciones
            freq_baseline_var = np.var(list(self.frequency_buffer)[:-2]) if len(self.frequency_buffer) > 7 else freq_var
            energy_baseline_var = np.var(list(self.energy_buffer)[:-2]) if len(self.energy_buffer) > 7 else energy_var
            spectral_baseline_var = np.var(list(self.spectral_buffer)[:-2]) if len(
                self.spectral_buffer) > 7 else spectral_var

            freq_change = freq_var / (freq_baseline_var + 1e-8) if freq_baseline_var > 0 else 0
            energy_change = energy_var / (energy_baseline_var + 1e-8) if energy_baseline_var > 0 else 0
            spectral_change = spectral_var / (spectral_baseline_var + 1e-8) if spectral_baseline_var > 0 else 0

            temporal_prob = max(0, min(1, (max(freq_change, energy_change, spectral_change) - 1.5) / 3))

        # Combinar scores con pesos optimizados para mayor sensibilidad
        combined_score = (
                0.3 * isolation_prob +
                0.25 * elliptic_prob +
                0.2 * robust_prob +
                0.15 * ae_prob +
                0.1 * temporal_prob
        )

        # Aplicar factor de sensibilidad
        combined_score *= self.detection_sensitivity
        combined_score = min(1.0, combined_score)  # Asegurar que no exceda 1.0

        return combined_score, {
            'isolation': isolation_prob,
            'elliptic': elliptic_prob,
            'robust_distance': robust_prob,
            'autoencoder': ae_prob,
            'temporal_change': temporal_prob,
            'ae_raw': autoencoder_error,
            'fundamental_freq': current_freq,
            'spectral_centroid': current_spectral,
            'rms_energy': current_energy,
            'crest_factor': features.get('crest_factor', 0),
            'thd': features.get('thd', 0),
            'periodicity': features.get('periodicity_strength', 0)
        }

    def detect_trend(self):
        """Detección de tendencias mejorada"""
        if len(self.error_history) < 4:  # Menos muestras requeridas
            return False, 0

        # Usar últimas mediciones para tendencia más reciente
        recent_history = list(self.error_history)[-6:] if len(self.error_history) >= 6 else list(self.error_history)

        if len(recent_history) < 3:
            return False, 0

        x = np.arange(len(recent_history))
        y = np.array(recent_history)

        # Regresión lineal
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Criterios más sensibles para tendencia
        is_trending_up = (
                slope > TREND_THRESHOLD and
                r_value > 0.5 and  # Correlación más permisiva
                p_value < 0.1  # Significancia más permisiva
        )

        return is_trending_up, slope

    def predict_failure_risk(self, current_score, trend_slope):
        """Predicción de riesgo mejorada y más sensible"""
        risk_factors = []

        # Umbrales más sensibles y escalonados
        if current_score > 0.6:
            risk_factors.append(("Anomalía crítica detectada", 0.95))
        elif current_score > 0.4:
            risk_factors.append(("Anomalía alta detectada", 0.75))
        elif current_score > 0.25:
            risk_factors.append(("Anomalía moderada detectada", 0.55))
        elif current_score > MIN_DETECTION_THRESHOLD:
            risk_factors.append(("Anomalía leve detectada", 0.35))

        # Análisis de tendencias más sensible
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

        # Actualizar umbral dinámico
        if current_score < self.dynamic_threshold * 0.5:
            self.dynamic_threshold = max(MIN_DETECTION_THRESHOLD,
                                         self.dynamic_threshold - self.threshold_adaptation_rate)
        elif current_score > self.dynamic_threshold:
            self.dynamic_threshold = min(0.8,
                                         self.dynamic_threshold + self.threshold_adaptation_rate)

        # Calcular riesgo total
        if not risk_factors:
            total_risk = 0
            risk_level = "NORMAL"
            action = "Sistema funcionando correctamente"
        else:
            # Usar combinación weighted de factores
            weights = [factor[1] for factor in risk_factors]
            total_risk = np.mean(weights) + 0.2 * (np.max(weights) - np.mean(weights))  # Media + bonus por pico
            total_risk = min(1.0, total_risk)

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


# Mantener el resto de funciones auxiliares igual
def preprocess_signal(signal):
    """Preprocesa la señal para el autoencoder"""
    s = signal.astype(np.float16)
    s /= (np.max(np.abs(s)) + 1e-6)
    S = librosa.feature.melspectrogram(y=s, sr=SAMPLING_RATE, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)
    if S_dB.shape[1] < FIXED_FRAMES:
        pad = FIXED_FRAMES - S_dB.shape[1]
        S_dB = np.pad(S_dB, ((0, 0), (0, pad)), mode='constant')
    else:
        S_dB = S_dB[:, :FIXED_FRAMES]
    mn, mx = S_dB.min(), S_dB.max()
    norm = (S_dB - mn) / (mx - mn + 1e-6)
    return norm.flatten().astype(np.float16)


def record_audio():
    """Graba audio del micrófono"""
    cmd = [
        'arecord', '-D', 'plughw:1', '-c1', '-r', str(SAMPLING_RATE),
        '-f', 'S32_LE', '-t', 'wav', '-d', str(int(DURATION)), '-q'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw = proc.stdout.read()
    proc.wait()
    wav = wave.open(io.BytesIO(raw), 'rb')
    frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float16)


def autoencoder_model(input_dim):
    """Modelo de autoencoder mejorado"""
    inp = Input(shape=(input_dim,))

    # Encoder más profundo
    x = Dense(128, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)

    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Bottleneck más pequeño para forzar compresión
    bottleneck = Dense(16, activation='relu')(x)

    # Decoder simétrico
    x = Dense(32, activation='relu')(bottleneck)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)

    output = Dense(input_dim, activation='sigmoid')(x)

    model = Model(inp, output)
    model.compile(optimizer=Adam(learning_rate=5e-4), loss='mse', metrics=['mae'])
    return model


def convertir_a_tflite(model, path):
    """Convierte el modelo a TensorFlow Lite"""
    model.save(MODEL_CHECKPOINT)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print(f'[INFO] Modelo convertido a {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--monitor_interval', type=int, default=3, help='Intervalo entre mediciones (segundos)')
    parser.add_argument('--sensitivity', type=float, default=1.5, help='Factor de sensibilidad (1.0-3.0)')
    args = parser.parse_args()

    input_dim = N_MELS * FIXED_FRAMES
    detector = ImprovedFaultDetector()
    detector.detection_sensitivity = args.sensitivity

    # Proceso de entrenamiento y configuración
    print(f'[INFO] Sistema de detección mejorado - Sensibilidad: {args.sensitivity}')
    print(f'[INFO] Recopilando {TRAIN_SAMPLES} audios sanos para entrenamiento...')
    X = []
    healthy_signals = []

    for i in range(TRAIN_SAMPLES):
        sig = record_audio()
        healthy_signals.append(sig)
        X.append(preprocess_signal(sig))
        print(f'  - {i + 1}/{TRAIN_SAMPLES} (RMS: {np.sqrt(np.mean(sig ** 2)):.1f})')

    X = np.array(X)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)

    # Entrenar autoencoder
    model = autoencoder_model(input_dim)
    cb_ckpt = ModelCheckpoint(MODEL_CHECKPOINT, monitor='val_loss', save_best_only=True, verbose=1)
    cb_es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    cb_rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=8, min_lr=1e-6, verbose=1)

    print('[INFO] Entrenando autoencoder mejorado...')
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=120,
        batch_size=args.batch_size,
        callbacks=[cb_ckpt, cb_es, cb_rlr],
        verbose=2
    )

    # Cargar mejor modelo y convertir a TFLite
    model.load_weights(MODEL_CHECKPOINT)
    convertir_a_tflite(model, TFLITE_MODEL_PATH)

    # Configurar detector mejorado
    print('[INFO] Configurando detector avanzado...')

    # Calcular errores de línea base del autoencoder
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH,num_threads=4)
    interpreter.allocate_tensors()
    inp_d, out_d = interpreter.get_input_details()[0], interpreter.get_output_details()[0]

    print('[INFO] Calculando errores de línea base del autoencoder...')
    baseline_ae_errors = []
    for i, sig in enumerate(healthy_signals):
        X_proc = preprocess_signal(sig)[None, :].astype(np.float32)
        interpreter.set_tensor(inp_d['index'], X_proc)
        interpreter.invoke()
        recon = interpreter.get_tensor(out_d['index'])
        ae_error = np.mean((X_proc - recon) ** 2)
        baseline_ae_errors.append(ae_error)
        if (i + 1) % 10 == 0:
            print(f'  - Procesadas {i + 1}/{len(healthy_signals)} muestras')

    # Configurar detector con errores de línea base
    detector.setup_baseline(healthy_signals, baseline_ae_errors)

    # Calibración del sistema
    print(f'[INFO] Calibrando sistema con {CALIBRATION_SAMPLES} audios adicionales...')
    calibration_scores = []
    for i in range(CALIBRATION_SAMPLES):
        print(f'[INFO] Grabando audio de calibración {i + 1}/{CALIBRATION_SAMPLES}...')
        sig = record_audio()
        X_proc = preprocess_signal(sig)[None, :].astype(np.float32)
        interpreter.set_tensor(inp_d['index'], X_proc)
        interpreter.invoke()
        recon = interpreter.get_tensor(out_d['index'])
        ae_error = np.mean((X_proc - recon) ** 2)

        score, details = detector.calculate_anomaly_score(sig, ae_error)
        calibration_scores.append(score)
        detector.error_history.append(score)

        print(f'  - Score: {score:.3f} | AE: {ae_error:.6f} | '
              f'Freq: {details["fundamental_freq"]:.1f}Hz | '
              f'RMS: {details["rms_energy"]:.1f}')

    cal_mean, cal_std, cal_max = np.mean(calibration_scores), np.std(calibration_scores), np.max(calibration_scores)
    print(f'[INFO] Calibración completa - Media: {cal_mean:.3f}, Std: {cal_std:.3f}, Max: {cal_max:.3f}')

    # Ajustar sensibilidad si los scores de calibración son muy altos
    if cal_mean > 0.3:
        print('[WARNING] Scores de calibración altos. Ajustando sensibilidad...')
        detector.detection_sensitivity = max(1.0, detector.detection_sensitivity - 0.3)
        print(f'[INFO] Nueva sensibilidad: {detector.detection_sensitivity:.1f}')

    # Sistema de monitorización
    print('\n' + '=' * 80)
    print('🔧 SISTEMA DE DETECCIÓN DE AVERÍAS - MODO MEJORADO')
    print('=' * 80)
    print('📊 Configuración:')
    print(f'   • Sensibilidad: {detector.detection_sensitivity:.1f}')
    print(f'   • Umbral dinámico: {detector.dynamic_threshold:.3f}')
    print(f'   • Intervalo de monitoreo: {args.monitor_interval}s')
    print(f'   • Características extraídas: {len(detector.baseline_stats["feature_names"])}')
    print('\n🎯 Umbrales de alerta:')
    print('   • CRÍTICO: >0.6 (Parar máquina)')
    print('   • ALTO: 0.4-0.6 (Mantenimiento urgente)')
    print('   • MEDIO: 0.25-0.4 (Inspección programada)')
    print('   • BAJO: 0.15-0.25 (Monitoreo intensivo)')
    print('\n[INFO] Presiona Ctrl+C para detener')
    print('=' * 80)

    measurement_count = 0
    anomaly_count = 0
    last_risk_level = "NORMAL"

    try:
        while True:
            measurement_count += 1
            print(f'\n🔍 --- MEDICIÓN #{measurement_count} ---')

            # Grabar y procesar audio
            sig = record_audio()
            X_proc = preprocess_signal(sig)[None, :].astype(np.float16)

            # Obtener error del autoencoder
            interpreter.set_tensor(inp_d['index'], X_proc)
            interpreter.invoke()
            recon = interpreter.get_tensor(out_d['index'])
            ae_error = np.mean((X_proc - recon) ** 2)

            # Calcular score de anomalía
            anomaly_score, details = detector.calculate_anomaly_score(sig, ae_error)
            detector.error_history.append(anomaly_score)

            # Detectar tendencias
            is_trending, trend_slope = detector.detect_trend()

            # Actualizar contador de anomalías consecutivas
            if anomaly_score > MIN_DETECTION_THRESHOLD:
                detector.consecutive_anomalies += 1
                anomaly_count += 1
            else:
                detector.consecutive_anomalies = 0

            # Predecir riesgo de falla
            risk_score, risk_level, action, risk_factors = detector.predict_failure_risk(
                anomaly_score, trend_slope
            )

            # Mostrar resultados principales
            print(f'📈 SCORE ANOMALÍA: {anomaly_score:.3f} (Umbral dinámico: {detector.dynamic_threshold:.3f})')
            print(f'🔧 ERROR AUTOENCODER: {ae_error:.6f}')

            # Mostrar detalles técnicos
            print(f'📊 SCORES INDIVIDUALES:')
            print(f'   • Isolation Forest: {details["isolation"]:.3f}')
            print(f'   • Elliptic Envelope: {details["elliptic"]:.3f}')
            print(f'   • Distancia Robusta: {details["robust_distance"]:.3f}')
            print(f'   • Autoencoder: {details["autoencoder"]:.3f}')
            print(f'   • Cambio Temporal: {details["temporal_change"]:.3f}')

            # Características clave del audio
            print(f'🎵 CARACTERÍSTICAS DEL AUDIO:')
            print(f'   • Freq. Fundamental: {details["fundamental_freq"]:.1f} Hz')
            print(f'   • Centroide Espectral: {details["spectral_centroid"]:.1f} Hz')
            print(f'   • Energía RMS: {details["rms_energy"]:.1f}')
            print(f'   • Factor de Cresta: {details["crest_factor"]:.2f}')
            print(f'   • Distorsión Armónica: {details["thd"]:.3f}')
            print(f'   • Periodicidad: {details["periodicity"]:.3f}')

            # Información de tendencias
            if is_trending:
                print(f'📈 TENDENCIA DETECTADA: Incremento de +{trend_slope:.4f} por medición')

            if detector.consecutive_anomalies > 0:
                print(f'⚡ ANOMALÍAS CONSECUTIVAS: {detector.consecutive_anomalies}')

            # Resultado principal
            risk_emoji = {
                "CRÍTICO": "🚨",
                "ALTO": "⚠️",
                "MEDIO": "📋",
                "BAJO": "👁️",
                "MÍNIMO": "✅",
                "NORMAL": "✅"
            }

            print(f'\n{risk_emoji.get(risk_level, "❓")} EVALUACIÓN DE RIESGO:')
            print(f'   🎯 NIVEL: {risk_level} ({risk_score:.2f})')
            print(f'   📋 ACCIÓN: {action}')

            if risk_factors:
                print(f'   🔍 FACTORES DETECTADOS:')
                for factor, weight in risk_factors:
                    print(f'      • {factor} (peso: {weight:.2f})')

            # Alertas especiales
            if risk_level == "CRÍTICO":
                print('\n' + '🚨' * 20)
                print('🚨 ALERTA CRÍTICA: REVISAR MÁQUINA INMEDIATAMENTE 🚨')
                print('🚨' * 20)
            elif risk_level == "ALTO" and last_risk_level != "ALTO":
                print('\n⚠️  NUEVA ALERTA: Programar mantenimiento urgente ⚠️')
            elif detector.consecutive_anomalies >= ANOMALY_CONSECUTIVE:
                print(f'\n⚠️  ATENCIÓN: {detector.consecutive_anomalies} anomalías consecutivas detectadas')

            # Estadísticas de sesión
            anomaly_rate = (anomaly_count / measurement_count) * 100
            print(f'\n📊 ESTADÍSTICAS DE SESIÓN:')
            print(f'   • Total mediciones: {measurement_count}')
            print(f'   • Anomalías detectadas: {anomaly_count} ({anomaly_rate:.1f}%)')
            print(f'   • Umbral dinámico actual: {detector.dynamic_threshold:.3f}')

            last_risk_level = risk_level
            print('─' * 60)
            time.sleep(args.monitor_interval)

    except KeyboardInterrupt:
        print('\n\n' + '=' * 80)
        print('🛑 MONITORIZACIÓN DETENIDA POR EL USUARIO')
        print('=' * 80)
        print(f'📊 RESUMEN FINAL:')
        print(f'   • Total de mediciones: {measurement_count}')
        print(f'   • Anomalías detectadas: {anomaly_count}')
        print(f'   • Tasa de anomalías: {(anomaly_count / measurement_count * 100):.1f}%')
        print(f'   • Último nivel de riesgo: {last_risk_level}')
        print(f'   • Umbral final: {detector.dynamic_threshold:.3f}')

        if anomaly_count > 0:
            print(f'\n💡 RECOMENDACIONES:')
            if anomaly_count / measurement_count > 0.3:
                print('   • Alta tasa de anomalías detectada')
                print('   • Considerar inspección de la máquina')
                print('   • Revisar condiciones de operación')
            elif anomaly_count / measurement_count > 0.1:
                print('   • Anomalías moderadas detectadas')
                print('   • Continuar monitoreo frecuente')
                print('   • Programar mantenimiento preventivo')
            else:
                print('   • Pocas anomalías detectadas')
                print('   • Sistema funcionando generalmente bien')
                print('   • Mantener monitoreo regular')

        print('\n✅ Sistema finalizado correctamente')


if __name__ == '__main__':
    main()
