import os
import base64
import io
import numpy as np
import librosa
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# ----------------------------------------------------------------------
# 1) CARGAR VARIABLES DE ENTORNO
# ----------------------------------------------------------------------
load_dotenv()  # busca un .env en el mismo directorio

MYSQL_HOST     = os.getenv("DB_HOST")
MYSQL_USER     = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
MYSQL_NAME     = os.getenv("DB_NAME")

# ----------------------------------------------------------------------
# 2) PARÁMETROS DE EXTRACCIÓN DE CARACTERÍSTICAS
# ----------------------------------------------------------------------
N_MFCC = 13  # número de MFCCs a extraer
# Cada muestra final tendrá 2*N_MFCC características (media + desviación de cada coeficiente)

# ----------------------------------------------------------------------
# 3) RUTA DONDE GUARDAR EL MODELO ENTRENADO
# ----------------------------------------------------------------------
OUTPUT_MODEL_PATH = "voting_classifier_validated.joblib"


# ----------------------------------------------------------------------
# 4) FUNCIÓN: CONEXIÓN A LA BASE DE DATOS
# ----------------------------------------------------------------------
def get_connection():
    """
    Retorna un objeto mysql.connector.connect() usando variables de entorno.
    """
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_NAME
    )


# ----------------------------------------------------------------------
# 5) FUNCIÓN: EXTRAER MFCCS (media + desviación) DESDE bytes WAV EN MEMORIA
# ----------------------------------------------------------------------
def extract_mfcc_features_from_bytes(wav_bytes: bytes, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Dado un buffer de bytes que contiene un WAV válido, carga con librosa y extrae:
      - n_mfcc coeficientes MFCC
      - devuelve un vector de 2*n_mfcc: [media_0, ..., media_{n_mfcc-1}, std_0, ..., std_{n_mfcc-1}]
    Si hay cualquier error, retorna un vector de ceros de longitud 2*n_mfcc.
    """
    try:
        # Cargar con librosa desde un BytesIO
        bio = io.BytesIO(wav_bytes)
        y, sr = librosa.load(bio, sr=None)  # conservamos la tasa original del WAV
        if y is None or len(y) == 0:
            raise ValueError("Señal vacía o cargada incorrectamente")

        # Extraer MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        features = np.concatenate([mfccs_mean, mfccs_std]).astype(np.float32)
        return features

    except Exception as e:
        print(f"[WARNING] No se pudieron extraer MFCCs: {e}")
        return np.zeros(2 * n_mfcc, dtype=np.float32)


# ----------------------------------------------------------------------
# 6) FUNCIÓN: OBTENER DATOS VALIDADOS DE LA BD (estado = 'Arreglada')
# ----------------------------------------------------------------------
def fetch_validated_rows():
    """
    Consulta en la tabla 'alert' (o 'alertas') todas las filas cuyo estado = 'Arreglada'.
    Retorna una lista de tuplas (alert_type, audio_record_base64).
    """
    query = """
        SELECT alert_type, audio_record
        FROM alert
        WHERE estado = 'Arreglada'
          AND audio_record IS NOT NULL
          AND alert_type <> ''
    """
    rows = []
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()  # cada fila es (alert_type, audio_record)
    except mysql.connector.Error as err:
        print(f"[ERROR] Al consultar la base de datos: {err}")
    finally:
        if conn:
            conn.close()
    return rows  # lista de tuplas


# ----------------------------------------------------------------------
# 7) MAIN: ARMAR X, y Y ENTRENAR MODELO
# ----------------------------------------------------------------------
def main():
    print("=== Retraining con audios validados (estado = 'Arreglada') ===")

    # 7.A) Obtener filas validadas
    validated = fetch_validated_rows()
    if not validated:
        print("[INFO] No se encontraron alertas validadas (estado = 'Arreglada'). No se entrena.")
        return

    print(f"[INFO] Encontradas {len(validated)} muestras validadas.")

    # 7.B) Recorrer cada fila y extraer características + etiqueta
    X_list = []
    y_list = []
    for idx, (alert_type, audio_b64) in enumerate(validated, start=1):
        # 1) Decodificar Base64
        try:
            wav_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            print(f"[WARNING] Fila {idx}: No se pudo decodificar Base64: {e}. Se omite.")
            continue

        # 2) Extraer características MFCC
        feats = extract_mfcc_features_from_bytes(wav_bytes, n_mfcc=N_MFCC)
        X_list.append(feats)
        y_list.append(alert_type)

    if len(X_list) == 0:
        print("[INFO] No se generaron características. Termina el script.")
        return

    # Convertir a matrices numpy
    X = np.vstack(X_list)  # forma: (n_muestras, 2*N_MFCC)
    y = np.array(y_list)   # forma: (n_muestras,)

    print(f"[INFO] Matriz X: {X.shape}, Vector y: {y.shape}")
    print("[INFO] Distribución de etiquetas:\n", pd.Series(y).value_counts())

    # 7.C) Dividir en train / test (opcional: aquí reentrenamos usando todas las muestras validadas,
    #      pero si quieres hacer validación interna, puedes usar train_test_split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Entrenando con {X_train.shape[0]} muestras, probando con {X_test.shape[0]} muestras")

    # 7.D) Definir clasificadores base
    rf_clf  = RandomForestClassifier(n_estimators=100, random_state=42)
    svc_clf = SVC(kernel="rbf", probability=True, random_state=42)
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[
            ("rf", rf_clf),
            ("svc", svc_clf),
            ("sgd", sgd_clf),
        ],
        voting="soft",  # usamos soft voting para aprovechar predict_proba
        n_jobs=-1
    )

    # 7.E) Entrenar VotingClassifier
    print("[INFO] Iniciando entrenamiento del VotingClassifier...")
    voting_clf.fit(X_train, y_train)

    # 7.F) Evaluar en test
    print("[INFO] Evaluando en el conjunto de prueba:")
    y_pred = voting_clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=voting_clf.classes_)
    cm_df = pd.DataFrame(cm, index=voting_clf.classes_, columns=voting_clf.classes_)
    print("=== Matriz de Confusión ===")
    print(cm_df)

    # 7.G) Guardar el modelo entrenado a disco
    try:
        joblib.dump(voting_clf, OUTPUT_MODEL_PATH)
        print(f"[INFO] Modelo guardado en: {OUTPUT_MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el modelo en disco: {e}")


if __name__ == "__main__":
    main()
