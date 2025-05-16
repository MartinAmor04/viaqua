# 🎧 Autoencoder para Detección de Anomalías Acústicas

Este proyecto implementa un sistema de entrenamiento y predicción en tiempo real con un autoencoder que detecta anomalías acústicas usando espectrogramas mel.

## 📋 Descripción

El script graba muestras de audio, las convierte en espectrogramas mel normalizados y entrena un autoencoder hasta que el error de validación cae por debajo de un umbral. Luego, usa el modelo para predecir errores en nuevas muestras y estima el nivel de "daño" acústico.

---

## ⚙️ Configuración

Parámetros definidos en el script:

| Parámetro       | Descripción                                        |
|-----------------|----------------------------------------------------|
| `N_MELS`        | Número de bandas mel en el espectrograma.         |
| `FIXED_FRAMES`  | Número fijo de frames en el eje temporal.         |
| `DURATION`      | Duración de la grabación (en segundos).           |
| `SAMPLING_RATE` | Frecuencia de muestreo del audio.                 |
| `BATCH_AUDIOS`  | Tamaño del lote de muestras por iteración.        |

---

## 🧩 Funciones

### `preprocess_signal(signal: np.ndarray) -> np.ndarray`

Convierte una señal de audio en un espectrograma mel normalizado y de tamaño fijo.

- **Parámetros:**  
  `signal`: señal de audio en un array de NumPy.

- **Retorna:**  
  Espectrograma mel normalizado como `np.ndarray`.

---

### `record_audio() -> np.ndarray`

Graba una muestra de audio de duración fija usando `arecord`.

- **Retorna:**  
  Señal de audio como array de enteros de 32 bits (`np.int32`).

---

### `autoencoder_model(input_dim: int) -> Model`

Crea y compila un autoencoder simétrico con cuello de botella.

- **Parámetros:**  
  `input_dim`: dimensión del vector de entrada (espectrograma aplanado).

- **Retorna:**  
  Modelo Keras compilado (`tensorflow.keras.models.Model`).

---

### `main()`

Función principal que:
1. Entrena el autoencoder por lotes hasta que `val_loss` < `threshold`.
2. Inicia predicción en tiempo real sobre nuevas muestras.
3. Calcula el error y estima el "daño" como un porcentaje relativo.

---

## 🚀 Ejecución

```bash
python script.py --batch_size 16 --threshold 0.1
