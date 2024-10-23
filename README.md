# Documentación Técnica (Para el Sistema/Desarrollador)

## Descripción General
Este sistema implementa un modelo de clasificación basado en **Random Forest** que reconoce gestos de las manos capturados en tiempo real por una cámara, y los traduce en caracteres. Utiliza **MediaPipe** para la detección de manos y **OpenCV** para la captura y procesamiento de video.

## Requisitos

1. **Librerías:**
   - `pickle`
   - `cv2` (OpenCV)
   - `mediapipe`
   - `numpy`
   - `scikit-learn`

2. **Estructura de Archivos:**
   - `data/` (directorio que contiene las imágenes de entrenamiento organizadas por clases)
   - `data.pickle` (archivo que contiene los datos y etiquetas de entrenamiento serializados)
   - `model.p` (archivo que contiene el modelo entrenado)

## Configuración del Sistema

1. **Recolección de Datos:**
   - El script `collect_imgs.py` captura imágenes desde la cámara y las guarda en el directorio `data/`, organizadas por clases.

2. **Procesamiento de Datos:**
   - El script `create_datasets.py` utiliza **MediaPipe** para detectar manos en las imágenes, extraer las coordenadas de los puntos clave, normalizarlas y almacenarlas junto con sus etiquetas en `data.pickle`.

3. **Entrenamiento del Modelo:**
   - El script `train_classifier.py` carga los datos desde `data.pickle`, entrena un modelo **Random Forest**, y guarda el modelo entrenado en `model.p`.

4. **Predicción en Tiempo Real:**
   - El script `inference_classifier.py` utiliza el modelo entrenado para predecir los gestos de la mano capturados en tiempo real por la cámara, mostrando los caracteres predichos en pantalla.
