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
   - fastapi
   - uvicorn
   - websockets
     ```bash
     pip install fastapi uvicorn websockets
     ```
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

5. **Predicción en Tiempo Real-Web:**
   - El script `classifier_web.py` utiliza el modelo entrenado para predecir los gestos de la mano capturados en tiempo real por la cámara, mostrando los caracteres predichos en pantalla pero en una web.

# Documentación para el Usuario (Mínimo Producto Viable)

## Descripción General
Este producto permite reconocer gestos de las manos a través de una cámara web y traducirlos en caracteres en tiempo real. Utiliza técnicas de aprendizaje automático y visión por computadora para detectar y clasificar los gestos de manera eficiente.

**Nota:** El traductor de señas actualmente es capaz de traducir solo algunas letras del abecedario, específicamente: **A, B, C, D, E, F, I, K, L, M, N, O, P, Q, R, T, U, V, W, X, Y**.
Su función es deletrear palabras con las letras existentes en el modelo.

## Requisitos del Sistema
- Una computadora con una cámara web.
- Python 3.7 o superior.
- Librerías necesarias: OpenCV, MediaPipe, NumPy, pickle, scikit-learn, fastapi, uvicorn.

## Instalación y Configuración

1. **Instalar Python:**
   - Asegúrese de tener **Python 3.7** o superior instalado en su sistema. Puede descargar e instalar Python desde [python.org](https://www.python.org/).

2. **Instalar Dependencias:**
   - Abra una terminal y ejecute el siguiente comando para instalar las librerías necesarias:
     ```bash
     pip install opencv-python mediapipe numpy scikit-learn
     ```

### Predicción en Tiempo Real:
Para predicción en tiempo real, existen dos opciones:
1. **Localmente**: Ejecutar el script `inference_classifier.py`.
2. **Web**: Ejecutar el script `classifier_web.py`, que inicia un servidor WebSocket mediante FastAPI para recibir fotogramas y enviar predicciones en tiempo real.

## Uso del Producto

1. **Recolectar Datos (Opcional):**
   - Si necesita recolectar datos de gestos personalizados, ejecute el script:
     ```bash
     python collect_imgs.py
     ```
   - Coloque las imágenes de los gestos capturados en el directorio `data/`, organizadas por clases (gestos).

2. **Procesar Datos:**
   - Ejecute el script `create_datasets.py` para procesar las imágenes y extraer las coordenadas clave de las manos:
     ```bash
     python create_datasets.py
     ```

3. **Entrenar el Modelo:**
   - Entrene el modelo de clasificación de gestos ejecutando el script:
     ```bash
     python train_classifier.py
     ```

4. **Ejecutar el Sistema de Predicción en Tiempo Real:**
   - **Si desea usar el modelo de manera local**: Ejecute el siguiente comando:
     ```bash
     python inference_classifier.py
     ```
   - **Si desea usar el modelo a través de la web**: Inicie el servidor WebSocket ejecutando:
     ```bash
     python classifier_web.py
     ```

## Instrucciones para el Usuario

1. **Iniciar el Sistema:**
   - Coloque su mano frente a la cámara, asegurándose de tener buena iluminación para mejorar la detección.
   - El sistema comenzará a capturar video en tiempo real, mostrando la detección de manos y gestos en pantalla.

2. **Interacción del Usuario:**
   - Cuando el sistema detecte un gesto, mostrará el carácter correspondiente en pantalla.
   - Use las siguientes teclas para interactuar:
     - Presione `a` para agregar el carácter detectado a una frase.
     - Presione `espacio` para agregar un espacio en blanco a la frase.
     - Presione `ESC` para salir del programa.

3. **Visualización:**
   - El sistema mostrará dos ventanas:
     - Una ventana con el video en tiempo real y las detecciones de manos.
     - Otra ventana que muestra la frase construida a partir de los gestos detectados.



