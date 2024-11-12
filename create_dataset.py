import os  # Importa el módulo os para interactuar con el sistema operativo
import pickle  # Importa el módulo pickle para serializar y deserializar objetos de Python

import mediapipe as mp  # Importa MediaPipe para la detección de manos
import cv2  # Importa OpenCV para la manipulación de imágenes
import matplotlib.pyplot as plt  # Importa Matplotlib para la visualización de datos

# Inicializa los componentes de MediaPipe para la detección y dibujo de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configura la detección de manos con MediaPipe
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.45)

# Define el directorio de datos
DATA_DIR = './data'

# Inicializa las listas para almacenar los datos de las imágenes y sus etiquetas
data = []
labels = []

# Itera sobre los directorios en DATA_DIR (cada directorio representa una clase)
for dir_ in os.listdir(DATA_DIR):
    # Itera sobre las imágenes en cada directorio de clase
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Lista auxiliar para almacenar las coordenadas de la mano

        x_ = []  # Lista para almacenar las coordenadas x de los puntos de la mano
        y_ = []  # Lista para almacenar las coordenadas y de los puntos de la mano

        # Lee la imagen utilizando OpenCV
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # Convierte la imagen de BGR a RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Procesa la imagen para detectar manos
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:  # Si se detectan manos
            # Itera sobre las manos detectadas
            for hand_landmarks in results.multi_hand_landmarks:
                # Itera sobre los puntos de referencia de la mano
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)  # Almacena la coordenada x
                    y_.append(y)  # Almacena la coordenada y

                # Calcula las coordenadas normalizadas respecto al mínimo valor
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normaliza x y la almacena
                    data_aux.append(y - min(y_))  # Normaliza y y la almacena

            data.append(data_aux)  # Almacena los datos de la mano
            labels.append(dir_)  # Almacena la etiqueta correspondiente a la clase

# Abre un archivo para escribir los datos serializados
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)  # Serializa los datos y etiquetas
