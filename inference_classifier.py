import pickle
import cv2
import mediapipe as mp
import numpy as np

# Carga del modelo entrenado
model_dict = pickle.load(open("./model.p", 'rb'))
model = model_dict['model']

# Configuración de la captura de video
cap = cv2.VideoCapture(0)

# Inicialización de MediaPipe para la detección de manosS
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuración de MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.75)

# Diccionario de etiquetas para clasificar las letras
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'I', 7: 'K', 8: 'L', 9: 'M', 10: 'N', 11: 'O', 12: 'P', 13: 'Q', 14: 'R', 15: 'T', 16: 'U', 17: 'V', 18: 'W', 19: 'X', 20: 'Y'}

# Variable para almacenar la frase
phrase = ""

def reset_camera():
    """Reinicia la interfaz de la cámara."""
    cap.release()
    cv2.destroyAllWindows()
    cap.open(0)

while True:
    data_aux = []  # Lista para almacenar las coordenadas de la mano procesada
    x_ = []  # Lista para almacenar las coordenadas x normalizadas
    y_ = []  # Lista para almacenar las coordenadas y normalizadas

    # Captura del cuadro actual de la cámara
    ret, frame = cap.read()
    H, W, _ = frame.shape

    # Conversión de la imagen de BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesamiento del cuadro actual para detectar manos
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Dibuja las conexiones de la mano en la imagen original
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Recolecta las coordenadas de la mano detectada
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normaliza las coordenadas y las agrega a la lista data_aux
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Calcula las coordenadas para el rectángulo que encierra la mano
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            # Realiza la predicción del modelo utilizando las coordenadas normalizadas
            prediction = model.predict([np.asarray(data_aux)])

            # Obtiene la letra predicha a partir de la predicción del modelo
            predicted_character = labels_dict[int(prediction[0])]

            # Dibuja un rectángulo y la letra predicha en el cuadro original
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
        except ValueError as e:
            print(f"Error: {e}. Reiniciando cámara.")
            reset_camera()
            continue

    # Muestra el cuadro de la cámara
    cv2.imshow('Camera Frame', frame)

    # Detección de teclas presionadas
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):  # Tecla 'a' para agregar la letra identificada
        phrase += predicted_character
    elif key == ord(' '):  # Tecla de espacio para agregar un espacio
        phrase += ' '
    elif key == 8:  # Tecla 'Backspace' (borrar) para eliminar la última letra agregada
        phrase = phrase[:-1]
    elif key == 27:  # Tecla 'ESC' para salir
        break

    # Crea una imagen en blanco para mostrar la frase construida
    phrase_frame = np.ones((200, 800, 3), dtype=np.uint8) * 255
    cv2.putText(phrase_frame, phrase, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow('Phrase', phrase_frame)

# Liberación de recursos
cap.release()
cv2.destroyAllWindows()
