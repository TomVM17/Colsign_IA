from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import pickle
import cv2
import mediapipe as mp
import numpy as np

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo y el diccionario de etiquetas
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'I', 7: 'K', 8: 'L', 9: 'M', 
    10: 'N', 11: 'O', 12: 'P', 13: 'Q', 14: 'R', 15: 'T', 16: 'U', 17: 'V', 18: 'W', 
    19: 'X', 20: 'Y'
}

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.75)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    phrase = ""  # Frase acumulada

    try:
        while True:
            # Recibir fotograma desde el WebSocket
            data = await websocket.receive_bytes()
            frame = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text("Error: Fotograma no válido")
                continue

            data_aux = []
            x_, y_ = [], []

            # Procesar el fotograma
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS, 
                        mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
                    )

                    # Obtener las coordenadas de la mano detectada
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        x_.append(landmark.x)
                        y_.append(landmark.y)

                    # Normalizar las coordenadas
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

                try:
                    # Realizar la predicción
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    # Enviar el resultado de la predicción
                    await websocket.send_text(predicted_character)

                except Exception as e:
                    print(f"Error en la predicción: {e}")
                    await websocket.send_text("Error en la predicción")

            else:
                await websocket.send_text("No se detecta mano")

    except WebSocketDisconnect:
        print("Cliente desconectado")
