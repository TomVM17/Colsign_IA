import os  # Importa el módulo os para interactuar con el sistema operativo
import cv2  # Importa el módulo cv2 para trabajar con OpenCV

# Define el directorio donde se almacenarán los datos
DATA_DIR = './data'

# Si el directorio no existe, lo crea
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define el número de clases y el tamaño del dataset por clase
number_of_classes = 21
dataset_size = 1000

# Inicia la captura de video desde la cámara (índice 0)
cap = cv2.VideoCapture(0)

# Itera sobre el número de clases
for j in range(number_of_classes):
    # Crea un directorio para cada clase si no existe
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Recogiendo datos para la clase {}'.format(j))

    # Bandera para controlar la captura de datos
    done = False

    # Bucle para esperar hasta que el usuario esté listo
    while True:
        ret, frame = cap.read()  # Lee un frame de la cámara
        # Muestra un mensaje en el frame para indicar que presione "Q" cuando esté listo
        cv2.putText(frame, 'Listo? Presiona "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Muestra el frame en una ventana llamada 'frame'
        if cv2.waitKey(25) == ord('q'):  # Espera a que el usuario presione 'q' para continuar
            break

    counter = 0  # Inicializa el contador de imágenes

    # Bucle para capturar el número especificado de imágenes para la clase actual
    while counter < dataset_size:
        ret, frame = cap.read()  # Lee un frame de la cámara
        cv2.imshow('frame', frame)  # Muestra el frame en una ventana llamada 'frame'
        cv2.waitKey(25)  # Espera 25 milisegundos entre capturas
        # Guarda el frame actual como una imagen en el directorio correspondiente a la clase
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1  # Incrementa el contador de imágenes

# Libera la captura de video y cierra todas las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
