import pickle  # Importa el módulo pickle para cargar y guardar objetos serializados
import numpy as np  # Importa NumPy para manejar arreglos numéricos

from sklearn.ensemble import RandomForestClassifier  # Importa el clasificador RandomForest de scikit-learn
from sklearn.model_selection import train_test_split  # Importa la función para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.metrics import accuracy_score  # Importa la función para calcular la precisión del modelo

# Carga el diccionario de datos desde el archivo pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Imprime información sobre los datos cargados
print("Tipo de dato de data_dict['data']: ", type(data_dict['data']))
print("Longitud de data_dict['data']: ", len(data_dict['data']))
print("Forma de cada elemento en data_dict['data']: ", [np.array(d).shape for d in data_dict['data']])

# Convierte los datos y etiquetas a arreglos de NumPy
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Divide los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inicializa el modelo RandomForestClassifier
model = RandomForestClassifier()

# Entrena el modelo con los datos de entrenamiento
model.fit(x_train, y_train)

# Realiza predicciones con los datos de prueba
y_predict = model.predict(x_test)

# Calcula la precisión del modelo
score = accuracy_score(y_predict, y_test)

# Imprime el porcentaje de muestras correctamente clasificadas
print('{}% de las muestras fueron clasificadas correctamente!'.format(score * 100))

# Guarda el modelo entrenado en un archivo pickle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
