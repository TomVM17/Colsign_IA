import pickle
import numpy as np
from sklearn.metrics import accuracy_score

# Carga del modelo
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Carga del diccionario de datos
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Realiza predicciones con los datos de prueba
y_predict = model.predict(data)

# Calcula la precisión para cada letra
accuracy_per_letter = {}
for i, label in enumerate(labels):
    if label not in accuracy_per_letter:
        accuracy_per_letter[label] = {'correct': 0, 'total': 0}
    accuracy_per_letter[label]['total'] += 1
    if label == y_predict[i]:
        accuracy_per_letter[label]['correct'] += 1

# Imprime la precisión para cada letra
for letter, stats in accuracy_per_letter.items():
    accuracy = stats['correct'] / stats['total']
    print(f"Letra {letter}: Precisión = {accuracy:.2f}")
