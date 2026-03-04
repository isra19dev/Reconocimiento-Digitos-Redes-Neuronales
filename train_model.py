import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

#? Carga de datos y se formatea a 8x8
digits = load_digits()

#! Imágenes de los números
X = digits.data
#! Etiquetas de los números (0-9)
y = digits.target  

#? Normalización los datos para que estén entre 0 y 1
X = X / 16.0 

#? Dividimos los datos en conjunto de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

#? Creación del modelo
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

#? Entrenamos el modelo
print("Entrenando el modelo...")
model.fit(X_train, y_train)

#? Evaluamos el modelo en el conjunto de prueba
score = model.score(X_test, y_test)
print(f"Precisión del modelo: {score * 100:.2f}%")

#? Guardamos el modelo en formato joblib
model_path = 'modelo_digitos.joblib'
joblib.dump(model, model_path)
print(f"Modelo guardado en: {model_path}")

#? Información del modelo
print(f"\nInformación del modelo:")
print(f"- Número de características de entrada: {model.n_features_in_}")
print(f"- Número de clases (dígitos): {len(np.unique(y))}")
print(f"- Capas del modelo: {model.hidden_layer_sizes}")
