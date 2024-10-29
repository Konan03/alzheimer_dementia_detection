import joblib
import pandas as pd

# Cargar el modelo entrenado
model = joblib.load('models/random_forest_model.pkl')

# Cargar los datos de prueba
test_features_encoded = pd.read_csv('data/test_features_encoded.csv')

# Asegurarse de que los datos de prueba no incluyan la columna 'uid' antes de predecir
test_data = test_features_encoded.drop(columns=['uid'], errors='ignore')

# Realizar predicciones
predictions = model.predict(test_data)
print(predictions)
