import pandas as pd
import joblib

# Cargar el modelo entrenado
model = joblib.load('../models/random_forest_model.pkl')

# Cargar los datos de prueba
test_features = pd.read_csv('../data/test_features_cleaned.csv')

# Guardar uid por separado
uid_column = test_features['uid']
X_test = test_features.drop(columns=['uid'], errors='ignore')

# Generar predicciones
predictions = model.predict(X_test)

# Mapear predicciones con uid
output = pd.DataFrame({'uid': uid_column, 'composite_score': predictions})
output.to_csv('../data/submission_predictions.csv', index=False)

print("Predicciones guardadas en '../data/submission_predictions.csv'")
