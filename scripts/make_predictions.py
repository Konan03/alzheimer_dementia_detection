import pandas as pd
import joblib

# Cargar el modelo entrenado
model = joblib.load('../models/random_forest_model.pkl')

# Cargar los datos de prueba
test_features = pd.read_csv('../data/test_features_cleaned.csv')

# Asegurarnos de que las columnas categóricas estén codificadas igual que en el entrenamiento
categorical_columns = ['glob_hlth_03', 'glob_hlth_12', 'memory_12']  # Asegúrate de incluir todas las columnas categóricas

# Aplicar codificación a las columnas categóricas
for col in categorical_columns:
    if col in test_features.columns:
        test_features[col] = test_features[col].astype('category').cat.codes

# Guardar 'uid' por separado si está presente
if 'uid' in test_features.columns:
    uid_column = test_features['uid']
    X_test = test_features.drop(columns=['uid'])
else:
    uid_column = None
    X_test = test_features

# Verificar que las columnas coincidan con las del modelo entrenado
required_columns = model.feature_names_in_  # Extrae las columnas que el modelo espera
X_test = X_test[required_columns]  # Filtrar las columnas relevantes

# Generar predicciones
predictions = model.predict(X_test)

# Crear un DataFrame con las predicciones
if uid_column is not None:
    output = pd.DataFrame({'uid': uid_column, 'composite_score': predictions})
else:
    output = pd.DataFrame({'composite_score': predictions})

# Guardar predicciones en un archivo CSV
output.to_csv('../submission/submission_predictions.csv', index=False)

print("Predicciones guardadas en '../submission/submission_predictions.csv'")
