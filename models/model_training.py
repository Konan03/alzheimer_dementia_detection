import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import joblib

# Cargar los datos limpios y etiquetas
train_features_cleaned = pd.read_csv('../data/train_features_cleaned.csv')
train_labels = pd.read_csv('../data/train_labels.csv')

# Filtrar etiquetas para incluir solo los identificadores presentes en las características
train_labels = train_labels[train_labels['uid'].isin(train_features_cleaned['uid'])]

# Filtrar características para incluir solo los identificadores presentes en las etiquetas
train_features_cleaned = train_features_cleaned[train_features_cleaned['uid'].isin(train_labels['uid'])]

# Ordenar ambas tablas por 'uid' para asegurar la alineación
train_features_cleaned = train_features_cleaned.sort_values(by='uid').reset_index(drop=True)
train_labels = train_labels.sort_values(by='uid').reset_index(drop=True)

# Validar que las filas coincidan
if len(train_features_cleaned) != len(train_labels):
    print(f"Warning: Mismatched rows after filtering. Features: {len(train_features_cleaned)}, Labels: {len(train_labels)}")
    # Ajustar etiquetas al número de características
    train_labels = train_labels.iloc[:len(train_features_cleaned)]
else:
    print(f"Features and labels are aligned. Total rows: {len(train_features_cleaned)}")

# Eliminar la columna 'uid' ya que no es necesaria para el modelo
train_features_cleaned = train_features_cleaned.drop(columns=['uid'])
y = train_labels['composite_score']

# Preprocesar columnas categóricas
categorical_columns = train_features_cleaned.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    print(f"Encoding categorical columns: {list(categorical_columns)}")
    encoder = OrdinalEncoder()
    train_features_cleaned[categorical_columns] = encoder.fit_transform(train_features_cleaned[categorical_columns])

# Separar las características (X) y el objetivo (y)
X = train_features_cleaned

# Dividir los datos en conjunto de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de Random Forest
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f'Mean Squared Error: {mse}')

# Guardar el modelo entrenado
joblib.dump(model, '../models/random_forest_model.pkl')
print("Modelo guardado en '../models/random_forest_model.pkl'")
