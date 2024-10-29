import pandas as pd

# Cargar los archivos CSV
train_features = pd.read_csv('../data/train_features.csv')
train_labels = pd.read_csv('../data/train_labels.csv')
test_features = pd.read_csv('../data/test_features.csv')

# Paso 1: Encontrar los 'uid' comunes en ambos conjuntos de datos
common_uids = set(train_features['uid']).intersection(set(train_labels['uid']))

# Filtrar ambos DataFrames para conservar solo los 'uid' comunes
train_features = train_features[train_features['uid'].isin(common_uids)].reset_index(drop=True)
train_labels = train_labels[train_labels['uid'].isin(common_uids)].reset_index(drop=True)

# Eliminar duplicados en train_labels promediando el composite_score por uid
train_labels = train_labels.groupby('uid', as_index=False).agg({'composite_score': 'mean'})

# Verificar alineación después del filtrado y agregación
print(f"Número de registros en train_features después del ajuste: {len(train_features)}")
print(f"Número de registros en train_labels después del ajuste: {len(train_labels)}")
assert len(train_features) == len(train_labels), "Error: Los conjuntos no están alineados."

# Paso 2: Imputar valores faltantes en columnas numéricas y categóricas
numeric_cols = train_features.select_dtypes(include=['number']).columns
categorical_cols = train_features.select_dtypes(include=['object']).columns

train_features[numeric_cols] = train_features[numeric_cols].fillna(train_features[numeric_cols].median())
test_features[numeric_cols] = test_features[numeric_cols].fillna(test_features[numeric_cols].median())
train_features[categorical_cols] = train_features[categorical_cols].fillna(train_features[categorical_cols].mode().iloc[0])
test_features[categorical_cols] = test_features[categorical_cols].fillna(test_features[categorical_cols].mode().iloc[0])

# Paso 3: Codificar variables categóricas con One-Hot Encoding
train_features_encoded = pd.get_dummies(train_features, columns=categorical_cols)
test_features_encoded = pd.get_dummies(test_features, columns=categorical_cols)

# Alinear columnas de train y test después de la codificación
train_features_encoded, test_features_encoded = train_features_encoded.align(test_features_encoded, join='left', axis=1, fill_value=0)

# Guardar los datos procesados
train_features_encoded.to_csv('../data/train_features_encoded.csv', index=False)
test_features_encoded.to_csv('../data/test_features_encoded.csv', index=False)
train_labels.to_csv('../data/train_labels_processed.csv', index=False)

print("Preprocesamiento completo. Los datos procesados se han guardado en la carpeta 'data'.")
