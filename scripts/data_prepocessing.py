import pandas as pd

# Cargar los archivos CSV
train_features = pd.read_csv('../data/train_features.csv')
train_labels = pd.read_csv('../data/train_labels.csv')
test_features = pd.read_csv('../data/test_features.csv')

# Paso 1: Verificar y mantener 'uid' para filtrar registros comunes
common_uids = set(train_labels['uid']).intersection(set(train_features['uid']))
print(f"Número de 'uid' comunes entre características y etiquetas: {len(common_uids)}")

# Filtrar solo los registros que tienen 'uid' en común
train_features = train_features[train_features['uid'].isin(common_uids)]
train_labels = train_labels[train_labels['uid'].isin(common_uids)]

# Paso 2: Identificar y manejar valores faltantes
numeric_cols = train_features.select_dtypes(include=['number']).columns
categorical_cols = train_features.select_dtypes(include=['object']).columns

# Imputar valores faltantes en columnas numéricas con la mediana
train_features[numeric_cols] = train_features[numeric_cols].fillna(train_features[numeric_cols].median())
test_features[numeric_cols] = test_features[numeric_cols].fillna(test_features[numeric_cols].median())

# Imputar valores faltantes en columnas categóricas con la moda
train_features[categorical_cols] = train_features[categorical_cols].fillna(train_features[categorical_cols].mode().iloc[0])
test_features[categorical_cols] = test_features[categorical_cols].fillna(test_features[categorical_cols].mode().iloc[0])

# Paso 3: Codificar variables categóricas con One-Hot Encoding, manteniendo 'uid'
train_features_encoded = pd.get_dummies(train_features, columns=categorical_cols)
test_features_encoded = pd.get_dummies(test_features, columns=categorical_cols)

# Alinear los conjuntos de datos para asegurar que tengan las mismas columnas
train_features_encoded, test_features_encoded = train_features_encoded.align(test_features_encoded, join='left', axis=1, fill_value=0)

# Paso 4: Guardar los datos procesados, manteniendo 'uid'
train_features_encoded.to_csv('../data/train_features_encoded.csv', index=False)
test_features_encoded.to_csv('../data/test_features_encoded.csv', index=False)
train_labels.to_csv('../data/train_labels_processed.csv', index=False)

print("Preprocesamiento completo. Los datos procesados se han guardado en la carpeta 'data'.")
