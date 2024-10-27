import pandas as pd

# Cargar los archivos CSV con rutas relativas correctas
train_features = pd.read_csv('../data/train_features.csv')
train_labels = pd.read_csv('../data/train_labels.csv')
test_features = pd.read_csv('../data/test_features.csv')
submission_format = pd.read_csv('../data/submission_format.csv')

# Ver las primeras filas de cada archivo
print("Train Features:")
print(train_features.head())
print("\nTrain Labels:")
print(train_labels.head())
print("\nTest Features:")
print(test_features.head())
print("\nSubmission Format:")
print(submission_format.head())

# Revisar información básica de los datasets
print("\nInformación de Train Features:")
print(train_features.info())
print("\nInformación de Train Labels:")
print(train_labels.info())
print("\nInformación de Test Features:")
print(test_features.info())
