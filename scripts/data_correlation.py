import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos limpios
train_data = pd.read_csv('../data/train_features_cleaned.csv')
test_data = pd.read_csv('../data/test_features_cleaned.csv')

# Filtrar solo las columnas numéricas
train_data_numeric = train_data.select_dtypes(include=['float64', 'int64'])

# Generar la matriz de correlación
correlation_matrix = train_data_numeric.corr().abs()

# Mostrar la matriz de correlación como tabla en la consola
print("Matriz de correlación:")
print(correlation_matrix)

# Visualizar la matriz de correlación con un mapa de calor mejorado
plt.figure(figsize=(40, 20))  # Aumentar el tamaño de la figura
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", annot_kws={"size": 8}, cbar_kws={"shrink": 0.75})
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Matriz de Correlación")
plt.show()

# Definir el umbral de correlación
threshold = 0.9

# Identificar columnas con alta correlación
high_corr_var = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if correlation_matrix.iloc[i, j] > threshold:
            colname = correlation_matrix.columns[i]
            high_corr_var.add(colname)

# Eliminar las columnas con alta correlación en ambos conjuntos
train_data_reduced = train_data.drop(columns=high_corr_var)
test_data_reduced = test_data.drop(columns=high_corr_var)

# Guardar los nuevos archivos reducidos
train_data_reduced.to_csv('../data/train_features_reduced.csv', index=False)
test_data_reduced.to_csv('../data/test_features_reduced.csv', index=False)

print("Archivos 'train_features_reduced.csv' y 'test_features_reduced.csv' guardados en la carpeta 'data'.")
