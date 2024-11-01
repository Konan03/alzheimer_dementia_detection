import pandas as pd

# Cargar los datos
train_features = pd.read_csv('../data/train_features.csv')
test_features = pd.read_csv('../data/test_features.csv')

# Lista de columnas importantes para la predicción
relevant_columns = [
    'uid', 'age_03', 'age_12', 'urban_03', 'urban_12', 'edu_gru_03', 'edu_gru_12',
    'glob_hlth_03', 'glob_hlth_12', 'adl_dress_03', 'adl_dress_12', 'adl_walk_03', 'adl_walk_12',
    'adl_bath_03', 'adl_bath_12', 'adl_eat_03', 'adl_eat_12', 'adl_bed_03', 'adl_bed_12',
    'adl_toilet_03', 'adl_toilet_12', 'n_adl_03', 'n_adl_12', 'iadl_money_03', 'iadl_money_12',
    'iadl_meds_03', 'iadl_meds_12', 'iadl_shop_03', 'iadl_shop_12', 'iadl_meals_03', 'iadl_meals_12',
    'n_iadl_03', 'n_iadl_12', 'depressed_03', 'depressed_12', 'hard_03', 'hard_12', 'restless_03',
    'restless_12', 'happy_03', 'happy_12', 'lonely_03', 'lonely_12', 'enjoy_03', 'enjoy_12',
    'sad_03', 'sad_12', 'tired_03', 'tired_12', 'energetic_03', 'energetic_12', 'n_depr_03',
    'n_depr_12', 'cesd_depressed_03', 'cesd_depressed_12', 'hypertension_03', 'hypertension_12',
    'diabetes_03', 'diabetes_12', 'resp_ill_03', 'resp_ill_12', 'arthritis_03', 'arthritis_12',
    'hrt_attack_03', 'hrt_attack_12', 'stroke_03', 'stroke_12', 'cancer_03', 'cancer_12',
    'n_illnesses_03', 'n_illnesses_12', 'exer_3xwk_03', 'exer_3xwk_12', 'alcohol_03', 'alcohol_12',
    'tobacco_03', 'tobacco_12', 'test_chol_03', 'test_chol_12', 'test_tuber_03', 'test_tuber_12',
    'test_diab_03', 'test_diab_12', 'test_pres_03', 'test_pres_12', 'hosp_03', 'hosp_12',
    'visit_med_03', 'visit_med_12', 'memory_12'
]

# Filtrar columnas relevantes y eliminar columnas con más del 50% de datos nulos
def clean_data(df, relevant_columns):
    # Mantener solo columnas relevantes
    df = df[relevant_columns]
    # Eliminar columnas con más del 50% de datos nulos
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)
    return df

# Limpiar datos de entrenamiento y prueba
train_features_cleaned = clean_data(train_features, relevant_columns)
test_features_cleaned = clean_data(test_features, relevant_columns)

# Guardar los datos limpios
train_features_cleaned.to_csv('../data/train_features_cleaned.csv', index=False)
test_features_cleaned.to_csv('../data/test_features_cleaned.csv', index=False)

print("Datos limpiados y guardados en 'data/train_features_cleaned.csv' y 'data/test_features_cleaned.csv'")
