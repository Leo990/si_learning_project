import numpy as np
from sklearn import datasets
from sklearn.impute import SimpleImputer


def impute_mean(dataset):
    x = dataset.data
    y = dataset.target
    # Crear un imputador para la estrategia de media
    imputer_mean = SimpleImputer(strategy='mean')
    x_imputed_mean = imputer_mean.fit_transform(x)

    return x_imputed_mean, y


def impute_median(dataset):
    X = dataset.data
    y = dataset.target
    # Crear un imputador para la estrategia de mediana
    imputer_median = SimpleImputer(strategy='median')
    X_imputed_median = imputer_median.fit_transform(X)

    return X_imputed_median, y


# Aplicar hot deck imputation
# Aquí asumiremos que los datos faltantes están representados como NaN en el dataset original
# y los reemplazaremos por los valores correspondientes en las mismas características de otras instancias
def impute_hot_deck(dataset):
    x = dataset.data
    y = dataset.target

    # Encuentra las posiciones de los valores faltantes
    missing_values = np.isnan(x)

    # Itera sobre cada columna
    for col_idx in range(x.shape[1]):
        col = x[:, col_idx]

        # Encuentra las posiciones de los valores faltantes en la columna actual
        col_missing_values = missing_values[:, col_idx]

        # Reemplaza los valores faltantes con el valor más cercano en la misma columna
        for idx, val in enumerate(col):
            if np.isnan(val):
                prev_val = col[idx - 1]
                next_val = col[idx + 1]

                if np.isnan(prev_val) or np.isnan(next_val):
                    # Si los valores previo y siguiente también son faltantes,
                    # se mantiene el valor faltante
                    continue
                # Reemplaza el valor faltante con el valor más cercano
                col[idx] = prev_val if abs(val - prev_val) <= abs(val - next_val) else next_val

    return x, y


# Carga el dataset de cáncer de mama
dataset = datasets.load_breast_cancer()

# Imputación por media
X_mean, y_mean = impute_mean(dataset)

# Imputación por mediana
X_median, y_median = impute_median(dataset)

# Imputación por hot deck
X_hot_deck, y_hot_deck = impute_hot_deck(dataset)
