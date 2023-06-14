from sklearn import datasets
#Calculos matemáticos y matrices
import numpy as np
#graficas
import pandas as pd
#Importacion de módulo para dividir el dataset
from sklearn.model_selection import train_test_split
#Importación knn
from sklearn.neighbors import KNeighborsClassifier
#Importación de matriz de confusión
from sklearn import metrics
#Importación modelo regresión lineal
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
#Métricas para la regresión lineal
from sklearn.metrics import mean_squared_error
#Importación de la regresión logística
#Normalización de los datos
from sklearn.preprocessing import MinMaxScaler
#Guardar modelos.
import joblib


def knn_training(dataset, neighborns):
    # Dividir el dataset en características (X) y etiquetas (y)
    X = dataset.data
    print(X.shape)
    print(X)
    print("------------")
    y = dataset.target
    print(y.shape)
    print(dataset.target_names)
    print(y)
    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=6)

    # Crear el clasificador KNN
    knn = KNeighborsClassifier(n_neighbors=neighborns)  # Puedes ajustar el valor de "n_neighbors" según tus necesidades

    # Entrenar el clasificador KNN
    knn.fit(XTrain, yTrain)
    # Predecir las etiquetas para el conjunto de prueba
    yPredict = knn.predict(XTest)
    print("Las predicciones son ")
    print(yPredict)
    print("Se esperaba ")
    print(yTest)
    print("Accuracy=", metrics.accuracy_score(yTest, yPredict))


def linear_regression_training(dataset, columna, etiqueta):
    X = np.array(dataset[:, [columna]])  # Característica específica (columna ingresada)
    y = dataset[:, etiqueta]  # Posición de la etiqueta ingresada

    regresionLineal = LinearRegression()
    regresionLineal.fit(X, y)
    w = regresionLineal.coef_
    b = regresionLineal.intercept_
    print("W =", w)
    print("b =", b)

    nuevaConsulta = np.array([[10]])  # Valor de consulta
    prediccion = regresionLineal.predict(nuevaConsulta)
    print(prediccion)

    r2 = regresionLineal.score(X, y)
    print("R2 =", r2)
    prediccionEntrenamiento = regresionLineal.predict(X)
    mse = mean_squared_error(y_true=y, y_pred=prediccionEntrenamiento)
    print("MSE=", mse)
    rmse = np.sqrt(mse)
    print("RMSE=", rmse)


def linear_logisticRegression_training(dataset):
    X = dataset.data
    print(X)
    print(X.shape)
    y = dataset.target

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=6)
    print("XTRain=", XTrain.shape)
    print("XTest=", XTest.shape)

    escalar = MinMaxScaler()
    # Fit solo halla los valores de min y max
    # Transform aplica la fórmula
    XTrain = escalar.fit_transform(XTrain)
    # Transform aplica la fórmula
    XTest = escalar.transform(XTest)

    # Implementación de la regresión logística
    modelo = LogisticRegression()
    modelo.fit(XTrain, yTrain)
    yPredicho = modelo.predict(XTest)
    # Guardar el modelo
    joblib.dump(modelo, './models.h5')
    print(yPredicho)
    print("vs")
    print(yTest)


data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])

#knn_training(data)

linear_regression_training(data, 5, 2)

dataset = datasets.load_breast_cancer()
print(dataset.DESCR)
knn_training(dataset, 3)

print(dataset)
print(data)

linear_logisticRegression_training(dataset)

