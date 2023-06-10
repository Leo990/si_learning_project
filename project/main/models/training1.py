#Calculos matemáticos y matrices
import numpy as np
#Importacion de módulo para dividir el dataset
from sklearn.model_selection import train_test_split
#Importación knn
from sklearn.neighbors import KNeighborsClassifier
#Importación de matriz de confusión
from sklearn import metrics
#Importación modelo regresión lineal
from sklearn.linear_model import LinearRegression
#Métricas para la regresión lineal
from sklearn.metrics import mean_squared_error
#Importación de la regresión logística
from sklearn.linear_model import LogisticRegression
#Normalización de los datos
from sklearn.preprocessing import MinMaxScaler


def knn_training(dataset):
    # Dividir el dataset en características (X) y etiquetas (y)
    X = dataset.iloc[:, :-1]  # Todas las columnas excepto la última
    y = dataset.iloc[:, -1]   # Última columna, que contiene las etiquetas

    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=6)

    # Crear el clasificador KNN
    knn = KNeighborsClassifier(n_neighbors=3)  # Puedes ajustar el valor de "n_neighbors" según tus necesidades

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
    print(yPredicho)
    print("vs")
    print(yTest)



