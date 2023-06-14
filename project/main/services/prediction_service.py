#Organiza la importacion de librerias
"""
from project.main.config.config import DBContext
from project.main.enums.db_enum import DBEnum, CollectionEnum
from project.main.dtos.service_dtos import RecordDTO
from bson.objectid import ObjectId

DB_CONTEXT = DBContext()
collection = DB_CONTEXT.get_collection(DBEnum.SI_DB, CollectionEnum.RECORD)
"""
#Mira si se necesitan estas importaciones
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
from project.main.services import dropbox_service as ds
from project.main.dtos.service_dtos import RecordDTO



def carga(file, filepath):
    """Mirar si es mas conveniente subir un archivo generado desde python"""
    return ds.load_file(file,"/"+filepath)

def lista_archivos(path:str):
    return ds.list_files(path)

def descarga_archivo(path: str):
    return ds.download_file(path)



def index():
    #record_list = []
    #for record in list(collection.find()):
    #    record_list.append(
    #        RecordDTO(record['my_data'], record['is_preprocessed'], str(record['_id'])).serialize(True))

    record_list = ["test1","test2"]

    #---------------Devuelve el index
    return record_list


def find(ident: str):
    #document_id = ObjectId(ident)  # ID del documento a consultar
    #found_record = collection.find_one({'_id': document_id})
    #if found_record is not None:
    #    return RecordDTO(found_record['my_data'], found_record['is_preprocessed'], ident).serialize(True)
    #raise Exception('No se ha encontrado ningun registro')
    test="Se encontro el dato"+str(ident)

    return str(test)



def remove(ident: str):
    #document_id = ObjectId(ident)  # ID del documento a consultar
    #return collection.find_one_and_delete({'_id': document_id}) is not None
    test = "Se elimino" + str(ident)

    return str(test)

"""Predice un dataset con un dataset ya establecido"""
def predice_model():
    """Dataset que debe cargarse"""
    """Aqui el data set debe ser el JSON que recive, o un csv de algun tipo"""
    dataset = datasets.load_breast_cancer()
    """Aqui el data set debe ser el JSON que recive, o un csv de algun tipo"""


    #Asume que ya tiene cargado un modelo
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    linear_regression_training(data, 5, 2)



    #Procesa el modelo
    dicc=knn_training(dataset, 3)

    return str(dicc)


#Datos extra
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

    datos = {
        "x": "[" + ", ".join([str(row) for row in X]) + "]",
        "y": "[" + ", ".join([str(row) for row in y]) + "]",
        "Prediccion": "[" + ", ".join([str(row) for row in yPredict]) + "]",
        "SeEsperaba": "[" + ", ".join([str(row) for row in yTest]) + "]",
        "Accuracy": metrics.accuracy_score(yTest, yPredict)
    }

    return str(datos)



"""BORRAR LUEGO--------------------------------------------------------"""
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

"""Inserta datos"""
def insert(record: RecordDTO):
    """ident = collection.insert_one(record.serialize(False)).inserted_id"""
    #------------HAZ UN METODO QUE INSERTE EN LA BASE Y DEVUELVA ALGO
    #
    #
    test=record.serialize(False)

    return str(test)

def descarga(record: RecordDTO):
    """ident = collection.insert_one(record.serialize(False)).inserted_id"""
    #------------HAZ UN METODO QUE INSERTE EN LA BASE Y DEVUELVA ALGO
    #
    #
    test=record.serialize(False)

    return str(test)




