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
#Importacion de módulo para dividir el dataset
from sklearn.model_selection import train_test_split
#Importación knn
from sklearn.neighbors import KNeighborsClassifier
#Importación de matriz de confusión
from sklearn import metrics
#Importación modulos de dropbox
from project.main.services import dropbox_service as ds
from project.main.dtos.service_dtos import RecordDTO
import joblib

def carga(file, filepath):
    """Mirar si es mas conveniente subir un archivo generado desde python"""
    return ds.load_file(file, "/"+filepath)


def lista_archivos(path:str):
    return ds.list_files(path)


def descarga_archivo(path: str):
    return ds.download_file(path)


def remove(ident: str):
    #document_id = ObjectId(ident)  # ID del documento a consultar
    #return collection.find_one_and_delete({'_id': document_id}) is not None
    test = "Se elimino" + str(ident)

    return str(test)


"""Predice un dataset con un dataset ya establecido"""
def predice_model():
    return predecir('./models.h5', [1,5,10])


def predecir(ruta_modelo, array):
    # Cargar el modelo desde la ruta especificada
    modelo = joblib.load(ruta_modelo)

    # Realizar las predicciones
    predicciones = modelo.predict(array)

    return predicciones


"""Inserta datos"""


def insert(record: RecordDTO):
    """ident = collection.insert_one(record.serialize(False)).inserted_id"""
    test = record.serialize(False)

    return str(test)


def descarga(record: RecordDTO):
    """ident = collection.insert_one(record.serialize(False)).inserted_id"""
    test=record.serialize(False)

    return str(test)




