from project.main.config.config import DBContext
from project.main.enums.db_enum import DBEnum
from project.main.enums.collection_enum import CollectionEnum
from project.main.dtos.dataset_dto import DataSetDTO
from bson.objectid import ObjectId

from project.main.services import record_service as rs

DB_CONTEXT = DBContext()
collection = DB_CONTEXT.get_collection(DBEnum.SI_DB, CollectionEnum.DATASET)


def insert(dataset: DataSetDTO):
    if dataset.record_id is not None:
        record = rs.find(dataset.record_id)
        if record is None:
            raise Exception("No existe registro asociado para el dataset")
        ident = collection.insert_one(dataset.serialize(False)).inserted_id
        return str(ident)
    else:
        raise Exception("El identificador del registro no se ingresó correctamente")


def index():
    dataset_list = []
    for dataset in list(collection.find()):
        dataset_list.append(
            DataSetDTO(dataset['name'], dataset['records'], dataset['model_name'],
                       dataset['accuracy'],
                       dataset['is_preprocessed'], str(dataset['_id'])).serialize(True))
    return dataset_list


def find(ident: str):
    document_id = ObjectId(ident)  # ID del documento a consultar
    found_dataset = collection.find_one({'_id': document_id})
    if found_dataset is not None:
        return DataSetDTO(found_dataset['name'], found_dataset['records'], found_dataset['model_name'],
                          found_dataset['accuracy'],
                          found_dataset['is_preprocessed'], str(found_dataset['_id'])).serialize(True)
    raise Exception('No se encontró el dataset')


def remove(ident: str):
    document_id = ObjectId(ident)  # ID del documento a consultar
    return collection.find_one_and_delete({'_id': document_id}) is not None
