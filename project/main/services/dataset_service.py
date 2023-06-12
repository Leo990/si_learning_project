from project.main.config.config import DBContext
from project.main.enums.db_enum import DBEnum, CollectionEnum
from bson.objectid import ObjectId
import json

from project.main.dtos.service_dtos import DataSetDTO, RecordDTO
from project.main.utils.utils import type_of

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
            DataSetDTO(dataset['name'], dataset['record_id'], dataset['model_name'],
                       dataset['accuracy'], str(dataset['_id'])).serialize(True))
    return dataset_list


def find(ident: str):
    document_id = ObjectId(ident)  # ID del documento a consultar
    found_dataset = collection.find_one({'_id': document_id})
    if found_dataset is not None:
        return DataSetDTO(found_dataset['name'], found_dataset['record_id'], found_dataset['model_name'],
                          found_dataset['accuracy'], str(found_dataset['_id'])).serialize(True)
    raise Exception('No se encontró el dataset')


def update(ident: str, dataset_dto: DataSetDTO):
    document_id = ObjectId(ident)  # ID del documento a consultar
    collection.update_one(
        {
            '_id': document_id
        },
        {
            '$set': {
                'name': dataset_dto.name,
                'record_id': dataset_dto.record_id,
                'model_name': dataset_dto.model_name,
                'accuracy': dataset_dto.accuracy}
        }
    )


def remove(ident: str):
    document_id = ObjectId(ident)  # ID del documento a consultar
    return collection.find_one_and_delete({'_id': document_id}) is not None


def info_columns(dataset_dto: DataSetDTO):
    record: RecordDTO = rs.find(dataset_dto.record_id)
    data = json.loads(record.my_data)
    item = data[0]
    info_columns_dic = {}
    for key in item.keys():
        info_columns_dic[key] = type_of(item[key])
    return info_columns_dic
