from project.main.config.config import DBContext
from project.main.enums.db_enum import DBEnum, CollectionEnum
from project.main.dtos.service_dtos import RecordDTO
from bson.objectid import ObjectId

DB_CONTEXT = DBContext()
collection = DB_CONTEXT.get_collection(DBEnum.SI_DB, CollectionEnum.RECORD)


def insert(record: RecordDTO):
    ident = collection.insert_one(record.serialize(False)).inserted_id
    return str(ident)


def index():
    record_list = []
    for record in list(collection.find()):
        record_list.append(
            RecordDTO(record['my_data'], record['is_preprocessed'], str(record['_id'])).serialize(True))
    return record_list


def find(ident: str):
    document_id = ObjectId(ident)  # ID del documento a consultar
    found_record = collection.find_one({'_id': document_id})
    if found_record is not None:
        return RecordDTO(found_record['my_data'], found_record['is_preprocessed'], ident).serialize(True)
    raise Exception('No se ha encontrado ningun registro')


def remove(ident: str):
    document_id = ObjectId(ident)  # ID del documento a consultar
    return collection.find_one_and_delete({'_id': document_id}) is not None
