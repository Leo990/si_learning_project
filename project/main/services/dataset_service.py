from project.main.config.db_config import DBContext
from project.main.enums.db_enum import DBEnum
from project.main.enums.collection_enum import CollectionEnum
from project.main.dtos.dataset_dto import DataSetDTO
from bson.objectid import ObjectId

DB_CONTEXT = DBContext()
collection = DB_CONTEXT.get_collection(DBEnum.SI_DB, CollectionEnum.DATASET)


def insert(dataset: DataSetDTO) -> dict:
    try:
        collection.insert_one(dataset.serialize(False))
        DB_CONTEXT.close()
        return {"EXITO": "Se ha insertado el dataset correctamente!!!"}
    except Exception as e:
        return {"ERROR": e}


def index():
    dataset_list = []
    for dataset in list(collection.find()):
        dataset_list.append(
            DataSetDTO(dataset['name'], dataset['records'], dataset['model_name'],
                       dataset['accuracy'],
                       dataset['is_preprocessed'], str(dataset['_id'])).serialize(True))
    DB_CONTEXT.close()
    return dataset_list


def find(ident: str):
    document_id = ObjectId(ident)  # ID del documento a consultar
    try:
        found_dataset = collection.find_one({'_id': document_id})
        return DataSetDTO(found_dataset['name'], found_dataset['records'], found_dataset['model_name'],
                          found_dataset['accuracy'],
                          found_dataset['is_preprocessed'], str(found_dataset['_id'])).serialize(True)
    except Exception:
        return None
    finally:
        DB_CONTEXT.close()


def remove(ident: str) -> dict:
    document_id = ObjectId(ident)  # ID del documento a consultar
    try:
        collection.find_one_and_delete({'_id': document_id})
        return {"EXITO": f"el dataset con el id: {ident} se ha eliminado correctamente!!!"}
    except Exception as e:
        return {"ERROR": e}
    finally:
        DB_CONTEXT.close()
