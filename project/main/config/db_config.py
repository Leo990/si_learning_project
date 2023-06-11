from pymongo.mongo_client import MongoClient
from project.main.enums.db_enum import DBEnum


class DBContext:

    def __init__(self):
        uri = "mongodb+srv://new_user_21:uCeRoi2boMgcbT2s@silearning.zxce2md.mongodb.net/?retryWrites=true&w=majority"
        self.client = MongoClient(uri) if self.client is None else self.client

    def get_db(self, db_enum):
        return self.client[db_enum]

    def get_collection(self, db_enum, collection_enum):
        db = self.get_db(db_enum)
        return db[collection_enum]

    def close(self):
        self.client.close()
