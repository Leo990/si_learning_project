from pymongo.mongo_client import MongoClient
import configparser
import dropbox

global config
config = configparser.ConfigParser()
config.read('./config/config.ini')


class DBContext:
    def __init__(self):
        uri = config.get('default', 'db_uri')
        self.client = MongoClient(uri)

    def get_db(self, db_enum):
        return self.client[db_enum]

    def get_collection(self, db_enum, collection_enum):
        db = self.get_db(db_enum)
        return db[collection_enum]


class DropBoxContext:
    def __init__(self):
        # Configurar el token de acceso
        access_token = config.get('default', 'access_token')
        # Crear una instancia de cliente de Dropbox
        self.dbx = dropbox.Dropbox(access_token)
