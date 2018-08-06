import logging
from pymongo import MongoClient
import settings

logger = logging.getLogger(__name__)


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


@singleton
class MongoConnection(object):
    __db = None

    def __init__(self):
        self.get_connection()

    @classmethod
    def get_connection(cls):
        if cls.__db is None:
            cls.__db = MongoClient(
                host=settings.MONGO['HOST'], port=settings.MONGO['PORT'],
                serverSelectionTimeoutMS=6000, maxPoolSize=None
            )
        return cls.__db

    def getCursor(self):
        self.__db_cursor = self.__db[settings.MONGO.get('DB')]

        if settings.MONGO['USER'] and settings.MONGO['PASSWORD']:
            self.__db_cursor.authenticate(
                settings.MONGO['USER'],
                settings.MONGO['PASSWORD'],
                source='admin')

        return self.__db_cursor
