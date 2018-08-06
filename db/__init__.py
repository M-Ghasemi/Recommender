from .database import MongoConnection
mongodb = MongoConnection().getCursor()

__all__ = ['mongodb']
