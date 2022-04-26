import sqlite3
from sqlite3 import Error
import sys

path = f'{sys.path[0]}/model_params.sqlite'
#print(sys.path[0])


def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

connection = create_connection(path)