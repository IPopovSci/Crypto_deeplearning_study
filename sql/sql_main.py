import sqlite3
from sqlite3 import Error
import sys
from sql.db_populate import insert_init_db
from sql.db_manipulation import delete_model,move_model_to_ensembly,save_model_to_db
from sql.sql_create_tables import create_adam_opt_table,create_model_table

path = f'{sys.path[0]}/model_params.sqlite'
#print(sys.path[0])
'''Module for executing SQL commands related to managing model database
Always use cursor to execute sql
If being deployed on a new enviroment,
1) Run create_adam_opt_table, create_model_table
2) Run insert_init_db if there are already existing models you would like to index

For existing system:
1) Once the model been trained and approved, use save_model_to_db to add it to database
2) To add same model to ensemble, use move_model_to_ensembly (Note: This will create a copy)
3) To delete a model from both hard-drive and the database, use delete_model'''


def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

connection = create_connection(path)

cursor_obj = connection.cursor() #cursor object is used to execute sql commands


