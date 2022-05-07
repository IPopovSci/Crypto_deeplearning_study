import sqlite3
from sqlite3 import Error
import sys
from sql.fetch_info import initial_folder_parse
import pickle


path = f'{sys.path[0]}/model_params.sqlite'


def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection


connection = create_connection(path)

cursor_obj = connection.cursor()  # cursor object is used to execute sql commands

params = initial_folder_parse()
# for model in list(params.keys()):
#     print(model, params[model]["type"])
'''Function to insert singular entry into adam_opt table
Accepts: Connection, parameter dictionary
Returns: Id of last affected row'''

def insert_adam_opt_table(conn,params):
    sql = '''INSERT OR IGNORE INTO ADAM_OPT (learning_rate,amsgrad,decay,b1,b2,epsilon)
                                VALUES (?,?,?,?,?,?)'''
    cursor_obj = conn.cursor()

    cursor_obj.execute(sql,params)
    conn.commit()

    return cursor_obj.lastrowid

'''Function to insert singular entry into model table
Accepts: Connection, parameter dictionary
Returns: Id of last affected row'''

def insert_model_table(conn,params):

    sql = '''INSERT INTO MODEL (model_name,type,depth,input_shape,optimizer_type,optimizer_id,ensemble,ensemble_type,lc_config,ticker,interval)
                                VALUES (?,?,?,?,?,?, ?,
                                ?,?,?,?)'''
    cursor_obj = conn.cursor()

    cursor_obj.execute(sql,params)
    conn.commit()

    return cursor_obj.lastrowid

'''Function to insert all parsed models into database
Accepts: Connection, parameter dictionary'''

def insert_init_db(connection,params):
    with connection:
        for model in list(params.keys()):

            params_adam_list = (float(params[model]["learning_rate"]),params[model]["amsgrad"],params[model]["decay"],float(params[model]["b1"]),float(params[model]["b2"]),params[model]["epsilon"])
            insert_adam_opt_table(connection,params_adam_list)
            #print(type(params[model]['lc_config']))

            sql_grab_optimizer_id = '''SELECT optimizer_id FROM adam_opt ORDER BY optimizer_id DESC LIMIT 1;'''
            cursor_obj.execute(sql_grab_optimizer_id)
            adam_id = cursor_obj.fetchone()
            #print(adam_id[0])

            params_model_list = (model,params[model]["type"],params[model]["depth"],str(params[model]["input_shape"]),params[model]["optimizer_type"],int(adam_id[0]), params[model]["ensemble"],
                                    params[model]["ensemble_type"],pickle.dumps(params[model]['lc_config']),params[model]['ticker'],params[model]['interval']);
            insert_model_table(connection,params_model_list)

insert_init_db(connection,params)
