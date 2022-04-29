
from sql.db_populate import insert_adam_opt_table, insert_model_table
from sql.fetch_info import initial_folder_parse, single_model_parse
import pickle
import os
import shutil
from pathlib import Path, PurePath

'''Function to save a singular model to the database
Accepts: Connection, Model path (Must be full path, including the filename)
ticker, interval, cursor object'''
def save_model_to_db(connection, model_path, ticker, interval, cursor_obj):
    with connection:
        params = single_model_parse(model_path, ticker, interval)
        params_adam_list = (float(params["learning_rate"]), params["amsgrad"], params["decay"],
                            float(params["b1"]), float(params["b2"]), params["epsilon"])

        insert_adam_opt_table(connection, params_adam_list)

        sql_grab_optimizer_id = '''SELECT optimizer_id FROM adam_opt ORDER BY optimizer_id DESC LIMIT 1;'''
        cursor_obj.execute(sql_grab_optimizer_id)
        adam_id = cursor_obj.fetchone()

        params_model_list = (str(Path(model_path).name), str(params["type"]), params["depth"], str(params["input_shape"]),
                             params["optimizer_type"], int(adam_id[0]), params["ensemble"],
                             params["ensemble_type"], pickle.dumps(params['lc_config']));

        insert_model_table(connection, params_model_list)

'''Function to delete a singular model from the database
Accepts: Connection, Model path (Must be full path, including the filename),cursor object'''

def delete_model(connection, model_path, cursor_obj):
    model_name = Path(model_path).name
    if os.path.exists(model_path):
        os.remove(model_path)
    else:
        print("The model does not exist")

    sql_check_ens = '''select ensemble from model where model_name = ?'''
    cursor_obj.execute(sql_check_ens, [str(model_name)])
    ensemble = cursor_obj.fetchone()
    if ensemble[0] == str(1):
        sql_ens_type = '''select ensemble_type from model where model_name = ?'''
        cursor_obj.execute(sql_ens_type, [str(model_name)])
        ens_type = cursor_obj.fetchone()
        ens_path = Path(model_path).parents[1].joinpath(ens_type[0], model_name)
        print(ens_path)
        if os.path.exists(ens_path):
            os.remove(ens_path)
            print('ensemble model removed')
        else:
            print(
                "The model does not exist (This shouldn't happen, indicates mismatch between sql and folder structure")

    sql = '''DELETE FROM Model WHERE model_name=?'''

    cursor_obj.execute(sql, [str(model_name)])
    connection.commit()

'''Function to copy the model to specified ensembly folder
Accepts: connection, full model path, ensemble type (Ensemble folder name), cursor object'''

def move_model_to_ensembly(connection,model_path,ens_type,cursor_obj):
    model_name = Path(model_path).name
    path_to_ensemble = Path(model_path).parents[1].joinpath(ens_type,model_name)

    if os.path.exists(path_to_ensemble)==False:
        shutil.copyfile(model_path, path_to_ensemble)
        sql_change_ensembly_status = '''update model set ensemble = ?, ensemble_type = ? where model_name = ?'''
        cursor_obj.execute(sql_change_ensembly_status, [True,ens_type,model_name])
        connection.commit()
    else:
        print('Model already exists in ensemble, or unexpected error')
