from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from os import path
import sys
from pathlib import Path

db = SQLAlchemy()
db_path = f'{Path(sys.path[0]).parents[0]}/sql/model_params.sqlite'
DB_NAME = "model_params.sqlite"


'''Creates flask app, loads in SQL database into SQLAlchemy'''
def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    db.app = app
    db.init_app(app)


    from flask_api.website.views import views
    import flask_api.website.models as models

    app.register_blueprint(views,url_prefix='/')

    from .models import Model_params

    return app

#print(db)