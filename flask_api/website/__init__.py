from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
import sys
from pathlib import Path

db = SQLAlchemy()
db_path = f'{Path(sys.path[0]).parents[0]}/sql/model_params.sqlite'
DB_NAME = "model_params.sqlite"


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = db_path
    db.init_app(app)

    from flask_api.website.views import views

    app.register_blueprint(views,url_prefix='/')

    return app

