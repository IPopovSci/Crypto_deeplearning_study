from flask import Flask
from os import path
DB_NAME = "database.db"


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'

    from flask_api.website.views import views

    app.register_blueprint(views,url_prefix='/')

    return app

