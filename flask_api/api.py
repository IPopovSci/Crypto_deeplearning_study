from flask import Flask
from flask_restful import Resource, Api, reqparse
import os
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from Networks.network_config import NetworkParams
import sys
from flask_api.website import create_app

load_dotenv()
pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

app = create_app()
api = Api(app)


os.environ['model_path'] = (f'{os.path.join(sys.path[0], os.pardir)}/models')
#print(os.environ['model_path'])


if __name__ == '__main__':
    app.run(debug=True)  # run our Flask app

