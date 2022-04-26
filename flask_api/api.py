from flask import Flask
from flask_restful import Resource, Api, reqparse
import os
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from Networks.network_config import NetworkParams
import sys

load_dotenv()
pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

app = Flask(__name__)
api = Api(app)

os.environ['model_path'] = (f'{os.path.join(sys.path[0], os.pardir)}/models')
print(os.environ['model_path'])

class Models(Resource):
    def get(self):
        self.list_of_files = []

        for root, dirs, files in os.walk(os.environ['model_path']):
            for file in files:
                self.list_of_files.append(os.path.join(root, file))
        return self.list_of_files, 200
    pass

api.add_resource(Models, '/models')  # '/users' is our entry point

if __name__ == '__main__':
    app.run(debug=True)  # run our Flask app

