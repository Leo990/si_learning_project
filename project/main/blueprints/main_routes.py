import json

from flask import request, jsonify, Blueprint
from project.main.services import upload_files_service as ufs
from project.main.services import dataset_service as ds
from project.main.services import train_service as ts
from project.main.dtos.param_train_dto import ParamTrainDTO

main_bp = Blueprint('main', __name__)


@main_bp.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        return ufs.upload_file(file), 200
    else:
        return jsonify({'error': 'No se recibió ningún archivo.'}), 400


@main_bp.route('/datasets', methods=['GET'])
def list_datasets():
    return jsonify(ds.index()), 200


@main_bp.route('/datasets/find/<id>', methods=['GET'])
def find_dataset(id):
    return ds.find(id), 200


@main_bp.route('/datasets/remove/<id>', methods=['POST'])
def remove_dataset(id):
    return ds.remove(id), 200

@main_bp.route('/train', methods=['POST'])
def remove_dataset():
    param_train_dto : ParamTrainDTO = request.json
    return ts.train(param_train_dto), 200