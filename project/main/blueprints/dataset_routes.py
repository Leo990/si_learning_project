from flask import jsonify, Blueprint, request
from project.main.services import dataset_service as ds
from project.main.dtos.service_dtos import DataSetDTO

dataset_bp = Blueprint('dataset', __name__)


@dataset_bp.route('/datasets/create', methods=['POST'])
def create_dataset():
    dataset_dto = DataSetDTO(**request.json)
    return jsonify(ds.insert(dataset_dto)), 200


@dataset_bp.route('/datasets/update', methods=['POST'])
def update_dataset():
    dataset_dto = DataSetDTO(**request.json)
    return jsonify(ds.update(dataset_dto).__dict__), 200


@dataset_bp.route('/datasets', methods=['GET'])
def list_datasets():
    return jsonify(ds.index()), 200


@dataset_bp.route('/datasets/find/<ident>', methods=['GET'])
def find_dataset(ident):
    return jsonify(ds.find(ident).__dict__), 200


@dataset_bp.route('/datasets/remove/<ident>', methods=['POST'])
def remove_dataset(ident):
    return jsonify(ds.remove(ident)), 200


@dataset_bp.route('/datasets/info_columns', methods=['GET'])
def info_columns(ident):
    dataset_dto = DataSetDTO(**request.json)
    return jsonify(ds.info_columns(dataset_dto)), 200
