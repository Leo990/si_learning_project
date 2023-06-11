from flask import jsonify, Blueprint
from project.main.services import dataset_service as ds

dataset_bp = Blueprint('dataset', __name__)


@dataset_bp.route('/datasets', methods=['GET'])
def list_datasets():
    return jsonify(ds.index()), 200


@dataset_bp.route('/datasets/find/<id>', methods=['GET'])
def find_dataset(ident):
    return ds.find(ident), 200


@dataset_bp.route('/datasets/remove/<id>', methods=['POST'])
def remove_dataset(ident):
    return ds.remove(ident), 200
