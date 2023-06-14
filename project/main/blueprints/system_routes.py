from flask import request, jsonify, Blueprint
from project.main.services import system_services as ss
from project.main.dtos.service_dtos import ParamTrainDTO, ParamPreprocessDTO

system_bp = Blueprint('system', __name__)


@system_bp.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        return ss.upload_files(file), 200
    else:
        return jsonify({'error': 'No se recibió ningún archivo.'}), 400


@system_bp.route('/preprocess_dataset', methods=['POST'])
def preprocess_dataset():
    param_preprocess_dto = ParamPreprocessDTO(**request.json)
    return ss.preprocess_dataset(param_preprocess_dto), 200


@system_bp.route('/manage_dataset', methods=['POST'])
def manage_dataset():
    param_train_dto = ParamTrainDTO(**request.json)
    return ss.manage_dataset(param_train_dto), 200
