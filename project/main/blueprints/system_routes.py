from flask import request, jsonify, Blueprint
from project.main.services import system_services as ss
from project.main.dtos.param_train_dto import ParamTrainDTO

system_bp = Blueprint('system', __name__)


@system_bp.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        return ss.upload_files(file), 200
    else:
        return jsonify({'error': 'No se recibió ningún archivo.'}), 400


@system_bp.route('/train', methods=['POST'])
def train_dataset():
    param_train_dto: ParamTrainDTO = request.json
    return ss.train(param_train_dto), 200
