from flask import jsonify, Blueprint,request
from project.main.services import prediction_service as ps #Importacion del servicio de prediccion
from project.main.dtos.service_dtos import RecordDTO #Talvez se use mas tarde

prediction_bp = Blueprint('prediction', __name__)


#METODO QUE CREA UNA PREDICCION
@prediction_bp.route('/prediction/create', methods=['POST'])
def create_prediction():
    #Cambiar modelo
    record_dto = RecordDTO(**request.json)


    return jsonify(ps.insert(record_dto)), 200


#METODO PARA OBTENER DATOS
@prediction_bp.route('/prediction', methods=['GET'])
def list_prediction():
    return jsonify(ps.index()), 200

#METODO QUE ENCUENTRA UN DATO
@prediction_bp.route('/prediction/find/<ident>', methods=['GET'])
def find_prediction(ident):
    return ps.find(ident), 200

#METODO QUE REMUEVE UN DATO
@prediction_bp.route('/prediction/remove/<ident>', methods=['POST'])
def remove_prediction(ident):
    return ps.remove(ident), 200
