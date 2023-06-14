from flask import jsonify, Blueprint,request
from project.main.services import record_service as rs
from project.main.dtos.service_dtos import RecordDTO

record_bp = Blueprint('record', __name__)


@record_bp.route('/records/create', methods=['POST'])
def create_record():
    record_dto = RecordDTO(**request.json)
    return jsonify(rs.insert(record_dto)), 200


@record_bp.route('/records', methods=['GET'])
def list_record():
    return jsonify(rs.index()), 200


@record_bp.route('/records/find/<ident>', methods=['GET'])
def find_record(ident):
    return rs.find(ident), 200


@record_bp.route('/records/remove/<ident>', methods=['POST'])
def remove_record(ident):
    return rs.remove(ident), 200
