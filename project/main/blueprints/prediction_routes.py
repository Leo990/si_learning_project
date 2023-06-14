from flask import jsonify, Blueprint,request,send_file,make_response
from project.main.services import prediction_service as ps #Importacion del servicio de prediccion
from project.main.dtos.service_dtos import RecordDTO #Talvez se use mas tarde
from io import BytesIO
from dropbox.exceptions import ApiError
import joblib

prediction_bp = Blueprint('prediction', __name__)


#METODO QUE CREA UNA PREDICCION
@prediction_bp.route('/prediction/predice', methods=['POST'])
def create_prediction():
    # Obtener los datos del cuerpo de la solicitud POST
    data = request.json

    # Verificar si se proporciona la ruta del archivo del modelo
    if 'ruta_modelo' not in data:
        return jsonify({'error': 'No se proporcionó la ruta del archivo del modelo.'}), 400

    # Verificar si se proporciona el arreglo de datos para predecir
    if 'datos' not in data:
        return jsonify({'error': 'No se proporcionó el arreglo de datos para predecir.'}), 400

    # Cargar el modelo desde la ruta del archivo
    ruta_modelo = data['ruta_modelo']
    modelo = joblib.load(ruta_modelo)

    # Realizar las predicciones con el modelo
    arreglo_datos = data['datos']
    predicciones = modelo.predict(arreglo_datos)

    # Retornar las predicciones como respuesta
    return jsonify({'predicciones': predicciones.tolist()}), 200

"""Se encarga de subir un archivo
Para probar ver en el postman body-> form-data -> donde la llave es file y el valor es el archivo
"""
@prediction_bp.route('/prediction/upload', methods=['POST'])
def upload_prediction():
    file = request.files['file']
    filename = file.filename
    file.save(filename)

    response = ps.carga(file,filename)
    return jsonify(response), 200

"""Lista los archivos que hay
Para probar poner llave es path y el valor es una ruta especifica, dejar el valor como ''
"""
@prediction_bp.route('/prediction/list_files')
def list_files():
    return jsonify(ps.lista_archivos(''))


""""Se encarga de descargar un archivo
Para probar poner como llave filepath y de valor alguno de los archivos listados


"filepath":"/archivo"

"""
@prediction_bp.route('/prediction/download', methods=['GET'])
def download_prediction():
    data = request.get_json()
    file_path = data['filepath']
    try:
        # Download the file from Dropbox
        metadata, file = ps.descarga_archivo(file_path)

        # Get the file contents and name
        file_contents = file.content
        file_stream = BytesIO(file_contents)
        file_name = metadata.name


        # Return the file as a Flask response
        return send_file(
            file_stream,
            download_name=file_name,
            as_attachment=True
        )

    except ApiError as e:
        # Handle Dropbox API errors
        return f"An error occurred while downloading the file: {str(e)}"


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
