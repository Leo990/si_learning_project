from flask import Flask
from project.main.blueprints.system_routes import system_bp
from project.main.blueprints.dataset_routes import dataset_bp
from project.main.blueprints.record_routes import record_bp
#Se agrega prediction
from project.main.blueprints.prediction_routes import prediction_bp


app = Flask(__name__)

# Registrar los Blueprints
app.register_blueprint(system_bp)
app.register_blueprint(dataset_bp)
app.register_blueprint(record_bp)
app.register_blueprint(prediction_bp)


if __name__ == '__main__':
    app.run(debug=True, port=9000)
