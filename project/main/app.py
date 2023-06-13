from flask import Flask
from project.main.blueprints.system_routes import system_bp
from project.main.blueprints.dataset_routes import dataset_bp

app = Flask(__name__)

# Registrar los Blueprints
app.register_blueprint(system_bp)
app.register_blueprint(dataset_bp)
app.register_blueprint(record_bp)

if __name__ == '__main__':
    app.run(debug=False, port=9000)
