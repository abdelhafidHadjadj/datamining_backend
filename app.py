from flask import Flask
from flask_cors import CORS
from routes.dataset_routes import dataset_bp
from routes.clustering_routes import clustering_bp
from routes.metrics_routes import metrics_bp
app = Flask(__name__)
CORS(app)

app.register_blueprint(metrics_bp)
app.register_blueprint(dataset_bp)
app.register_blueprint(clustering_bp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
