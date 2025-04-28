from flask import Blueprint, jsonify
from services.metrics_service import get_dataset_info, get_all_algorithms_metrics

metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/metrics/<dataset_id>', methods=['GET'])
def get_metrics(dataset_id):
    try:
        dataset = get_dataset_info(dataset_id)
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        algorithms = get_all_algorithms_metrics(dataset_id)
        return jsonify({
            "dataset": dataset,
            "algorithms": algorithms
        }), 200

    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500
