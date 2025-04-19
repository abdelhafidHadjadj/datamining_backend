from flask import Blueprint, request, jsonify
from services.clustering_service import perform_clustering

clustering_bp = Blueprint("clustering", __name__, url_prefix="/clustering")


@clustering_bp.route("/unsupervised/<dataset_id>", methods=["POST"])
def cluster_dataset(dataset_id):
    try:
        data = request.json or {}
        method = data.get("method").lower()
        columns = data.get("columns")
        n_clusters = data.get("n_clusters")
        eps = data.get("eps")
        min_samples = data.get("min_samples")
        print(request.json)
        result = perform_clustering(
            dataset_id=dataset_id,
            method=method,
            columns=columns,
            n_clusters=n_clusters,
            eps=eps,
            min_samples=min_samples
        )

        return jsonify({
            "message": f"Clustering '{method}' applied successfully",
            "result": result
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
